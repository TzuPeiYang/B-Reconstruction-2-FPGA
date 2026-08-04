#!/usr/bin/env python3
"""
fix_types.py  — all type fixes for a self-contained onnx2c C file.
Run AFTER rewrite_types.py.  Usage:
    python fix_types.py  input.cpp  output.cpp  [--log log.txt] [--dry-run]
"""
import argparse, re, sys

WEIGHT_TYPE = 'ap_fixed<18,5>'
MSE_TYPE    = 'ap_fixed<17,4>'
PF_TYPES = {
    'tensor_pf_points':   'ap_fixed<18,5>',
    'tensor_pf_features': 'ap_fixed<18,5>',
    'tensor_pf_mask':     'ap_fixed<17,4>',
}

# ── helpers ──────────────────────────────────────────────────────────────────

def angle_aware_split(s):
    parts, cur, depth = [], '', 0
    for ch in s:
        if ch == '<':   depth += 1; cur += ch
        elif ch == '>': depth -= 1; cur += ch
        elif ch == ',' and depth == 0: parts.append(cur.strip()); cur = ''
        else: cur += ch
    if cur.strip(): parts.append(cur.strip())
    return parts

def replace_param_type(params_str, param_name, new_type):
    return re.sub(
        r'((?:const\s+)?)(?:float|ap_fixed<[^>]+>|int64_t)'
        r'(\s+\*?\s*' + re.escape(param_name) + r'(?:\s*[\[,\)]|\s*$))',
        r'\g<1>' + new_type + r'\2',
        params_str
    )

def replace_func_signature(src, func_name, replacements, log):
    RE = re.compile(
        r'(FUNC_PREFIX\s+void\s+' + re.escape(func_name)
        + r'\s*\()([^)]*(?:<[^>]*>[^)]*)*)(\)(?=\s*\n?\{))',
        re.DOTALL
    )
    m = RE.search(src)
    if not m: return src, 0
    prefix, params, close = m.group(1), m.group(2), m.group(3)
    new_params, count = params, 0
    for pname, new_type in replacements:
        old = new_params
        new_params = replace_param_type(new_params, pname, new_type)
        if new_params != old:
            log.append(f'    {func_name} [{pname}] -> {new_type}')
            count += 1
    if count:
        src = src[:m.start()] + prefix + new_params + close + src[m.end():]
    return src, count

def build_call_graph(src):
    """Returns (sigs, tensor_map)."""
    sigs = {}
    for m in re.finditer(
        r'FUNC_PREFIX\s+void\s+(node\w+)\s*\(([^)]*(?:<[^>]*>[^)]*)*)\)',
        src, re.DOTALL
    ):
        params = []
        for p in angle_aware_split(m.group(2)):
            pm = re.search(
                r'(const\s+)?(ap_fixed<[^>]+>|float|int64_t|fixed)\s*\*?\s*(\w+)((?:\[\d+\])*)', p
            )
            if pm:
                params.append({'type': pm.group(2), 'name': pm.group(3)})
        sigs[m.group(1)] = params

    entry_m = re.search(r'void\s+entry\s*\([^)]*\)\s*\{', src)
    body = src[entry_m.end():] if entry_m else src
    calls, pos = {}, 0
    while pos < len(body):
        m = re.search(r'\b(node\w+)\s*\(', body[pos:])
        if not m: break
        fname, start = m.group(1), pos + m.end()
        depth, i = 1, start
        while i < len(body) and depth > 0:
            if body[i] == '(':   depth += 1
            elif body[i] == ')': depth -= 1
            i += 1
        calls[fname] = angle_aware_split(body[start:i-1])
        pos += m.start() + 1

    tensor_map = {}
    for fname, args in calls.items():
        if fname not in sigs: continue
        params = sigs[fname]
        if len(params) != len(args): continue
        for arg, param in zip(args, params):
            arg_clean = re.sub(r'^&|\([^)]+\)\s*', '', arg).strip()
            tensor_map.setdefault(arg_clean, []).append(
                {'func': fname, 'param_name': param['name'], 'param_type': param['type']}
            )
    return sigs, tensor_map

def retype_tensor(src, tensor_name, new_type, log, sigs=None, tensor_map=None):
    """Retype a declaration and cascade to all consuming function parameters."""
    if sigs is None or tensor_map is None:
        sigs, tensor_map = build_call_graph(src)
    changes = []
    # 1. Declaration
    new_src = re.sub(
        r'(?:ap_fixed<[^>]+>|float|int64_t)(\s+' + re.escape(tensor_name) + r'\b)',
        new_type + r'\1', src
    )
    if new_src != src:
        changes.append(f'DECL  {tensor_name} -> {new_type}')
        src = new_src
    # 2. Consuming function parameters
    for c in tensor_map.get(tensor_name, []):
        fname, pname, old_ptype = c['func'], c['param_name'], c['param_type']
        if old_ptype == new_type or old_ptype == 'int64_t': continue
        src, n = replace_func_signature(src, fname, [(pname, new_type)], log)
        if n:
            changes.append(f'PARAM {fname} [{pname}]: {old_ptype} -> {new_type}')
    for c in changes:
        log.append(f'    {c}')
    return src, len(changes)

# ── fixes ────────────────────────────────────────────────────────────────────

def fix_accumulator_typedef(src, log):
    """ap_fixed<32,19>: max=32767, 24 frac bits — covers fusion Conv accumulator."""
    log.append('FIX 1: accumulator typedef -> ap_fixed<32,19>')
    result = re.sub(
        r'typedef\s+ap_fixed<[^;]+fixed\s*;(?:\s*/\*[^*]*\*/)?',
        'typedef ap_fixed<32,19> fixed;  /* accumulator: max=32767, 24 frac bits */',
        src
    )
    changed = result != src
    print(f'  Accumulator typedef  : {"1" if changed else "0 (not found)"}')
    if changed: log.append('    -> ap_fixed<32,19>')
    return result

def fix_max_param_types(src, log):
    """in_0 and in_1 of node__mod_Max must be same type (ap_fixed<32,19>)."""
    log.append('FIX 2: node__mod_Max in_0/in_1 -> ap_fixed<32,19>')
    count = 0
    for pname in ('in_0', 'in_1'):
        src, n = replace_func_signature(src, 'node__mod_Max',
                                        [(pname, 'ap_fixed<32,19>')], log)
        count += n
    result = re.sub(
        r'((?:static\s+const\s+)?)ap_fixed<[^>]+>(\s+tensor__mod_Constant_2_output_0\b)',
        r'\g<1>ap_fixed<32,19>\2', src
    )
    if result != src:
        log.append('    tensor__mod_Constant_2_output_0 -> ap_fixed<32,19>')
        count += 1; src = result
    print(f'  Max param types      : {count}')
    return src

def fix_pointer_cast_type_mismatch(src, log):
    """ap_fixed<W,I> *p = (fixed*)x: after widening fixed, sizes differ."""
    log.append('FIX 3: pointer cast LHS/cast-target mismatch')
    count = [0]
    def replace(m):
        lhs, name, spacing, cast, rhs = (
            m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        )
        if lhs == cast: return m.group(0)
        count[0] += 1
        return f'{lhs} *{name}{spacing}({lhs}*){rhs}'
    result = re.sub(
        r'(ap_fixed<[^>]+>)\s+\*(\w+)(\s*=\s*)\((\w+)\*\)(\w+)',
        replace, src
    )
    log.append(f'    {count[0]} casts corrected')
    print(f'  Pointer cast fix     : {count[0]}')
    return result

def fix_mse_type(src, log):
    """tensor_MSE and fc_out Y -> ap_fixed<17,4> to match testbench."""
    log.append(f'FIX 4: tensor_MSE and fc_out Y -> {MSE_TYPE}')
    count = 0
    result = re.sub(
        r'(?:float|ap_fixed<[^>]+>)(\s+tensor_MSE(?:\[\d+\])*)',
        lambda m: MSE_TYPE + m.group(1), src
    )
    if result != src:
        count += 1; log.append(f'    entry() tensor_MSE -> {MSE_TYPE}'); src = result
    result = re.sub(
        r'(FUNC_PREFIX\s+void\s+node__fc_out_fc_out_0_Gemm\s*\()([^)]*)'
        r'(?:float|ap_fixed<[^>]+>)(\s+Y(?:\[\d+\])*\s*\))',
        lambda m: m.group(1) + m.group(2) + MSE_TYPE + m.group(3),
        src, flags=re.DOTALL
    )
    if result != src:
        count += 1; log.append(f'    fc_out_Gemm Y -> {MSE_TYPE}'); src = result
    print(f'  MSE type             : {count}')
    return src

def fix_entry_params(src, log):
    log.append('FIX 5: entry() pf_* params float -> ap_fixed')
    count = 0
    for tensor, new_type in PF_TYPES.items():
        result = re.sub(
            r'(const\s+)float(\s+' + re.escape(tensor) + r'(?:\[\d+\])*)',
            r'\g<1>' + new_type + r'\2', src
        )
        if result != src:
            count += 1; log.append(f'    entry() {tensor} -> {new_type}'); src = result
    print(f'  entry() pf_* params  : {count}')
    return src

def fix_pf_func_params(src, log):
    log.append('FIX 6: pf_*-receiving function parameters')
    count = 0
    for func_name, pname, new_type in [
        ('node__mod_Mul',    'A', 'ap_fixed<18,5>'),
        ('node__mod_Mul',    'B', 'ap_fixed<17,4>'),
        ('node__mod_Mul_1',  'A', 'ap_fixed<18,5>'),
        ('node__mod_Mul_1',  'B', 'ap_fixed<17,4>'),
        ('node__mod_Equal',  'A', 'ap_fixed<17,4>'),
        ('node__mod_Cast_1', 'input', 'ap_fixed<17,4>'),
    ]:
        result = re.sub(
            r'(FUNC_PREFIX\s+void\s+' + re.escape(func_name) + r'\s*\([^)]*?)'
            r'((?:const\s+)?)float(\s+' + re.escape(pname) + r'(?:\[\d+\])*)',
            r'\1\g<2>' + new_type + r'\3', src, flags=re.DOTALL
        )
        if result != src:
            count += 1; log.append(f'    {func_name} [{pname}] -> {new_type}'); src = result
    print(f'  pf_* func params     : {count}')
    return src

def fix_transpose_matmul_int64(src, log):
    log.append('FIX 7: Transpose/MatMul/Pow int64_t -> ap_fixed<32,19>')
    count = 0
    for stage in ['0', '1', '2']:
        for func_name, param_name in [
            (f'node__mod_edge_convs_{stage}_Transpose', 'input'),
            (f'node__mod_edge_convs_{stage}_MatMul',    'B'),
            (f'node__mod_edge_convs_{stage}_Pow',       'A'),
        ]:
            result = re.sub(
                r'(FUNC_PREFIX\s+void\s+' + re.escape(func_name) + r'\s*\([^)]*?)'
                r'((?:const\s+)?)int64_t(\s+' + re.escape(param_name) + r'(?:\[\d+\])*)',
                r'\1\g<2>ap_fixed<32,19>\3', src, flags=re.DOTALL
            )
            if result != src:
                count += 1; log.append(f'    {func_name} [{param_name}]: int64_t -> ap_fixed<32,19>'); src = result
    print(f'  Transpose/MatMul/Pow : {count}')
    return src

def fix_equal_scalar_type(src, log):
    log.append('FIX 8: Equal *B + Constant_output_0 -> ap_fixed<17,4>')
    count = 0
    src, n = replace_func_signature(src, 'node__mod_Equal', [('B', 'ap_fixed<17,4>')], log)
    count += n
    result = re.sub(
        r'((?:static\s+const\s+)?)ap_fixed<[^>]+>(\s+tensor__mod_Constant_output_0\b)',
        r'\g<1>ap_fixed<17,4>\2', src
    )
    if result != src:
        log.append('    tensor__mod_Constant_output_0 -> ap_fixed<17,4>'); count += 1; src = result
    print(f'  Equal scalar type    : {count}')
    return src

def fix_cast1_body(src, log):
    log.append('FIX 9: Cast_1 body -> direct element conversion')
    sig_m = re.search(r'FUNC_PREFIX\s+void\s+node__mod_Cast_1\s*\([^)]*'
                       r'(ap_fixed<[^>]+>)\s+output\[', src)
    out_type = sig_m.group(1) if sig_m else 'ap_fixed<17,4>'
    RE_BODY = re.compile(
        r'(FUNC_PREFIX\s+void\s+node__mod_Cast_1\s*\([^)]*\)\s*\{).*?(?=\n\})',
        re.DOTALL
    )
    new_body = (f'\n#pragma HLS INLINE OFF\n\n'
                f'\t/* Cast: direct element-wise conversion */\n'
                f'\tfor (unsigned i=0; i<35; i++)\n'
                f'\t\toutput[0][0][i] = ({out_type})input[0][0][i];\n')
    result = RE_BODY.sub(r'\1' + new_body, src)
    changed = result != src
    print(f'  Cast_1 body          : {"1" if changed else "0 (not found)"}')
    if changed: log.append(f'    -> {out_type}')
    return result

def fix_float_pointer_casts(src, log):
    log.append('FIX 10: (float*) -> (fixed*)')
    count = [0]
    def repl(m): count[0] += 1; return '(fixed*)'
    result = re.sub(r'\(float\*\)', repl, src)
    log.append(f'    {count[0]} occurrences')
    print(f'  (float*) casts       : {count[0]}')
    return result

def fix_weight_params(src, log):
    log.append('FIX 11: inline weight decls + BN/Gemm sig params')
    decl_count = 0
    def retype_decl(m):
        nonlocal decl_count
        qualifiers, name, rest = m.group(1), m.group(2), m.group(3)
        new_type = 'ap_fixed<32,19>' if 'Constant_2_output_0' in name else WEIGHT_TYPE
        decl_count += 1
        log.append(f'    DECL {name}: float -> {new_type}')
        return f'{qualifiers}{new_type} {name}{rest}'
    src = re.sub(
        r'((?:static\s+const\s+)?)float\s+(tensor\w+)((?:\[\d+\])+\s*=)',
        retype_decl, src
    )
    sig_count = 0
    for func_name, params in [
        ('node__mod_bn_fts_BatchNormalization',                 ['scale','bias','mean','var']),
        ('node__mod_edge_convs_0_output_bn_BatchNormalization', ['scale','bias','mean','var']),
        ('node__mod_edge_convs_1_output_bn_BatchNormalization', ['scale','bias','mean','var']),
        ('node__mod_edge_convs_2_output_bn_BatchNormalization', ['scale','bias','mean','var']),
        ('node__mod_fc_fc_0_fc_0_0_Gemm',                      ['B','C']),
        ('node__fc_out_fc_out_0_Gemm',                         ['B','C']),
    ]:
        src, n = replace_func_signature(src, func_name,
                                        [(p, WEIGHT_TYPE) for p in params], log)
        sig_count += n
    print(f'  weight decls         : {decl_count}')
    print(f'  weight sig params    : {sig_count}')
    return src

def fix_add_output_type(src, log):
    log.append('FIX 12: tensor__mod_Add_output_0 int64_t -> ap_fixed<32,19> (cascade)')
    sigs, tm = build_call_graph(src)
    src, n = retype_tensor(src, 'tensor__mod_Add_output_0', 'ap_fixed<32,19>', log, sigs, tm)
    print(f'  Add int64_t bug      : {n}')
    return src

def fix_reducesum_div_types(src, log):
    """ReduceSum_1 holds raw sum (max=325, not post-mean 9.3). Div max=9.3 > 8."""
    log.append('FIX 14: ReduceSum_1 + Div types (cascade to consumers)')
    total = 0
    for tensor_name, new_type in [
        ('tensor__mod_ReduceSum_1_output_0', 'ap_fixed<20,7>'),
        ('tensor__mod_Div_output_0',          'ap_fixed<20,7>'),
    ]:
        sigs, tm = build_call_graph(src)
        src, n = retype_tensor(src, tensor_name, new_type, log, sigs, tm)
        total += n
    print(f'  ReduceSum/Div types  : {total}')
    return src

def fix_masking_constant(src, log):
    log.append('FIX 13: masking constant 1e9 -> 500')
    result = re.sub(
        r'((?:static\s+const\s+)?ap_fixed<[^>]+>\s+tensor__mod_Constant_1_output_0\s*=\s*)'
        r'[\d.eE+]+f?',
        r'\g<1>500.0f', src
    )
    changed = result != src
    print(f'  Masking constant     : {"1" if changed else "0 (not found)"}')
    if changed: log.append('    1e9 -> 500')
    return result


# ── verify ───────────────────────────────────────────────────────────────────

def verify(src):
    lines = src.split('\n')
    issues = []
    for i, l in enumerate(lines):
        if 'FUNC_PREFIX' in l and re.search(r'\bfloat\b', l) and 'entry' not in l:
            if 'fc_out_fc_out_0_Gemm' in l and 'ap_fixed<17,4> Y' in l: continue
            issues.append(f'  [L{i+1}] float in sig: {l.strip()}')
        if '(float*)' in l:
            issues.append(f'  [L{i+1}] (float*) cast: {l.strip()}')
        if 'int64_t' in l and 'tensor__mod_Add_output_0[1][3]' in l:
            issues.append(f'  [L{i+1}] int64_t Add: {l.strip()}')
        m = re.search(r'(ap_fixed<[^>]+>)\s+\*\w+\s*=\s*\((\w+)\*\)', l)
        if m and m.group(1) != m.group(2) and m.group(2) != 'int64_t':
            issues.append(f'  [L{i+1}] ptr mismatch: {l.strip()}')
    print(f'\n  Issues remaining: {len(issues)}')
    for issue in issues: print(issue)
    return len(issues) == 0


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('output', nargs='?', default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--log', default=None)
    args = p.parse_args()
    if args.output is None:
        args.output = re.sub(r'\.\w+$', '', args.source) + '_fixed.cpp'

    with open(args.source) as f:
        src = f.read()
    print(f'Input : {args.source}  ({src.count(chr(10))} lines)\n')

    log = []
    print('Applying fixes...')
    src = fix_accumulator_typedef(src, log)
    src = fix_max_param_types(src, log)
    src = fix_pointer_cast_type_mismatch(src, log)
    src = fix_mse_type(src, log)
    src = fix_entry_params(src, log)
    src = fix_pf_func_params(src, log)
    src = fix_transpose_matmul_int64(src, log)
    src = fix_equal_scalar_type(src, log)
    src = fix_float_pointer_casts(src, log)
    src = fix_weight_params(src, log)
    src = fix_add_output_type(src, log)
    src = fix_reducesum_div_types(src, log)
    src = fix_masking_constant(src, log)

    total = sum(1 for l in log if l.startswith('    '))
    print(f'\nTotal changes: {total}')
    clean = verify(src)

    if args.dry_run:
        print('\n[DRY RUN]')
        for e in log: print(e)
        return

    with open(args.output, 'w') as f:
        f.write(src)
    print(f'\nOutput: {args.output}  ({src.count(chr(10))} lines)')
    if args.log:
        with open(args.log, 'w') as f:
            f.write('\n'.join(log) + '\n')
        print(f'Log   : {args.log}')
    if clean: print('\n[OK] All checks passed.')
    else:      print('\n[WARN] Some issues remain.')

if __name__ == '__main__':
    main()


# ── type audit utility (call independently) ──────────────────────────────────

def audit_types(src, ranges_csv_path=None):
    """
    Print a table of all ap_fixed tensor types with their empirical ranges
    (if ranges.csv is provided) and flag any overflows or mismatches.
    Call directly:  python fix_types.py --audit model_inference.cpp
    """
    import math, csv as csv_mod

    ranges = {}
    if ranges_csv_path:
        with open(ranges_csv_path) as f:
            for row in csv_mod.DictReader(f):
                if float(row['max_abs']) < 1e8:
                    ranges[row['name']] = float(row['max_abs'])

    def onnx_to_c(name):
        if name.startswith('onnx::'):
            return 'tensor_onnx__' + name[6:]
        return 'tensor__' + name.lstrip('/').replace('/','_').replace('.','_')

    def ap_max(W, I): return 2**(I-1) - 2**(-(W-I))

    decls = {}
    for m in re.finditer(r'^(ap_fixed<(\d+),(\d+)>)\s+(tensor\w+)', src, re.MULTILINE):
        decls[m.group(4)] = (m.group(1), int(m.group(2)), int(m.group(3)))

    print(f"\n{'Tensor':<55} {'Type':<16} {'type_max':>10}  {'emp_max':>10}  Status")
    print("-"*105)

    overflows = []
    for onnx_name, emp_max in sorted(ranges.items(), key=lambda x: -x[1]):
        c_name = onnx_to_c(onnx_name)
        if c_name not in decls: continue
        typ, W, I = decls[c_name]
        type_max = ap_max(W, I)
        frac = W - I
        if emp_max > type_max * 1.01:
            status = f'OVERFLOW (needs I>={math.ceil(math.log2(emp_max+1e-9))+1})'
            overflows.append((onnx_name, c_name, typ, emp_max, type_max))
        elif emp_max > type_max * 0.9:
            status = 'near saturation'
        else:
            status = 'ok'
        short = onnx_name[-55:]
        print(f"  {short:<55} {typ:<16} {type_max:>10.2f}  {emp_max:>10.3f}  {status}")

    print(f"\nOverflows: {len(overflows)}")
    for onnx_name, c_name, typ, emp_max, type_max in overflows:
        print(f"  {c_name}: {typ} max={type_max:.1f} but emp={emp_max:.1f}")
    return overflows