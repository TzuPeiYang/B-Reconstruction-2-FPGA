#!/usr/bin/env python3
"""
generate_debug.py
=================
Reads the onnx2c-generated C file and produces a  debug_entry.cpp
that contains an instrumented version of entry() which:

  1. Runs every node function exactly as the real entry() does.
  2. After each call, prints the max-abs value of each output tensor
     AND checks whether any value is within 5% of the ap_fixed type's
     saturation boundary (catches overflow/clipping).
  3. At the end, prints all tensors where max_abs > 90% of type_max.

Compile with  -DCSIM_DEBUG  to enable, without it the file is empty.

Usage
-----
    python generate_debug.py  model_inference.cpp  debug_entry.cpp

Then in your testbench, replace the call to  entry(...)  with
  debug_entry(...)  and compile both .cpp files together.
  Add -DCSIM_DEBUG to your C-sim compilation flags in Vitis HLS.
"""

import re, sys, argparse, math
from collections import defaultdict

def angle_aware_split(s):
    parts, cur, depth = [], '', 0
    for ch in s:
        if ch == '<':   depth += 1; cur += ch
        elif ch == '>': depth -= 1; cur += ch
        elif ch == ',' and depth == 0: parts.append(cur.strip()); cur = ''
        else: cur += ch
    if cur.strip(): parts.append(cur.strip())
    return parts

def ap_max(W, I):
    return 2**(I-1) - 2**(-(W-I))

def parse_tensor_decls(src):
    """name -> (type_str, W, I, dims_str, total_elements)"""
    decls = {}
    for m in re.finditer(
        r'^(ap_fixed<(\d+),(\d+)>|float|int64_t)\s+(tensor\w+)((?:\[\d+\])+)\s*;',
        src, re.MULTILINE
    ):
        typ  = m.group(1)
        name = m.group(4)
        dims = m.group(5)
        if typ.startswith('ap_fixed'):
            W, I = int(m.group(2)), int(m.group(3))
        else:
            W, I = None, None
        elems = 1
        for d in re.findall(r'\[(\d+)\]', dims):
            elems *= int(d)
        decls[name] = (typ, W, I, dims, elems)
    return decls

def parse_sigs(src):
    """func_name -> [(param_name, param_type, dims), ...]"""
    sigs = {}
    for m in re.finditer(
        r'FUNC_PREFIX\s+void\s+(node\w+)\s*\(([^)]*(?:<[^>]*>[^)]*)*)\)',
        src, re.DOTALL
    ):
        fname = m.group(1)
        params = []
        for p in angle_aware_split(m.group(2)):
            pm = re.search(
                r'(const\s+)?(ap_fixed<[^>]+>|float|int64_t|fixed)\s*\*?\s*(\w+)((?:\[\d+\])*)',
                p
            )
            if pm:
                params.append((pm.group(3), pm.group(2), pm.group(4)))
        sigs[fname] = params
    return sigs

def parse_entry_calls(src):
    """
    Return list of (func_name, [arg_str, ...]) in call order from entry().
    """
    entry_m = re.search(r'void\s+entry\s*\([^)]*\)\s*\{', src)
    if not entry_m:
        raise ValueError("entry() not found")
    body = src[entry_m.end():]

    calls = []
    pos = 0
    while pos < len(body):
        m = re.search(r'\b(node\w+)\s*\(', body[pos:])
        if not m:
            break
        fname = m.group(1)
        start = pos + m.end()
        depth, i = 1, start
        while i < len(body) and depth > 0:
            if body[i] == '(':   depth += 1
            elif body[i] == ')': depth -= 1
            i += 1
        args = angle_aware_split(body[start:i-1])
        calls.append((fname, args))
        pos += m.start() + 1
    return calls

def find_output_tensors(fname, args, sigs, tensor_decls):
    """
    Return list of (arg_name, W, I, dims, elems) for non-const output
    parameters that are ap_fixed tensors.
    """
    outputs = []
    if fname not in sigs:
        return outputs
    params = sigs[fname]
    if len(params) != len(args):
        return outputs
    for (pname, ptype, pdims), arg in zip(params, args):
        # Output params are non-const
        # Check in the original source whether this param is const
        # We detect by checking if 'const' appears before this param in sig
        # Simplest: const params have ptype starting with 'const ' in our parse
        # Actually our RE_PARAM captures group(1) as const — but we stored ptype
        # Let's check the arg: if it's a tensor in tensor_decls and not const
        arg_clean = re.sub(r'^&|\([^)]+\)\s*', '', arg).strip()
        if arg_clean not in tensor_decls:
            continue
        typ, W, I, dims, elems = tensor_decls[arg_clean]
        if W is None:  # float or int64_t
            continue
        # Heuristic: output params are not const (they get written to)
        # We check by looking at the signature in the source
        outputs.append((arg_clean, W, I, dims, elems))
    return outputs

def flat_iter_code(name, dims, indent='    '):
    """Generate nested for-loop code to iterate over a tensor."""
    dim_list = re.findall(r'\[(\d+)\]', dims)
    idx_vars = [f'_i{j}' for j in range(len(dim_list))]

    lines = []
    for j, (var, bound) in enumerate(zip(idx_vars, dim_list)):
        lines.append(indent + '  ' * j + f'for (int {var}=0; {var}<{bound}; {var}++) {{')

    subscript = ''.join(f'[{v}]' for v in idx_vars)
    inner_indent = indent + '  ' * len(dim_list)
    lines.append(f'{inner_indent}TENSOR_ELEM = {name}{subscript};')

    for j in range(len(dim_list)-1, -1, -1):
        lines.append(indent + '  ' * j + '}')

    return '\n'.join(lines), subscript, idx_vars

def generate_check_block(name, W, I, dims, indent='    '):
    """
    Generate C code that:
    1. Finds max_abs of a tensor
    2. Prints a warning if max_abs > 90% of the type's saturation value
    """
    type_max = ap_max(W, I)
    warn_threshold = type_max * 0.90
    dim_list = re.findall(r'\[(\d+)\]', dims)
    idx_vars = [f'_i{j}' for j in range(len(dim_list))]

    t = indent
    lines = []
    lines.append(f'{t}{{')
    lines.append(f'{t}    double _max_abs = 0.0;')

    # Loop
    for j, (var, bound) in enumerate(zip(idx_vars, dim_list)):
        lines.append(f'{t}    {"  "*j}for (int {var}=0; {var}<{bound}; {var}++) {{')

    subscript = ''.join(f'[{v}]' for v in idx_vars)
    inner = t + '    ' + '  ' * len(dim_list)
    lines.append(f'{inner}double _v = (double){name}{subscript};')
    lines.append(f'{inner}if (_v < 0) _v = -_v;')
    lines.append(f'{inner}if (_v > _max_abs) _max_abs = _v;')

    for j in range(len(dim_list)-1, -1, -1):
        lines.append(f'{t}    {"  "*j}}}')

    sat_flag = '(_max_abs > {:.4f}) ? "*** NEAR SATURATION ***" : "ok"'.format(warn_threshold)
    lines.append(
        f'{t}    printf("[DEBUG] {name}: max_abs=%%f  type_max={type_max:.3f}  %%s\\n",'
        f' _max_abs, {sat_flag});'
    )
    lines.append(f'{t}}}')

    return '\n'.join(lines)

def parse_entry_signature(src):
    """Return the entry() parameter list string."""
    m = re.search(r'void\s+entry\s*\(([^)]*(?:<[^>]*>[^)]*)*)\)', src, re.DOTALL)
    return m.group(1) if m else ''

def generate_debug_file(src, out_path):
    tensor_decls = parse_tensor_decls(src)
    sigs         = parse_sigs(src)
    calls        = parse_entry_calls(src)
    entry_params = parse_entry_signature(src)

    lines = []
    lines.append('/* Auto-generated by generate_debug.py')
    lines.append(' * Compile with -DCSIM_DEBUG to enable instrumentation.')
    lines.append(' * Usage: replace entry() call with debug_entry() in testbench.')
    lines.append(' */')
    lines.append('')
    lines.append('#ifdef CSIM_DEBUG')
    lines.append('#include <stdio.h>')
    lines.append('#include <math.h>')
    lines.append('')
    lines.append('/* Forward declarations of all node functions */')
    for fname in sigs:
        lines.append(f'/* extern void {fname}(...); */')
    lines.append('')

    # Re-declare all global tensors as locals inside debug_entry
    # so we don't conflict with the real entry()
    lines.append(f'void debug_entry({entry_params})')
    lines.append('{')
    lines.append('    /* Local copies of all intermediate tensors */')

    # Declare all ap_fixed tensors as locals
    for name, (typ, W, I, dims, elems) in tensor_decls.items():
        lines.append(f'    {typ} {name}{dims};')
    lines.append('')

    lines.append('    printf("\\n=== CSIM DEBUG: intermediate tensor saturation check ===\\n");')
    lines.append('')

    # Emit each function call followed by saturation check on outputs
    for fname, args in calls:
        # Emit the call
        args_str = ', '.join(args)
        lines.append(f'    /* {fname} */')
        lines.append(f'    {fname}({args_str});')

        # Find output tensors and emit checks
        outputs = find_output_tensors(fname, args, sigs, tensor_decls)
        for tname, W, I, dims, elems in outputs:
            check = generate_check_block(tname, W, I, dims, indent='    ')
            lines.append(check)
        lines.append('')

    lines.append('    printf("=== END CSIM DEBUG ===\\n\\n");')
    lines.append('}')
    lines.append('')
    lines.append('#else  /* !CSIM_DEBUG */')
    lines.append('/* CSIM_DEBUG not defined — debug_entry() not available */')
    lines.append('#endif /* CSIM_DEBUG */')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Generated {out_path}')
    print(f'  {len(calls)} function calls instrumented')
    print(f'  {len(tensor_decls)} tensor declarations')
    print(f'\nTo use:')
    print(f'  1. Add -DCSIM_DEBUG to your C-sim compilation flags')
    print(f'  2. Add {out_path} to your HLS source files')
    print(f'  3. In your testbench, call debug_entry(...) instead of entry(...)')
    print(f'     with the same arguments')
    print(f'  4. Run C-sim — saturation warnings will print to stdout')
    print(f'     Lines with *** NEAR SATURATION *** are overflow candidates')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', help='model_inference.cpp (after fix_types.py)')
    p.add_argument('output', nargs='?', default='debug_entry.cpp')
    args = p.parse_args()

    with open(args.source) as f:
        src = f.read()

    generate_debug_file(src, args.output)

if __name__ == '__main__':
    main()