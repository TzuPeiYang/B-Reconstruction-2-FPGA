#!/usr/bin/env python3
"""
apply_tree_reduction.py
=======================
Reads a Vitis HLS C/C++ file and rewrites every fixed-bound accumulation loop
into a binary-tree reduction, which HLS can schedule with log2(N) latency
instead of N sequential additions.

Patterns detected and rewritten
---------------------------------
  Pattern A - multiply-accumulate (Conv, MatMul, Gemm inner loops):
      fixed acc = bias[m];
      for (unsigned c = 0; c < 8; c++) {
      #pragma HLS UNROLL factor=8
          acc += x[b][c][h][i] * w[m][c][0][0];
      }
      y[b][m][h][i] = acc;

      --> buf[8] of products, then acc = bias[m] + tree_sum_8(buf)

  Pattern B - sum-only (ReduceSum, ReduceMean inner loops):
      fixed acc = 0.0;
      for (unsigned i3 = 0; i3 < 16; i3++) {
          acc += x[i0][i1][i2][i3];
      }
      y[...] = acc;

      --> buf[16] of values, then acc = tree_sum_16(buf)

Rules / conservatism
---------------------
  - Loop variable must appear in the accumulation expression.
  - N (upper bound) must be a literal integer.
  - Loop body must contain exactly ONE non-pragma statement.
  - If any of the above fail, the loop is left completely unchanged
    and a warning is printed.

Generated helpers
------------------
  tree_sum_N() inline functions are emitted once, just before the first
  function definition in the file. Non-power-of-2 N values are padded
  to the next power-of-2 with zeros.

Usage
------
    python apply_tree_reduction.py  input.h  output.h

"""

import re
import sys
import math
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def make_tree_sum(N, fixed_type='fixed'):
    """Return the C source for a tree_sum_N() inline function."""
    P = next_pow2(N)
    depth = int(math.log2(P)) if P > 1 else 0

    lines = []
    lines.append(f'/* Tree-reduction: sum {N} {fixed_type} values, depth={depth} */')
    lines.append(f'static inline {fixed_type} tree_sum_{N}(const {fixed_type} v[{N}]) {{')
    lines.append(f'#pragma HLS INLINE')

    if N == 1:
        lines.append(f'    return v[0];')
        lines.append(f'}}')
        return '\n'.join(lines)

    if N == 2:
        lines.append(f'    return v[0] + v[1];')
        lines.append(f'}}')
        return '\n'.join(lines)

    # Padding buffer when N is not a power-of-2
    if P > N:
        lines.append(f'    {fixed_type} _p[{P}];')
        lines.append(f'#pragma HLS ARRAY_PARTITION variable=_p complete')
        lines.append(f'    for (int _i=0;_i<{N};_i++){{')
        lines.append(f'#pragma HLS UNROLL')
        lines.append(f'        _p[_i]=v[_i];}}')
        lines.append(f'    for (int _i={N};_i<{P};_i++){{')
        lines.append(f'#pragma HLS UNROLL')
        lines.append(f'        _p[_i]=({fixed_type})0;}}')
        src = '_p'
    else:
        src = 'v'

    # Intermediate levels: P/2, P/4, ..., 2
    # We stop at size-2 and return the final sum directly.
    sizes = []
    s = P // 2
    while s >= 2:
        sizes.append(s)
        s //= 2

    level_names = []
    for idx, sz in enumerate(sizes):
        nm = f'_l{idx}'
        lines.append(f'    {fixed_type} {nm}[{sz}];')
        lines.append(f'#pragma HLS ARRAY_PARTITION variable={nm} complete')
        level_names.append(nm)

    prev = src
    for nm, sz in zip(level_names, sizes):
        lines.append(f'    for (int _i=0;_i<{sz};_i++){{')
        lines.append(f'#pragma HLS UNROLL')
        lines.append(f'        {nm}[_i]={prev}[_i*2]+{prev}[_i*2+1];}}')
        prev = nm

    if level_names:
        last = level_names[-1]
        lines.append(f'    return {last}[0]+{last}[1];')
    else:
        lines.append(f'    return {src}[0]+{src}[1];')

    lines.append(f'}}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# for (...; var < N; var++)  –  catches common HLS loop styles
RE_FOR = re.compile(
    r'for\s*\(\s*(?:unsigned\s+|int\s+|uint32_t\s+|uint64_t\s+)?'
    r'(?P<var>\w+)\s*=\s*0\s*;'
    r'\s*(?P=var)\s*<\s*(?P<N>\d+)\s*;'
    r'\s*(?P=var)\s*(?:\+\+|\+=\s*1)\s*\)'
)

# acc += lhs * rhs ;  (trailing } allowed – generated code puts } on same line)
RE_MAC = re.compile(
    r'^(?P<acc>\w+)\s*\+=\s*(?P<lhs>.+?)\s*\*\s*(?P<rhs>.+?)\s*;'
)

# acc += expr ;   (no * in expr, trailing } allowed)
RE_SUM = re.compile(
    r'^(?P<acc>\w+)\s*\+=\s*(?P<expr>[^*]+?)\s*;'
)

# [type] acc = val ;
RE_INIT = re.compile(
    r'^(?P<indent>\s*)(?:(?P<type>(?:ap_)?fixed\b[^;=]*?)\s+)?'
    r'(?P<acc>\w+)\s*=\s*(?P<val>[^;]+?)\s*;'
)

# Values we treat as zero initialisation
ZERO_VALS = {'0', '0.0', '0.0f', '(fixed)0', '(fixed)0.0f',
             '(fixed)0.0', '(ap_fixed<40,21>)0'}


# ---------------------------------------------------------------------------
# Parse the text inside a for-loop body.
# Returns (mode, acc, lhs_or_expr, rhs_or_None) or None.
# ---------------------------------------------------------------------------
def parse_body(body_text, loop_var):
    content = []
    for line in body_text.split('\n'):
        # Strip trailing closing braces (generated code puts } on same line as statement)
        s = line.strip().rstrip('}').strip()
        if not s or s.startswith('#pragma'):
            continue
        content.append(s)

    if len(content) != 1:
        return None  # multiple statements or empty body – skip

    stmt = content[0]

    m = RE_MAC.match(stmt)
    if m:
        acc = m.group('acc')
        lhs = m.group('lhs').strip()
        rhs = m.group('rhs').strip()
        if loop_var not in lhs and loop_var not in rhs:
            return None
        return ('mac', acc, lhs, rhs)

    m = RE_SUM.match(stmt)
    if m and '*' not in m.group('expr'):
        acc = m.group('acc')
        expr = m.group('expr').strip()
        if loop_var not in expr:
            return None
        return ('sum', acc, expr, None)

    return None


# ---------------------------------------------------------------------------
# Build the replacement block for one rewritten loop.
# ---------------------------------------------------------------------------
def build_replacement(indent, init_line, loop_var, N, mode, acc, lhs, rhs):
    lines = []

    # Re-declare acc without initialisation (assigned at end of block)
    if init_line is not None:
        m = RE_INIT.search(init_line.strip())
        if m and m.group('type'):
            typ = m.group('type').strip()
            lines.append(f'{indent}{typ} {acc};')
        # If no type token, acc was already declared elsewhere – no new decl

    # Temporary buffer
    buf = f'_buf_{acc}'
    lines.append(f'{indent}fixed {buf}[{N}];')
    lines.append(f'{indent}#pragma HLS ARRAY_PARTITION variable={buf} complete')

    # Unrolled fill  (rename loop var with leading _ to avoid shadowing)
    lv2 = f'_{loop_var}'
    lines.append(f'{indent}for (int {lv2}=0; {lv2}<{N}; {lv2}++) {{')
    lines.append(f'{indent}#pragma HLS UNROLL')
    if mode == 'mac':
        lhs_i = lhs.replace(loop_var, lv2)
        rhs_i = rhs.replace(loop_var, lv2)
        lines.append(f'{indent}    {buf}[{lv2}] = {lhs_i} * {rhs_i}; }}')
    else:
        expr_i = lhs.replace(loop_var, lv2)
        lines.append(f'{indent}    {buf}[{lv2}] = {expr_i}; }}')

    # Determine init value
    init_val = '0'
    if init_line is not None:
        m = RE_INIT.search(init_line.strip())
        if m:
            init_val = m.group('val').strip()

    tree_call = f'tree_sum_{N}({buf})'
    if init_val in ZERO_VALS:
        rhs_full = tree_call
    else:
        rhs_full = f'{init_val} + {tree_call}'

    lines.append(f'{indent}{acc} = {rhs_full};')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------
def transform(src):
    lines = src.split('\n')
    out   = []
    needed_N  = OrderedDict()
    rewrites  = 0
    skips     = 0

    i = 0
    while i < len(lines):
        raw = lines[i]

        m_for = RE_FOR.search(raw)
        if not m_for:
            out.append(raw)
            i += 1
            continue

        loop_var = m_for.group('var')
        N        = int(m_for.group('N'))
        indent   = raw[:len(raw) - len(raw.lstrip())]

        # Find matching { }
        j       = i
        depth   = 0
        in_body = False
        body_lines_raw = []
        found_close    = False

        while j < len(lines):
            line = lines[j]
            opens  = line.count('{')
            closes = line.count('}')
            if not in_body and opens > 0:
                in_body = True
            # Collect every line after the for-header, INCLUDING the closing-brace
            # line (which may contain the last statement: "acc+=...;}" )
            if in_body and j > i:
                body_lines_raw.append(line)
            depth += opens - closes
            if in_body and depth == 0:
                found_close = True
                break
            j += 1

        if not found_close:
            out.append(raw)
            i += 1
            skips += 1
            continue

        body_close = j
        body_text  = '\n'.join(body_lines_raw)

        result = parse_body(body_text, loop_var)
        if result is None:
            out.append(raw)
            i += 1
            skips += 1
            continue

        mode, acc, lhs, rhs = result

        # Look back in output for an init line for `acc` (within 5 lines)
        init_line = None
        for k in range(len(out)-1, max(len(out)-6, -1), -1):
            candidate = out[k]
            if not candidate.strip():
                continue
            m_init = RE_INIT.search(candidate.strip())
            if m_init and m_init.group('acc') == acc:
                init_line = candidate
                out[k] = ''   # blank – we will re-emit in replacement
                break

        replacement = build_replacement(
            indent, init_line, loop_var, N, mode, acc, lhs, rhs
        )
        out.append(replacement)
        needed_N[N] = True
        rewrites += 1

        # Skip all lines consumed by this loop
        i = body_close + 1

    # ---- Inject tree_sum helpers before first function definition ----
    helper_block = [
        '',
        '/* ================================================================',
        ' * TREE REDUCTION HELPERS (auto-generated by apply_tree_reduction.py)',
        ' *',
        ' * Each tree_sum_N() reduces N ap_fixed values in log2(N) adder stages.',
        ' * Intermediate arrays use ARRAY_PARTITION complete so HLS keeps them',
        ' * in flip-flop registers with zero memory access latency.',
        ' * Non-power-of-2 sizes are zero-padded to the next power of 2.',
        ' * ================================================================ */',
        '',
    ]
    for N_val in sorted(needed_N.keys()):
        helper_block.append(make_tree_sum(N_val))
        helper_block.append('')

    # Insert just before the first line that looks like a function definition
    insert_idx = 0
    for idx, line in enumerate(out):
        s = line.strip()
        if (s.startswith('FUNC_PREFIX') or
                s.startswith('static inline') or
                s.startswith('static void') or
                re.match(r'^void\s+\w+\s*\(', s)):
            insert_idx = idx
            break

    for offset, hl in enumerate(helper_block):
        out.insert(insert_idx + offset, hl)

    result_src = '\n'.join(out)
    print(f'  Loops rewritten : {rewrites}')
    print(f'  Loops unchanged : {skips}  (complex body, variable bound, or loop var not in expr)')
    print(f'  Helpers emitted : tree_sum_N for N = {sorted(needed_N.keys())}')
    return result_src


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python apply_tree_reduction.py  input.h  output.h')
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path, 'r') as f:
        src = f.read()

    print(f'Input  : {in_path}  ({src.count(chr(10))} lines)')
    result = transform(src)
    print(f'Output : {out_path}  ({result.count(chr(10))} lines)')

    with open(out_path, 'w') as f:
        f.write(result)

    print('Done.')