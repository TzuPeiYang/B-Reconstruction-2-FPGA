#!/usr/bin/env python3
"""
rewrite_types.py  v3
====================
Rewrites an onnx2c-generated C file (using `float` everywhere) so that
each tensor uses its assigned ap_fixed type from tensor_types.json.

Handles all four categories of float usage:
  1. Global tensor array declarations:
       float tensor__mod_..._output_0[dims];
  2. Static const scalar initialisations (may be split across two lines):
       static const float tensor__mod_Constant_output_0 =
       0.0f;
  3. Function parameter types (matched via call-site argument names):
       FUNC_PREFIX void node__X(const float x[...], float y[...])
  4. Pointer casts inside function bodies  (float*, (float*)):
       float *X = (float*)input;         →  uses 'fixed' base type
       *((float*)output + i) = ...       →  replaced with fixed*
  5. entry() interface parameters        →  kept as float (AXI interface)

Usage
------
  python rewrite_types.py  network.cpp  tensor_types.json  network_typed.cpp

  --base-type ap_fixed<18,5>   type for local temporaries (default)
  --dry-run                    print changes, don't write
  --log rewrite_log.txt        where to write the change log
"""

import argparse
import json
import re
import sys
from collections import Counter


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# Global tensor array declaration on one line:
#   float tensor__mod_...[1][16][35][16];
RE_ARRAY_DECL = re.compile(
    r'^((?:static\s+)?(?:const\s+)?)'   # 1: qualifiers
    r'(float)'                           # 2: type  (only float, not int64_t)
    r'(\s+)'                             # 3: space
    r'(tensor\w+)'                       # 4: name
    r'((?:\[\d+\])+)'                   # 5: dims (at least one)
    r'(\s*;)',                           # 6: semicolon
    re.MULTILINE
)

# Static const scalar  — name on line 1, value on line 2:
#   static const float tensor__mod_Constant_output_0 =\n  0.0f;
# OR all on one line:
#   static const float tensor__mod_Constant_output_0 = 0.0f;
RE_SCALAR_DECL = re.compile(
    r'^(static\s+const\s+)'             # 1: static const
    r'(float)'                          # 2: type
    r'(\s+)'                            # 3: space
    r'(tensor\w+)'                      # 4: name
    r'(\s*=)',                          # 5: equals (value follows on same or next line)
    re.MULTILINE
)

# Function definition header — everything up to the opening brace
RE_FUNC_HEADER = re.compile(
    r'(FUNC_PREFIX\s+void\s+)(node\w+)\s*(\([^)]*\))(?=\s*[\r\n]*\{)',
    re.DOTALL
)

# Single parameter within a parameter list
RE_PARAM = re.compile(
    r'(const\s+)?(float|int64_t|ap_fixed<[^>]+>)(\s+\*?\s*)(\w+)((?:\[\d+\])*)'
)

# Call site:  node__X(arg0, arg1, ...);
RE_CALL = re.compile(r'\b(node\w+)\s*\(([^;]+)\)\s*;')

# Pointer cast patterns inside function bodies — replace float* with fixed*
# Pattern A:  float *name = (float*)expr;
RE_PTR_DECL = re.compile(r'\bfloat\s*\*(\s*\w+\s*=\s*\(float\*\))')
# Pattern B:  *((float*)expr + offset)
RE_PTR_CAST = re.compile(r'\(\(float\*\)')

# entry() function signature — keep as float (AXI interface boundary)
RE_ENTRY = re.compile(r'\bvoid\s+entry\s*\([^)]*\)', re.DOTALL)


# ---------------------------------------------------------------------------
# Build call-site arg map
# ---------------------------------------------------------------------------

def build_call_args(src):
    """func_name -> [arg_tensor_name, ...]"""
    call_args = {}
    for m in RE_CALL.finditer(src):
        func_name = m.group(1)
        args = []
        for a in m.group(2).split(','):
            a = a.strip().lstrip('&')
            a = re.sub(r'^\([^)]+\)\s*', '', a)  # strip casts
            args.append(a)
        call_args[func_name] = args
    return call_args


# ---------------------------------------------------------------------------
# Rewrite 1: global array declarations
# ---------------------------------------------------------------------------

def rewrite_array_decls(src, type_map, log):
    changes = 0
    def replace(m):
        nonlocal changes
        qualifiers = m.group(1)
        old_type   = m.group(2)
        space      = m.group(3)
        name       = m.group(4)
        dims       = m.group(5)
        semi       = m.group(6)
        new_type = type_map.get(name)
        if new_type is None or new_type == 'float':
            return m.group(0)
        changes += 1
        log.append(f'DECL  {name}: float → {new_type}')
        return f'{qualifiers}{new_type}{space}{name}{dims}{semi}'
    result = RE_ARRAY_DECL.sub(replace, src)
    print(f'  Array declarations   : {changes}')
    return result


# ---------------------------------------------------------------------------
# Rewrite 2: static const scalar declarations
# ---------------------------------------------------------------------------

def rewrite_scalar_decls(src, type_map, log):
    changes = 0
    def replace(m):
        nonlocal changes
        sc         = m.group(1)
        old_type   = m.group(2)
        space      = m.group(3)
        name       = m.group(4)
        eq         = m.group(5)
        new_type = type_map.get(name)
        if new_type is None or new_type == 'float':
            return m.group(0)
        changes += 1
        log.append(f'SCALAR {name}: float → {new_type}')
        return f'{sc}{new_type}{space}{name}{eq}'
    result = RE_SCALAR_DECL.sub(replace, src)
    print(f'  Scalar declarations  : {changes}')
    return result


# ---------------------------------------------------------------------------
# Rewrite 3: function signatures
# ---------------------------------------------------------------------------

def rewrite_signatures(src, type_map, call_args, log):
    changes = 0

    def replace_sig(m):
        nonlocal changes
        prefix     = m.group(1)
        func_name  = m.group(2)
        params_str = m.group(3)

        if func_name not in call_args:
            return m.group(0)

        tensor_names = call_args[func_name]
        params = list(RE_PARAM.finditer(params_str))

        if len(params) != len(tensor_names):
            return m.group(0)

        new_params = params_str
        offset = 0

        for param_m, tensor_name in zip(params, tensor_names):
            const_q  = param_m.group(1) or ''
            old_type = param_m.group(2)
            space    = param_m.group(3)
            pname    = param_m.group(4)
            dims     = param_m.group(5)

            if old_type == 'int64_t':
                continue

            new_type = type_map.get(tensor_name)
            if new_type is None:
                clean = re.sub(r'^&|\([^)]+\)\s*', '', tensor_name)
                new_type = type_map.get(clean)

            if new_type is None or new_type == old_type or new_type == 'float':
                continue

            adj_start = param_m.start() + offset
            adj_end   = param_m.end()   + offset
            old_frag  = new_params[adj_start:adj_end]
            new_frag  = re.sub(
                r'\b' + re.escape(old_type) + r'\b',
                new_type, old_frag, count=1
            )

            if new_frag != old_frag:
                new_params = new_params[:adj_start] + new_frag + new_params[adj_end:]
                offset += len(new_frag) - len(old_frag)
                changes += 1
                log.append(
                    f'PARAM {func_name} [{pname}]: '
                    f'{old_type} → {new_type} (tensor: {tensor_name})'
                )

        return f'{prefix}{func_name}{new_params}'

    result = RE_FUNC_HEADER.sub(replace_sig, src)
    print(f'  Signatures           : {changes}')
    return result


# ---------------------------------------------------------------------------
# Rewrite 4: pointer casts inside function bodies
# Replace (float*) and float* with (fixed*) and fixed*
# so that Reshape/Unsqueeze/Cast memcpy tricks keep working
# ---------------------------------------------------------------------------

def rewrite_pointer_casts(src, base_type, log):
    changes = 0

    # We do NOT touch:
    #   - The entry() signature (AXI boundary)
    #   - Lines inside entry() that pass float tensors in
    #   - Comments

    # Strategy: find every function body (between { and matching })
    # that is NOT entry(), and replace float* / (float*) within it.
    # Simpler approach: replace globally but protect entry().

    # Save the entry() function text so we can restore it
    entry_match = RE_ENTRY.search(src)
    entry_placeholder = '___ENTRY_PLACEHOLDER___'
    if entry_match:
        entry_text = entry_match.group(0)
        src_work = src.replace(entry_text, entry_placeholder)
    else:
        src_work = src
        entry_text = ''

    # Replace  float *name = (float*)  →  fixed *name = (fixed*)
    def replace_ptr_decl(m):
        nonlocal changes
        changes += 1
        log.append(f'PTR_DECL  float* → {base_type}*')
        return f'{base_type} *{m.group(1)}'
    src_work = RE_PTR_DECL.sub(replace_ptr_decl, src_work)

    # Replace  ((float*)  →  (({base_type}*)
    def replace_ptr_cast(m):
        nonlocal changes
        changes += 1
        log.append(f'PTR_CAST  (float*) → ({base_type}*)')
        return f'(({base_type}*)'
    src_work = RE_PTR_CAST.sub(replace_ptr_cast, src_work)

    # Replace  (float)X  cast in Cast node  →  ({base_type})X
    def replace_val_cast(m):
        nonlocal changes
        changes += 1
        log.append(f'VAL_CAST  (float) → ({base_type})')
        return f'({base_type}){m.group(1)}'
    src_work = re.sub(r'\(float\)(\w)', replace_val_cast, src_work)

    # Replace  float tmp_X  local variable in BatchNorm
    def replace_local_float(m):
        nonlocal changes
        changes += 1
        log.append(f'LOCAL  float {m.group(1)} → {base_type} {m.group(1)}')
        return f'{base_type} {m.group(1)}'
    src_work = re.sub(r'\bfloat\s+(\w+\s*=)', replace_local_float, src_work)

    # Restore entry()
    if entry_text:
        src_work = src_work.replace(entry_placeholder, entry_text)

    print(f'  Pointer/local casts  : {changes}')
    return src_work


# ---------------------------------------------------------------------------
# Rewrite 5: add ap_fixed includes + base typedef (if not already present)
# ---------------------------------------------------------------------------

def ensure_includes(src, base_type, log):
    insert = (
        f'\n#include <ap_fixed.h>\n'
        f'#include <hls_math.h>\n'
        f'\ntypedef {base_type} fixed;\n'
    )
    if '#include <ap_fixed.h>' not in src:
        # Insert after weights.h include
        m = re.search(r'#include\s+"weights/weights\.h"', src)
        pos = m.end() if m else 0
        src = src[:pos] + insert + src[pos:]
        log.append(f'INCLUDE  added ap_fixed.h + typedef fixed={base_type}')
    return src


# ---------------------------------------------------------------------------
# Sanity check: report remaining floats
# ---------------------------------------------------------------------------

def report_remaining(src):
    lines = src.split('\n')
    remaining = []
    for i, l in enumerate(lines):
        if re.search(r'\bfloat\b', l):
            remaining.append((i+1, l.rstrip()))

    if not remaining:
        print('\n  ✓ No remaining float usages.')
        return

    print(f'\n  Remaining float usages: {len(remaining)}')
    # Categorise
    for lineno, l in remaining[:20]:
        print(f'    [{lineno}] {l}')
    if len(remaining) > 20:
        print(f'    ... and {len(remaining)-20} more')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('source',      help='Input .cpp/.h file (onnx2c output)')
    p.add_argument('type_json',   help='tensor_types.json from assign_types.py')
    p.add_argument('output',      nargs='?', default=None)
    p.add_argument('--base-type', default='ap_fixed<18,5>',
                   help='Type for local temporaries (default: ap_fixed<18,5>)')
    p.add_argument('--dry-run',   action='store_true')
    p.add_argument('--log',       default='rewrite_log.txt')
    args = p.parse_args()

    if args.output is None:
        stem = re.sub(r'\.\w+$', '', args.source)
        args.output = stem + '_typed.cpp'

    print(f'Reading : {args.source}')
    with open(args.source) as f:
        src = f.read()

    print(f'Type map: {args.type_json}')
    with open(args.type_json) as f:
        data = json.load(f)

    type_map = data['type_map']
    print(f'  {len(type_map)} tensor assignments\n')

    log = []
    print('Applying rewrites...')

    src = ensure_includes(src, args.base_type, log)
    src = rewrite_array_decls(src, type_map, log)
    src = rewrite_scalar_decls(src, type_map, log)
    call_args = build_call_args(src)
    print(f'  Call sites found     : {len(call_args)}')
    src = rewrite_signatures(src, type_map, call_args, log)
    src = rewrite_pointer_casts(src, args.base_type, log)

    print(f'\nTotal changes: {len(log)}')
    report_remaining(src)

    if args.dry_run:
        print('\n[DRY RUN] First 50 changes:')
        for entry in log[:50]:
            print(f'  {entry}')
        return

    with open(args.output, 'w') as f:
        f.write(src)
    print(f'\nWritten → {args.output}')

    with open(args.log, 'w') as f:
        f.write('\n'.join(log) + '\n')
    print(f'Log     → {args.log}')

    type_counts = Counter(type_map.values())
    print('\nTypes in output:')
    for typ, cnt in sorted(type_counts.items()):
        print(f'  {typ:<25}  {cnt:>4} tensors')


if __name__ == '__main__':
    main()
