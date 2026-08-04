#!/usr/bin/env python3
"""
split_weights.py
================
Splits large static const array declarations out of a self-contained
onnx2c-generated C file into individual files, and generates a weights.h
that declares them all as extern so the main file can include it.

Designed to work AFTER fix_types.py has run, so weight arrays are
already typed as  static const ap_fixed<W,I>  rather than float.

What gets split
---------------
Any  static const <type> tensor_name[dims] = { ... };  declaration
where the total number of elements is >= MIN_ELEMENTS (default 8).

What stays inline
-----------------
Small operational constants (Expand shapes, ReduceSum axes, Slice
params, TopK K values, etc.) — typically 1–4 elements — stay in the
main file unchanged.

Output
------
  output_dir/tensor_name.cpp    one file per split tensor
  weights.h                     extern declarations for all split tensors
  model_inference_split.cpp     main file with declarations replaced by
                                `#include "weights.h"` + comment stubs

Usage
-----
  python split_weights.py  input.cpp  \\
      [--output-dir weights_split]   \\
      [--weights-h  weights.h]       \\
      [--output     model_split.cpp] \\
      [--min-elements 8]             \\
      [--include    ap_fixed.h]
"""

import argparse
import os
import re
import sys


# ---------------------------------------------------------------------------
# Regex to match a single static const declaration (possibly multi-line value)
#
# Handles:
#   static const ap_fixed<16,7> tensor_name[16][8][1][1] = \n{ ... };
#   static const int64_t        tensor_name[4]           = {1, 4, 35, 16};
#   static const ap_fixed<40,21> tensor__mod_Constant_1_output_0 = \n500.0f;
#     (scalar with no [] dims)
#
# The value part can be a bare scalar or a brace-enclosed block.
# We match it by finding the semicolon at brace depth 0.
# ---------------------------------------------------------------------------
DECL_RE = re.compile(
    r'(static\s+const\s+)'                        # group 1: static const
    r'(ap_fixed<[^>]+>|int64_t|float|double|int)' # group 2: type
    r'(\s+)'                                       # group 3: whitespace
    r'(\w+)'                                       # group 4: tensor name
    r'((?:\s*\[[^\]]*\])*)'                        # group 5: dims (may be empty for scalars)
    r'\s*=\s*',                                    # equals
    re.DOTALL
)


def count_elements(dims_str):
    """Return total element count from a dims string like '[16][8][1][1]'."""
    dims = re.findall(r'\[(\d+)\]', dims_str)
    if not dims:
        return 1   # scalar
    result = 1
    for d in dims:
        result *= int(d)
    return result


def find_value_end(src, pos):
    """
    Starting at pos (just after the '='), find the end of the initialiser
    (including the terminating ';').  Handles nested braces and the case
    of a bare scalar value (no braces).
    Returns the index just AFTER the semicolon, or -1 if not found.
    """
    i = pos
    n = len(src)

    # Skip leading whitespace
    while i < n and src[i] in ' \t\n\r':
        i += 1

    if i >= n:
        return -1

    if src[i] == '{':
        # Brace-enclosed value — find matching }
        depth = 0
        while i < n:
            if src[i] == '{':
                depth += 1
            elif src[i] == '}':
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        # Now skip optional whitespace and require ';'
        while i < n and src[i] in ' \t\n\r':
            i += 1
        if i < n and src[i] == ';':
            return i + 1
        return -1
    else:
        # Bare scalar (e.g. 500.0f or 9223372036854775807)
        # Read until ';'
        while i < n and src[i] != ';':
            i += 1
        if i < n:
            return i + 1
        return -1


def parse_declarations(src, min_elements):
    """
    Find all static const declarations in src.
    Returns a list of dicts:
      start, end, type, name, dims, n_elements, full_text, value_text
    Sorted by start position.
    """
    decls = []
    for m in DECL_RE.finditer(src):
        sc        = m.group(1)
        typ       = m.group(2)
        ws        = m.group(3)
        name      = m.group(4)
        dims      = m.group(5)
        val_start = m.end()

        val_end = find_value_end(src, val_start)
        if val_end == -1:
            continue  # couldn't parse — skip

        n_elems = count_elements(dims)
        full_text  = src[m.start():val_end]
        value_text = src[val_start:val_end].strip()  # includes trailing ;

        decls.append({
            'start':      m.start(),
            'end':        val_end,
            'type':       typ,
            'name':       name,
            'dims':       dims.strip(),
            'n_elements': n_elems,
            'full_text':  full_text,
            'value_text': value_text,
        })

    return decls


def split_weights(src, min_elements, output_dir, weights_h_path,
                  extra_includes, verbose):
    """
    Core logic.  Returns (new_src, split_decls, kept_decls).
    """
    decls = parse_declarations(src, min_elements)

    split_decls = [d for d in decls if d['n_elements'] >= min_elements]
    kept_decls  = [d for d in decls if d['n_elements'] <  min_elements]

    if verbose:
        print(f'  Total declarations found : {len(decls)}')
        print(f'  To split (>= {min_elements} elements): {len(split_decls)}')
        print(f'  Kept inline (< {min_elements} elements): {len(kept_decls)}')

    # --- Write individual weight files ---
    os.makedirs(output_dir, exist_ok=True)

    # Compute relative path from output_dir back to weights_h_path
    # so each .cpp can include weights.h correctly regardless of where
    # both directories sit.
    weights_h_rel = os.path.relpath(weights_h_path, output_dir)

    for d in split_decls:
        fname = os.path.join(output_dir, d['name'] + '.cpp')
        inc_lines = [
            f'#include "{weights_h_rel}"',   # weights.h first (has ap_fixed.h + type defs)
        ]
        for inc in extra_includes:
            inc_lines.append(f'#include "{inc}"')
        inc_block = '\n'.join(inc_lines) + '\n\n'

        # Reconstruct the declaration with `static const` → `const`
        # (extern in weights.h + definition here, no `static` needed)
        decl_line = f'const {d["type"]} {d["name"]}{d["dims"]} =\n'
        content = inc_block + decl_line + d['value_text'] + '\n'

        with open(fname, 'w') as f:
            f.write(content)

        if verbose:
            print(f'    → {fname}  ({d["n_elements"]} elements)')

    # --- Write weights.h ---
    h_lines = [
        '#pragma once',
        '#include <ap_fixed.h>',
        '',
        '/* Extern declarations for split weight tensors.',
        ' * Each tensor is defined in weights_split/<name>.cpp',
        ' */',
        '',
    ]
    for d in split_decls:
        h_lines.append(f'extern const {d["type"]} {d["name"]}{d["dims"]};')

    with open(weights_h_path, 'w') as f:
        f.write('\n'.join(h_lines) + '\n')

    if verbose:
        print(f'  weights.h: {len(split_decls)} extern declarations → {weights_h_path}')

    # --- Rewrite main source ---
    # Replace each split declaration with a comment stub.
    # Work in reverse order so character positions stay valid.
    new_src = src
    offset = 0
    for d in sorted(split_decls, key=lambda x: x['start']):
        start = d['start'] + offset
        end   = d['end']   + offset
        stub  = f'// moved to {output_dir}/{d["name"]}.cpp\n'
        new_src = new_src[:start] + stub + new_src[end:]
        offset += len(stub) - (end - start)

    # Add #include "weights.h" after the last existing #include
    weights_include = f'#include "{weights_h_path}"\n'
    if weights_include not in new_src:
        last_include = None
        for m in re.finditer(r'^#include\s+[<"][^>"]+[>"]', new_src, re.MULTILINE):
            last_include = m
        if last_include:
            pos = last_include.end() + 1
            new_src = new_src[:pos] + weights_include + new_src[pos:]
        else:
            new_src = weights_include + new_src

    return new_src, split_decls, kept_decls


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Split static const weight arrays into individual files.'
    )
    p.add_argument('source',
                   help='Input .cpp file (after fix_types.py)')
    p.add_argument('--output',      default=None,
                   help='Output .cpp file (default: <source>_split.cpp)')
    p.add_argument('--output-dir',  default='weights_split',
                   help='Directory for individual weight files (default: weights_split)')
    p.add_argument('--weights-h',   default='weights.h',
                   help='Path for the generated weights header (default: weights.h)')
    p.add_argument('--min-elements', type=int, default=8,
                   help='Minimum elements to trigger splitting (default: 8). '
                        'Keeps small operational constants inline.')
    p.add_argument('--include',     nargs='*', default=[],
                   metavar='HEADER',
                   help='Extra headers to include in each weight file '
                        '(e.g. --include hls_math.h)')
    p.add_argument('--verbose',     action='store_true')
    p.add_argument('--dry-run',     action='store_true',
                   help='Show what would be split without writing any files')
    args = p.parse_args()

    if args.output is None:
        stem = re.sub(r'\.\w+$', '', args.source)
        args.output = stem + '_split.cpp'

    with open(args.source) as f:
        src = f.read()

    print(f'Input  : {args.source}  ({src.count(chr(10))} lines)')
    print(f'Splitting declarations with >= {args.min_elements} elements\n')

    new_src, split_decls, kept_decls = split_weights(
        src,
        min_elements=args.min_elements,
        output_dir=args.output_dir,
        weights_h_path=args.weights_h,
        extra_includes=args.include,
        verbose=True,
    )

    # Summary table
    print()
    print(f'{"Tensor":<55}  {"Type":<22}  {"Elements":>8}')
    print('-' * 90)
    for d in split_decls:
        short = d['name'][-55:]
        print(f'  {short:<55}  {d["type"]:<22}  {d["n_elements"]:>8}')

    print(f'\nTotal elements split out: {sum(d["n_elements"] for d in split_decls):,}')

    if args.dry_run:
        print('\n[DRY RUN] No files written.')
        return

    with open(args.output, 'w') as f:
        f.write(new_src)

    print(f'\nOutput : {args.output}  ({new_src.count(chr(10))} lines)')
    print(f'Header : {args.weights_h}')
    print(f'Files  : {len(split_decls)} in {args.output_dir}/')
    print('\nTo compile, add all weights_split/*.cpp to your HLS source list.')


if __name__ == '__main__':
    main()