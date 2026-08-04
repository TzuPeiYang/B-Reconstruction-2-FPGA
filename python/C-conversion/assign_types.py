#!/usr/bin/env python3
"""
assign_types.py
===============
Walks an ONNX graph and assigns a specific ap_fixed<W,I> type to every
tensor, using the empirically-derived groupings from ranges.csv.

Outputs a JSON file  tensor_types.json  mapping  tensor_name -> C_type_string.

The groupings and their recommended types are:

  GROUP               CRITERION                          TYPE
  ──────────────────────────────────────────────────────────────────────
  distance_graph      MatMul/Pow/ReduceSum/Neg/Sub/       ap_fixed<40,21>
                      Mul/Transpose_1/TopK_values in
                      the KNN distance pipeline
  index_tensor        int64 index arrays (Slice out,      int64_t  (unchanged)
                      TopK indices, Add out, Reshape,
                      Gather indices, INT64_MAX constants)
  sentinel            Constant_5 = INT64_MAX              int64_t
  activation_wide     I=7 needed  (ec1/ec2 conv chain,    ap_fixed<16,7>
                      output_bn outputs, max ~43)
  activation_mid      I=6 needed  (ec0/ec1/ec2 conv,      ap_fixed<16,6>
                      fusion, max ~29)
  activation_narrow   I=5 needed  (BN out, ec0 gather,    ap_fixed<16,5>
                      max ~16)
  input_small         I≤4  (inputs, mask, MSE output,     ap_fixed<16,4>
                      cast, equal, constants ≤16)
  weight              All initializer tensors              ap_fixed<16,7>
                      (weights/biases)

Classification is done in two passes:
  Pass 1 – classify using the ranges.csv empirical data if provided.
  Pass 2 – for any tensor not in ranges.csv, use structural rules
           (op type, position in graph, name patterns).

Usage
-----
  # With empirical ranges (most accurate):
  python assign_types.py model.onnx --ranges ranges.csv --output tensor_types.json

  # Structural rules only (no calibration data):
  python assign_types.py model.onnx --output tensor_types.json

  # Override the type for a specific group:
  python assign_types.py model.onnx --ranges ranges.csv \
      --override distance_graph=ap_fixed<40,21> \
      --output tensor_types.json
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict

try:
    import onnx
    from onnx import numpy_helper, TensorProto
except ImportError:
    print("pip install onnx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Default type assignments per group
# ---------------------------------------------------------------------------
DEFAULT_TYPES = {
    'distance_graph':   'ap_fixed<32,19>',   # KNN distances — must stay wide
    'index_tensor':     'int32_t',            # particle indices, not features
    'sentinel':         'int64_t',
    'activation_wide':  'ap_fixed<20,7>',    # I=7, max ~43
    'activation_mid':   'ap_fixed<19,6>',    # I=6, max ~29
    'activation_narrow':'ap_fixed<18,5>',    # I=5, max ~16
    'input_small':      'ap_fixed<17,4>',    # I≤4, inputs/mask/MSE
    'weight':           'ap_fixed<18,5>',    # initializer weights & biases
    'unknown':          'ap_fixed<32,19>',    # conservative fallback
}

INT64_MAX = 9223372036854775807


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def min_int_bits(max_abs):
    if max_abs <= 0:
        return 1
    return math.ceil(math.log2(max_abs + 1e-9)) + 1


def load_ranges_csv(path):
    """Returns dict  tensor_name -> (max_abs, int_bits)"""
    result = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            result[row['name']] = (
                float(row['max_abs']),
                int(row['int_bits']),
            )
    return result


# ---------------------------------------------------------------------------
# Graph-structure classifier
# ---------------------------------------------------------------------------

# Ops that are part of the KNN distance pipeline (squared L2 distance)
DISTANCE_OPS = {
    'MatMul', 'Pow', 'ReduceSum', 'Neg', 'Mul',
    'Transpose',   # the Transpose_1 that transposes the norm vector
    'TopK',        # TopK values output (not indices)
    'Sub',         # Sub and Sub_1 in the distance computation
}

# Op output names (substrings) that are always int64 indices
INDEX_NAME_PATTERNS = [
    'TopK_output_1',     # TopK second output = indices
    'Slice_output',
    'Add_output_0',      # index offset Add
    'Reshape_output_0',  # flat index reshape
    'Constant_7',        # the [[0]] constant
    'Constant_8',        # the -1 reshape constant
]

# Constant name patterns that are always int64 sentinel/config values
SENTINEL_PATTERNS = [
    'Constant_5',   # INT64_MAX  "slice to end"
    'Constant_3',   # axis=2
    'Constant_4',   # starts=1
    'Constant_6',   # steps=1
]


def is_index_name(name):
    return any(p in name for p in INDEX_NAME_PATTERNS)


def is_sentinel_name(name):
    return any(p in name for p in SENTINEL_PATTERNS)


def classify_by_structure(model, initializer_names):
    """
    Walk the graph and classify each output tensor by structural rules.
    Returns dict  name -> group_string
    """
    groups = {}

    # All initializers are weights
    for name in initializer_names:
        groups[name] = 'weight'

    # Classify node outputs
    for node in model.graph.node:
        op = node.op_type

        for out_name in node.output:
            if not out_name:
                continue

            # Index tensors by name pattern
            if is_index_name(out_name):
                groups[out_name] = 'index_tensor'
                continue

            if is_sentinel_name(out_name):
                groups[out_name] = 'sentinel'
                continue

            # Distance graph detection:
            # We need to know if this node is in the KNN distance sub-graph.
            # Key heuristic: in onnx2c-generated ParticleNet code, the
            # distance pipeline has a very specific sequence:
            #   Transpose → MatMul → Mul(-2) → Pow(x,2) → ReduceSum →
            #   Neg → Sub → Transpose → Sub_1 → TopK → Slice → Add → Reshape
            # All tensors in this chain are distance_graph EXCEPT:
            #   - the input features (they feed into Transpose but also into
            #     the feature gather path)
            #   - the TopK indices output (index_tensor)
            #
            # We detect by checking if the *immediate consumer* of this
            # output is a distance op, OR the *producer* is a distance op
            # and the output name matches distance patterns.

            # Producer-based: if this op is a distance op and not a
            # feature-path op, mark as distance_graph
            if op in DISTANCE_OPS:
                # Extra check: MatMul and Mul appear in both the distance
                # pipeline AND in Conv (1x1). Distinguish by output shape
                # names — distance MatMul outputs are [35,35] shaped,
                # Conv inputs/outputs are [1,C,35,16] shaped.
                # We use name heuristics:
                if op == 'MatMul' and ('MatMul_output' in out_name):
                    groups[out_name] = 'distance_graph'
                elif op == 'Pow' and 'Pow_output' in out_name:
                    groups[out_name] = 'distance_graph'
                elif op == 'ReduceSum' and 'ReduceSum_output' in out_name:
                    # ReduceSum_1 is the spatial pooling in fusion — NOT distance
                    if 'ReduceSum_1' not in out_name:
                        groups[out_name] = 'distance_graph'
                elif op == 'Neg' and 'Neg_output' in out_name:
                    groups[out_name] = 'distance_graph'
                elif op == 'Sub' and ('Sub_output_0' in out_name or 'Sub_1_output' in out_name):
                    # Sub_2 is the feature subtraction (neighbour - self), NOT distance
                    if 'Sub_2' not in out_name:
                        groups[out_name] = 'distance_graph'
                elif op == 'Mul' and 'Mul_output_0' in out_name:
                    # The -2*XY scaling Mul — these are the bare /mod/edge_convs.N/Mul_output_0
                    # (not Mul_2, Mul_3, Mul_4 etc. which are feature masking ops)
                    if out_name.count('_') <= 5:  # short name = distance path
                        groups[out_name] = 'distance_graph'
                elif op == 'Transpose' and 'Transpose_1_output' in out_name:
                    groups[out_name] = 'distance_graph'
                elif op == 'TopK':
                    # output 0 = values (distance), output 1 = indices
                    if out_name == node.output[0]:
                        groups[out_name] = 'distance_graph'
                    else:
                        groups[out_name] = 'index_tensor'

            # Default: unknown (will be resolved by ranges.csv or fallback)
            if out_name not in groups:
                groups[out_name] = 'unknown'

    # Input tensors
    for inp in model.graph.input:
        if inp.name not in groups:
            groups[inp.name] = 'input_small'

    return groups


def classify_by_ranges(ranges_csv, groups):
    """
    Override structural groups with empirically-measured int_bits.
    Returns updated groups dict.
    """
    INT64_SENTINEL = 9.223372036854776e+18  # INT64_MAX as float

    for name, (max_abs, int_bits) in ranges_csv.items():

        # Already correctly classified as distance/index/sentinel by structure
        if groups.get(name) in ('distance_graph', 'index_tensor', 'sentinel'):
            continue

        # Catch any remaining INT64_MAX sentinels
        if abs(max_abs - INT64_SENTINEL) < 1e9:
            groups[name] = 'sentinel'
            continue

        # Values that suspiciously large are likely distance overflow artefacts
        # we missed structurally
        if max_abs > 1e15:
            groups[name] = 'distance_graph'
            continue

        # Assign by int_bits
        if int_bits <= 4:
            groups[name] = 'input_small'
        elif int_bits == 5:
            groups[name] = 'activation_narrow'
        elif int_bits == 6:
            groups[name] = 'activation_mid'
        elif int_bits <= 8:
            groups[name] = 'activation_wide'
        else:
            # Genuinely large real activation — keep at original wide type
            groups[name] = 'distance_graph'  # conservative: use wide type

    return groups


# ---------------------------------------------------------------------------
# Build the final type mapping
# ---------------------------------------------------------------------------

def build_type_map(model, ranges_csv, type_assignments):
    """
    Returns dict  tensor_name -> C_type_string
    """
    inits = {init.name for init in model.graph.initializer}

    # Pass 1: structural classification
    groups = classify_by_structure(model, inits)

    # Pass 2: override with empirical ranges
    if ranges_csv:
        groups = classify_by_ranges(ranges_csv, groups)

    # Map groups to C types
    type_map = {}
    for name, group in groups.items():
        type_map[name] = type_assignments.get(group, type_assignments['unknown'])

    return type_map, groups


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(groups, type_assignments):
    from collections import Counter
    counts = Counter(groups.values())
    print('\nGroup summary:')
    print(f'  {"Group":<22}  {"Count":>6}  {"Type"}')
    print(f'  {"-"*22}  {"-"*6}  {"-"*25}')
    for group in sorted(counts.keys()):
        typ = type_assignments.get(group, '?')
        print(f'  {group:<22}  {counts[group]:>6}  {typ}')

    print(f'\nTotal tensors classified: {sum(counts.values())}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_overrides(override_list):
    """Parse ['group=type', ...] into a dict."""
    result = {}
    if not override_list:
        return result
    for item in override_list:
        if '=' not in item:
            print(f'Warning: ignoring malformed override "{item}" (expected group=type)')
            continue
        group, typ = item.split('=', 1)
        result[group.strip()] = typ.strip()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model',   help='Path to .onnx file')
    p.add_argument('--ranges',   help='Path to ranges.csv from find_range_requirements.py')
    p.add_argument('--output',   default='tensor_types.json')
    p.add_argument('--override', nargs='*', metavar='GROUP=TYPE',
                   help='Override type for a group, e.g. distance_graph=ap_fixed<32,16>')
    p.add_argument('--dump-groups', action='store_true',
                   help='Print every tensor and its group to stdout')
    args = p.parse_args()

    print(f'Loading {args.model}')
    model = onnx.load(args.model)

    # Build type assignments (defaults + user overrides)
    type_assignments = dict(DEFAULT_TYPES)
    type_assignments.update(parse_overrides(args.override))
    print('\nType assignments:')
    for group, typ in type_assignments.items():
        print(f'  {group:<22} → {typ}')

    # Load ranges
    ranges_csv = {}
    if args.ranges:
        ranges_csv = load_ranges_csv(args.ranges)
        print(f'\nLoaded {len(ranges_csv)} empirical ranges from {args.ranges}')

    # Build map
    type_map, groups = build_type_map(model, ranges_csv, type_assignments)

    # Print summary
    print_summary(groups, type_assignments)

    if args.dump_groups:
        print('\nAll tensors:')
        for name in sorted(groups.keys()):
            print(f'  {groups[name]:<22}  {type_map[name]:<25}  {name}')

    # Convert ONNX tensor names → onnx2c C variable names before saving.
    # ONNX:  /mod/edge_convs.0/Concat_output_0
    # C:     tensor__mod_edge_convs_0_Concat_output_0
    # Without this, rewrite_types.py can't find any matching keys.
    def onnx_to_c_varname(name):
        if name.startswith('onnx::'):
            return 'tensor_onnx__' + name[6:]
        s = name.lstrip('/')
        s = s.replace('/', '_').replace('.', '_').replace(':', '_')
        return 'tensor__' + s

    c_type_map = {onnx_to_c_varname(k): v for k, v in type_map.items()}
    c_groups   = {onnx_to_c_varname(k): v for k, v in groups.items()}

    # Save JSON
    output = {
        'type_map': c_type_map,
        'groups':   c_groups,
        'type_assignments': type_assignments,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nWritten → {args.output}')
    print(f'  Keys are C variable names (tensor__...), ready for rewrite_types.py')


if __name__ == '__main__':
    main()
