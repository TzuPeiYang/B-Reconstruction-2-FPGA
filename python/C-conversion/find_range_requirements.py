#!/usr/bin/env python3
"""
find_range_requirements.py
===========================
Traces value ranges through an ONNX model using TWO complementary methods:

  Method 1 – Empirical (calibration data):
    Runs real data through the model using ONNXRuntime with intermediate
    output hooks, records the actual min/max seen at every tensor.

  Method 2 – Analytic (weight inspection only):
    For each op, propagates worst-case bounds from the weight tensors.
    Useful when you don't have calibration data handy.

For every tensor it recommends:
  - Minimum integer bits needed  (ceil(log2(max_abs)) + 1 for sign)
  - A suggested ap_fixed<W, I> pairing assuming W=16 total bits
  - Flags any tensor where I > a threshold (default 8) as a "wide int" hotspot

Usage
------
  # With calibration data (most accurate):
  python find_range_requirements.py model.onnx --data calibration.npz --input-names pf_points pf_features pf_mask

  # Weight-only analytic mode (no data needed):
  python find_range_requirements.py model.onnx --analytic

  # Both:
  python find_range_requirements.py model.onnx --data calibration.npz --input-names pf_points pf_features pf_mask --analytic

The calibration .npz should contain arrays named the same as --input-names.
If you have a PyTorch dataloader instead, see the helper at the bottom of
this file.
"""

import argparse
import math
import sys
import numpy as np
from collections import defaultdict

try:
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper, TensorProto
except ImportError:
    print("Missing dependencies. Run:  pip install onnx onnxruntime numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def min_int_bits(max_abs: float) -> int:
    """
    Number of integer bits needed to represent ±max_abs in two's complement.
    Includes the sign bit.
      max_abs ≤ 1     → 2 bits  (fits in [-1, 1) with 1 int bit + sign)
      max_abs ≤ 2     → 2 bits
      max_abs ≤ 4     → 3 bits
      max_abs ≤ 64    → 7 bits
      max_abs ≤ 1024  → 11 bits
    Formula: ceil(log2(max_abs + 1)) + 1  (the +1 handles exact powers of 2)
    """
    if max_abs <= 0:
        return 1
    return math.ceil(math.log2(max_abs + 1e-9)) + 1


def suggest_apfixed(max_abs: float, total_bits: int = 16) -> str:
    I = min_int_bits(max_abs)
    I = max(I, 1)
    if I >= total_bits:
        # Need more total bits
        total_bits = I + 4   # keep at least 4 fractional bits
    return f"ap_fixed<{total_bits},{I}>"


def fmt_range(lo, hi):
    return f"[{lo:+.4f}, {hi:+.4f}]"


# ---------------------------------------------------------------------------
# Method 1: Empirical range collection via ONNXRuntime
# ---------------------------------------------------------------------------

def add_intermediate_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Clone the model and add every intermediate value as a graph output,
    so ONNXRuntime returns them all.
    """
    model = onnx.ModelProto()
    model.CopyFrom(onnx.load(model) if isinstance(model, str) else model)

    existing_outputs = {o.name for o in model.graph.output}
    value_info_map = {vi.name: vi for vi in model.graph.value_info}

    for node in model.graph.node:
        for output_name in node.output:
            if output_name and output_name not in existing_outputs:
                if output_name in value_info_map:
                    model.graph.output.append(value_info_map[output_name])
                else:
                    # Add with unknown shape – ONNX RT will still return it
                    vi = onnx.helper.make_tensor_value_info(
                        output_name, TensorProto.FLOAT, None
                    )
                    model.graph.output.append(vi)

    return model


def empirical_ranges(model_path: str, input_data: dict, n_samples: int = 100):
    """
    Run up to n_samples batches through ONNXRuntime and record min/max
    for every intermediate tensor.

    input_data: dict mapping input_name -> np.ndarray of shape [N, ...]
    Returns: dict  tensor_name -> (min_val, max_val)
    """
    print(f"\n[Empirical] Loading model: {model_path}")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    # Extend model outputs to capture intermediates
    print("[Empirical] Extending model outputs to capture intermediates...")
    ext_model = onnx.ModelProto()
    ext_model.CopyFrom(model)

    existing_outputs = {o.name for o in ext_model.graph.output}
    value_info_map = {vi.name: vi for vi in ext_model.graph.value_info}

    added = 0
    for node in ext_model.graph.node:
        for out_name in node.output:
            if out_name and out_name not in existing_outputs:
                if out_name in value_info_map:
                    ext_model.graph.output.append(value_info_map[out_name])
                else:
                    vi = onnx.helper.make_tensor_value_info(
                        out_name, TensorProto.FLOAT, None
                    )
                    ext_model.graph.output.append(vi)
                existing_outputs.add(out_name)
                added += 1

    print(f"[Empirical] Added {added} intermediate outputs")

    # Serialise to bytes (RT can load from bytes)
    ext_bytes = ext_model.SerializeToString()

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3   # suppress warnings
    sess = ort.InferenceSession(ext_bytes, sess_opts)

    output_names = [o.name for o in sess.get_outputs()]

    # Figure out number of samples
    first_key = next(iter(input_data))
    N = len(input_data[first_key])
    n_samples = min(n_samples, N)

    ranges = defaultdict(lambda: [float('inf'), float('-inf')])

    print(f"[Empirical] Running {n_samples} samples...")
    for idx in range(n_samples):
        feed = {k: v[idx:idx+1].astype(np.float32) for k, v in input_data.items()}
        try:
            outputs = sess.run(output_names, feed)
        except Exception as e:
            print(f"  Warning: sample {idx} failed: {e}")
            continue

        for name, arr in zip(output_names, outputs):
            if arr is None or not hasattr(arr, 'min'):
                continue
            arr = np.array(arr, dtype=np.float32)
            if arr.size == 0:
                continue
            lo = float(arr.min())
            hi = float(arr.max())
            if lo < ranges[name][0]:
                ranges[name][0] = lo
            if hi > ranges[name][1]:
                ranges[name][1] = hi

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{n_samples} samples processed")

    # Convert to plain tuples
    return {k: (v[0], v[1]) for k, v in ranges.items()
            if v[0] != float('inf')}


# ---------------------------------------------------------------------------
# Method 2: Analytic range propagation from weights
# ---------------------------------------------------------------------------

def get_initializers(model: onnx.ModelProto) -> dict:
    """Extract all weight tensors as numpy arrays."""
    return {
        init.name: numpy_helper.to_array(init)
        for init in model.graph.initializer
    }


def analytic_ranges(model_path: str, input_max: float = 1.0):
    """
    Propagate worst-case absolute bounds through the graph analytically.

    Rules per op type:
      Conv/Gemm:   out_max = in_max * weight_max * fan_in + bias_max
      Add/Sub:     out_max = sum of input maxima
      Mul:         out_max = product of input maxima
      Relu:        out_max = in_max  (clips negatives, doesn't amplify)
      BatchNorm:   out_max = (in_max + |mean|_max) / sqrt(var_min) * |scale|_max + |bias|_max
      ReduceMean:  out_max = in_max  (averaging can't amplify)
      ReduceSum:   out_max = in_max * N  (worst case all same sign)
      MatMul:      out_max = in_max^2 * K  (dot product of two activations)
      TopK/Gather/Transpose/Reshape/Slice/Expand/Unsqueeze/Concat:
                   out_max = max of input maxima (data rearrangement only)
      Neg:         out_max = in_max
      Pow(x,2):    out_max = in_max^2

    Returns: dict  tensor_name -> (analytic_lo, analytic_hi)
             where lo = -hi (symmetric worst-case bound)
    """
    print(f"\n[Analytic] Loading model: {model_path}")
    model = onnx.load(model_path)
    inits = get_initializers(model)

    # Input tensor maxima (treat all inputs as ±input_max)
    input_names = {inp.name for inp in model.graph.input
                   if inp.name not in inits}

    # bounds[name] = max absolute value (symmetric assumed)
    bounds = {}
    for name in input_names:
        bounds[name] = input_max
        print(f"  Input '{name}' assumed max abs = {input_max}")

    # Weight bounds
    for name, arr in inits.items():
        bounds[name] = float(np.abs(arr).max())

    def get_bound(name):
        return bounds.get(name, input_max)

    def node_shape_info(node, attr_name):
        """Try to get an integer attribute from a node."""
        for attr in node.attribute:
            if attr.name == attr_name:
                return attr.i
        return None

    # Walk nodes in topological order (ONNX guarantees this)
    for node in model.graph.node:
        op = node.op_type
        inputs = list(node.input)
        outputs = list(node.output)

        in_bounds = [get_bound(i) for i in inputs if i]

        # ---- per-op rules ----
        if op in ('Conv',):
            # inputs: x, w, bias(optional)
            x_max = get_bound(inputs[0]) if len(inputs) > 0 else input_max
            w_name = inputs[1] if len(inputs) > 1 else None
            b_name = inputs[2] if len(inputs) > 2 else None

            if w_name and w_name in inits:
                w = inits[w_name]
                w_max = float(np.abs(w).max())
                # fan_in = product of all dims except output channel (dim 0)
                fan_in = int(np.prod(w.shape[1:]))
            else:
                w_max = get_bound(w_name) if w_name else 1.0
                fan_in = 1

            b_max = float(np.abs(inits[b_name]).max()) if (b_name and b_name in inits) else 0.0
            out_max = x_max * w_max * fan_in + b_max

        elif op in ('Gemm', 'MatMul'):
            a_max = get_bound(inputs[0]) if len(inputs) > 0 else input_max
            b_name = inputs[1] if len(inputs) > 1 else None
            bias_name = inputs[2] if len(inputs) > 2 else None

            if b_name and b_name in inits:
                b = inits[b_name]
                b_max = float(np.abs(b).max())
                K = b.shape[-1] if op == 'Gemm' else b.shape[0]
            else:
                b_max = get_bound(b_name) if b_name else 1.0
                K = 1

            bias_max = float(np.abs(inits[bias_name]).max()) if (bias_name and bias_name in inits) else 0.0
            out_max = a_max * b_max * K + bias_max

        elif op in ('Add',):
            out_max = sum(get_bound(i) for i in inputs if i)

        elif op in ('Sub',):
            # A - B: worst case is |A| + |B|
            out_max = sum(get_bound(i) for i in inputs if i)

        elif op in ('Mul',):
            out_max = 1.0
            for i in inputs:
                if i:
                    out_max *= get_bound(i)

        elif op in ('Relu', 'LeakyRelu', 'Sigmoid', 'Tanh'):
            out_max = get_bound(inputs[0]) if inputs else input_max
            if op == 'Sigmoid':
                out_max = 1.0
            elif op == 'Tanh':
                out_max = 1.0

        elif op == 'BatchNormalization':
            x_max = get_bound(inputs[0]) if inputs else input_max
            scale_name = inputs[1] if len(inputs) > 1 else None
            bias_name  = inputs[2] if len(inputs) > 2 else None
            mean_name  = inputs[3] if len(inputs) > 3 else None
            var_name   = inputs[4] if len(inputs) > 4 else None

            scale_max = float(np.abs(inits[scale_name]).max()) if (scale_name and scale_name in inits) else 1.0
            bias_max  = float(np.abs(inits[bias_name]).max())  if (bias_name  and bias_name  in inits) else 0.0
            mean_max  = float(np.abs(inits[mean_name]).max())  if (mean_name  and mean_name  in inits) else 0.0
            var_min   = float(np.abs(inits[var_name]).min())   if (var_name   and var_name   in inits) else 1.0
            var_min   = max(var_min, 1e-5)  # epsilon

            out_max = (x_max + mean_max) / math.sqrt(var_min) * scale_max + bias_max

        elif op == 'ReduceMean':
            out_max = get_bound(inputs[0]) if inputs else input_max

        elif op == 'ReduceSum':
            # Need to know the reduction axis size
            # Conservative: assume worst-case = in_max * 35 (spatial size)
            in_max = get_bound(inputs[0]) if inputs else input_max
            # Try to get from the keepdims / axes attributes or just be conservative
            out_max = in_max * 35  # worst case for N=35 points

        elif op == 'Pow':
            # Exponent is always 2.0 in this graph
            in_max = get_bound(inputs[0]) if inputs else input_max
            out_max = in_max ** 2

        elif op == 'Neg':
            out_max = get_bound(inputs[0]) if inputs else input_max

        elif op == 'Sqrt':
            in_max = get_bound(inputs[0]) if inputs else input_max
            out_max = math.sqrt(in_max)

        elif op == 'Div':
            a_max = get_bound(inputs[0]) if len(inputs) > 0 else input_max
            b_min = get_bound(inputs[1]) if len(inputs) > 1 else 1.0
            b_min = max(b_min, 1e-9)
            out_max = a_max / b_min

        elif op == 'Exp':
            in_max = get_bound(inputs[0]) if inputs else input_max
            out_max = math.exp(min(in_max, 20))  # cap to avoid overflow

        elif op in ('TopK',):
            # TopK just selects values; max value unchanged
            out_max = get_bound(inputs[0]) if inputs else input_max

        elif op in ('Transpose', 'Reshape', 'Flatten', 'Squeeze',
                    'Unsqueeze', 'Concat', 'Slice', 'Gather',
                    'Expand', 'Tile', 'Cast', 'Identity',
                    'GlobalAveragePool', 'AveragePool'):
            out_max = max((get_bound(i) for i in inputs if i), default=input_max)

        elif op == 'Constant':
            # Extract the constant value
            for attr in node.attribute:
                if attr.name == 'value' and attr.HasField('t'):
                    arr = numpy_helper.to_array(attr.t)
                    out_max = float(np.abs(arr).max())
                    break
                elif attr.name == 'value_float':
                    out_max = abs(attr.f)
                    break
                elif attr.name == 'value_int':
                    out_max = abs(attr.i)
                    break
            else:
                out_max = 1.0

        elif op == 'Max':
            out_max = max((get_bound(i) for i in inputs if i), default=input_max)

        elif op == 'Equal':
            out_max = 1.0  # boolean 0/1

        else:
            # Unknown op: conservative passthrough
            out_max = max((get_bound(i) for i in inputs if i), default=input_max)

        # Assign to all outputs (usually just one)
        for out_name in outputs:
            if out_name:
                bounds[out_name] = out_max

    # Build range dict  name -> (lo, hi)
    result = {}
    for name, max_abs in bounds.items():
        if name not in inits and name not in input_names:
            result[name] = (-max_abs, max_abs)
    # Also include inputs
    for name in input_names:
        result[name] = (-bounds[name], bounds[name])

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(ranges: dict, method: str, hotspot_threshold: int = 8,
                 total_bits: int = 16) -> list:
    """
    Build a list of report rows sorted by integer bits needed (descending).
    Returns list of dicts.
    """
    rows = []
    for name, (lo, hi) in ranges.items():
        max_abs = max(abs(lo), abs(hi))
        if max_abs == 0:
            continue
        I = min_int_bits(max_abs)
        apfixed = suggest_apfixed(max_abs, total_bits)
        rows.append({
            'name': name,
            'lo': lo,
            'hi': hi,
            'max_abs': max_abs,
            'int_bits': I,
            'apfixed': apfixed,
            'hotspot': I > hotspot_threshold,
            'method': method,
        })
    rows.sort(key=lambda r: r['int_bits'], reverse=True)
    return rows


def print_report(rows: list, hotspot_threshold: int = 8, show_all: bool = False):
    hotspots = [r for r in rows if r['hotspot']]
    normal   = [r for r in rows if not r['hotspot']]

    print(f"\n{'='*72}")
    print(f"  HOTSPOTS  (int_bits > {hotspot_threshold})  —  {len(hotspots)} tensors")
    print(f"{'='*72}")
    if hotspots:
        print(f"  {'Tensor name':<55} {'Range':>22}  {'I':>4}  {'Suggested type'}")
        print(f"  {'-'*55} {'-'*22}  {'-'*4}  {'-'*20}")
        for r in hotspots:
            flag = ' ◄◄◄' if r['int_bits'] > 12 else ''
            name_short = r['name'][-55:] if len(r['name']) > 55 else r['name']
            print(f"  {name_short:<55} {fmt_range(r['lo'],r['hi']):>22}  {r['int_bits']:>4}  {r['apfixed']}{flag}")
    else:
        print("  None — all tensors fit within the threshold.")

    if show_all:
        print(f"\n{'='*72}")
        print(f"  ALL TENSORS  ({len(rows)} total, sorted by int bits needed)")
        print(f"{'='*72}")
        print(f"  {'Tensor name':<55} {'Range':>22}  {'I':>4}  {'Suggested type'}")
        print(f"  {'-'*55} {'-'*22}  {'-'*4}  {'-'*20}")
        for r in rows:
            name_short = r['name'][-55:] if len(r['name']) > 55 else r['name']
            print(f"  {name_short:<55} {fmt_range(r['lo'],r['hi']):>22}  {r['int_bits']:>4}  {r['apfixed']}")

    return hotspots


def print_summary(rows: list):
    if not rows:
        return
    from collections import Counter
    dist = Counter(r['int_bits'] for r in rows)
    print(f"\n{'='*72}")
    print("  INTEGER BIT DISTRIBUTION")
    print(f"{'='*72}")
    for bits in sorted(dist.keys()):
        bar = '█' * dist[bits]
        print(f"  I={bits:>2}  {bar}  ({dist[bits]} tensors)")

    max_I = max(r['int_bits'] for r in rows)
    typical_I = sorted([r['int_bits'] for r in rows])[len(rows)//2]
    print(f"\n  Max integer bits seen   : {max_I}")
    print(f"  Median integer bits     : {typical_I}")
    print(f"\n  Recommendation:")
    print(f"    Activations (most tensors): ap_fixed<16,{typical_I}>")
    print(f"    Hotspot tensors           : ap_fixed<{max_I+8},{max_I}>")
    print(f"    Weights (near ±1)         : ap_fixed<16, 2> or ap_fixed<8, 2>")


def export_csv(rows: list, path: str):
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name','lo','hi','max_abs','int_bits','apfixed','hotspot','method'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Export] CSV written to {path}")


# ---------------------------------------------------------------------------
# Weaver/PyTorch dataloader helper
# ---------------------------------------------------------------------------

def collect_from_pytorch_dataloader(model_path, dataloader, input_keys,
                                    n_batches=50, output_npz=None):
    """
    Helper: collect calibration data from a weaver-style PyTorch DataLoader
    and run empirical range analysis.

    model_path  : path to .onnx file
    dataloader  : PyTorch DataLoader that yields dicts or tuples
    input_keys  : list of keys/indices to extract from each batch
    n_batches   : number of batches to collect
    output_npz  : if set, save collected data to .npz for future reuse

    Example usage:
        from weaver.utils.data import SimpleIterDataset
        # ... build your weaver dataset as usual ...
        ranges = collect_from_pytorch_dataloader(
            "model.onnx",
            val_loader,
            input_keys=["pf_points", "pf_features", "pf_mask"],
            n_batches=100,
        )
    """
    import torch
    collected = defaultdict(list)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        for key in input_keys:
            if isinstance(batch, dict):
                arr = batch[key]
            elif isinstance(batch, (list, tuple)):
                arr = batch[input_keys.index(key)]
            else:
                raise ValueError(f"Don't know how to index batch with key {key}")
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            collected[key].append(arr)

    input_data = {k: np.concatenate(v, axis=0) for k, v in collected.items()}

    if output_npz:
        np.savez(output_npz, **input_data)
        print(f"[Helper] Saved {sum(v.shape[0] for v in input_data.values())} "
              f"samples to {output_npz}")

    return empirical_ranges(model_path, input_data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find integer bit requirements for every tensor in an ONNX model."
    )
    parser.add_argument("model", help="Path to .onnx file")
    parser.add_argument("--data", help="Path to .npz calibration file")
    parser.add_argument("--input-names", nargs="+",
                        default=["pf_points", "pf_features", "pf_mask"],
                        help="Input tensor names (must match .npz keys)")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of calibration samples to run (default 200)")
    parser.add_argument("--analytic", action="store_true",
                        help="Also run analytic range propagation from weights")
    parser.add_argument("--input-max", type=float, default=1.0,
                        help="Assumed max absolute value of input tensors for analytic mode")
    parser.add_argument("--hotspot-threshold", type=int, default=8,
                        help="Integer bits above which a tensor is flagged as a hotspot (default 8)")
    parser.add_argument("--total-bits", type=int, default=16,
                        help="Total ap_fixed bit width to target (default 16)")
    parser.add_argument("--show-all", action="store_true",
                        help="Print all tensors, not just hotspots")
    parser.add_argument("--csv", help="Export results to CSV file")
    args = parser.parse_args()

    all_rows = []

    # --- Empirical mode ---
    if args.data:
        print(f"\n[Empirical] Loading calibration data from {args.data}")
        npz = np.load(args.data)
        input_data = {name: npz[name] for name in args.input_names if name in npz}
        missing = [n for n in args.input_names if n not in npz]
        if missing:
            print(f"  Warning: keys not found in .npz: {missing}")
            print(f"  Available keys: {list(npz.keys())}")

        if input_data:
            emp_ranges = empirical_ranges(args.model, input_data, args.n_samples)
            emp_rows = build_report(emp_ranges, 'empirical',
                                    args.hotspot_threshold, args.total_bits)
            print(f"\n[Empirical] Analysed {len(emp_rows)} intermediate tensors")
            print_report(emp_rows, args.hotspot_threshold, args.show_all)
            print_summary(emp_rows)
            all_rows.extend(emp_rows)
        else:
            print("  No matching input data found — skipping empirical mode.")

    # --- Analytic mode ---
    if args.analytic or not args.data:
        if not args.data:
            print("\n(No --data provided; running analytic mode only)")
        ana_ranges = analytic_ranges(args.model, args.input_max)
        ana_rows = build_report(ana_ranges, 'analytic',
                                args.hotspot_threshold, args.total_bits)
        print(f"\n[Analytic] Analysed {len(ana_rows)} tensors")
        print_report(ana_rows, args.hotspot_threshold, args.show_all)
        print_summary(ana_rows)
        all_rows.extend(ana_rows)

    # --- CSV export ---
    if args.csv and all_rows:
        export_csv(all_rows, args.csv)

    print("\nDone.")


if __name__ == "__main__":
    main()