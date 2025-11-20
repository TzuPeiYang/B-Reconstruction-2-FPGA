#!/usr/bin/env python3
import onnx
import numpy as np
from onnx import helper, numpy_helper, shape_inference


"""
Patches expand shape input and remove training_mode attribute from all nodes
"""

def manual_shape_override(node_name, shape_input_name):
    """
    Called when automatic shape inference fails.

    Return:
        - A list of ints for static shape, e.g. [32,64,10,1]
        - Or None if you do not want to override
    """

    # Example:
    if node_name == "/mod/edge_convs.0/Expand":
        return [1, 4, 35, 16]
    if node_name == "/mod/edge_convs.1/Expand":
        return [1, 64, 35, 16]
    if node_name == "/mod/edge_convs.2/Expand":
        return [1, 128, 35, 16]

    # By default, no override:
    return None


def load_model(path):
    model = onnx.load(path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print("[WARN] ONNX shape inference failed:", e)
    return model


def get_tensor_shape_from_value_info(model, name):
    """
    Returns:
        - list[int] = valid static shape
        - None = dynamic, unknown, or bogus ONNX rank-only inference

    Adds detection for ONNX "rank-only" junk shape inference, e.g. [4].
    """

    # search value_info, inputs, outputs
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name != name:
            continue

        shp = vi.type.tensor_type.shape

        dims = []
        has_valid_dim = False
        has_any_dim_value = False

        for d in shp.dim:
            if d.HasField("dim_value"):
                has_any_dim_value = True
                if d.dim_value > 1:
                    has_valid_dim = True
                dims.append(int(d.dim_value))
            else:
                # dynamic
                dims.append(None)

        # -------------------------
        # Detect ONNX bogus "shape = [rank]"
        # -------------------------

        # case 1: ONNX gave only a single number equal to rank
        if len(dims) == 1:
            return None    # invalid, skip

        # case 2: all dim_values are tiny (<=1) → almost always bogus
        if has_any_dim_value and not has_valid_dim:
            return None

        # case 4: all dims are None or symbolic
        if all(d is None for d in dims):
            return None

        # If any real dimensions exist, return only the real dims.
        # Replace None with 1 (safe broadcast anchor)
        final = [d if d is not None else 1 for d in dims]

        return final

    return None


def build_producer_map(graph):
    """Map tensor -> node that produces it"""
    producer = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node
    return producer


def delete_upstream_chain(graph, start_tensor, producer_map, safe_tensors):
    """
    Delete all nodes producing 'start_tensor' unless their outputs
    are used elsewhere or are protected.
    """
    if start_tensor not in producer_map:
        return

    stack = [start_tensor]
    to_delete = {}   # id(node) -> node

    while stack:
        t = stack.pop()
        if t not in producer_map:
            continue

        node = producer_map[t]
        nid = id(node)

        if nid in to_delete:
            continue

        # Check if this tensor is consumed anywhere else
        used_elsewhere = False
        for consumer in graph.node:
            if consumer is node:
                continue
            if t in consumer.input:
                used_elsewhere = True
                break

        # Do not delete if used elsewhere or protected
        if used_elsewhere or t in safe_tensors:
            continue

        # Mark node for deletion
        to_delete[nid] = node

        # Traverse its inputs
        for inp in node.input:
            stack.append(inp)

    # Apply deletion
    for node in to_delete.values():
        try:
            graph.node.remove(node)
        except ValueError:
            pass


def fix_expand_nodes(model):
    g = model.graph
    producer = build_producer_map(g)

    init_names = {init.name for init in g.initializer}
    fixed = 0

    for node in list(g.node):
        if node.op_type != "Expand":
            continue

        shape_input = node.input[1]

        # Skip if already constant
        if shape_input in init_names:
            continue

        # Try automatic inference
        inferred_shape = get_tensor_shape_from_value_info(model, shape_input)

        if inferred_shape is None:
            # Try user-supplied override
            override = manual_shape_override(node.name or "<no_name>", shape_input)

            if override is None:
                print(f"[SKIP] Expand {node.name}: dynamic shape, no inference, no override")
                continue
            else:
                print(f"[OVERRIDE] Expand {node.name}: using manual shape {override}")
                inferred_shape = override

        else:
            print(f"[PATCH] Expand {node.name}: inferred static shape {inferred_shape}")

        # Make constant initializer for new shape
        const_name = f"{shape_input}_static"
        arr = np.array(inferred_shape, dtype=np.int64)

        g.initializer.append(
            numpy_helper.from_array(arr, name=const_name)
        )

        # Patch Expand node
        node.input[1] = const_name

        # Delete upstream dynamic shape chain
        delete_upstream_chain(
            g,
            shape_input,
            producer,
            safe_tensors=set(init_names)
        )

        fixed += 1

    print(f"[INFO] Patched {fixed} Expand nodes.\n")
    return model


def fix_training_mode(model):
    for node in model.graph.node:
        new_attr = [a for a in node.attribute if a.name != "training_mode"]
        del node.attribute[:]
        node.attribute.extend(new_attr)
    return model

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = load_model(args.model)
    model = fix_expand_nodes(model)
    model = fix_training_mode(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, args.out)

    print(f"[DONE] Saved patched model to {args.out}")


if __name__ == "__main__":
    main()
