#!/usr/bin/env python3
import onnx
import numpy as np
from onnx import helper, numpy_helper, shape_inference, TensorProto


def get_shape(vi):
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return None
    out = []
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            out.append(d.dim_value)
        else:
            out.append(None)
    return out


def build_shape_map(graph):
    sm = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        sh = get_shape(vi)
        if sh is not None:
            sm[vi.name] = sh
    return sm


def build_producer_map(graph):
    pm = {}
    for node in graph.node:
        for o in node.output:
            pm[o] = node
    return pm


def build_node_map(graph):
    nm = {}
    for node in graph.node:
        for i in node.input:
            nm.setdefault(i, []).append(node)
    return nm


def get_node_id(node):
    if node.name:
        return node.name
    return "NODE_" + "_".join(node.output)


def delete_upstream_chain(graph, start_tensor_name, producer_map):
    """
    Deletes all upstream nodes feeding start_tensor_name.
    Actually removes nodes from graph.node.
    """
    to_delete = set()
    stack = [start_tensor_name]

    while stack:
        tname = stack.pop()
        if tname not in producer_map:
            continue
        node = producer_map[tname]
        if get_node_id(node) in to_delete:
            continue
        to_delete.add(get_node_id(node))
        # Push all inputs to stack
        for inp in node.input:
            stack.append(inp)

    # Remove nodes physically
    new_nodes = [n for n in graph.node if get_node_id(n) not in to_delete]

    # Clear existing repeated field and extend
    graph.ClearField("node")
    graph.node.extend(new_nodes)


def fix_expand_transpose_sub(model):
    graph = model.graph
    shape_map = build_shape_map(graph)
    producer_map = build_producer_map(graph)
    node_map = build_node_map(graph)

    # Step 1: collect all Expand nodes first (static copy)
    expand_nodes = [n for n in graph.node if n.op_type == "Expand"]
    for i in range(len(expand_nodes)):
        node = expand_nodes[i]
        expand_out = node.output[0]

        # Step 2: find Sub consumer
        consumers = node_map.get(expand_out, [])
        sub_node = next((c for c in consumers if c.op_type == "Sub"), None)
        if sub_node is None:
            continue

        # Step 3: find Transpose input of Sub
        other_input = next((i for i in sub_node.input if i != expand_out), None)
        if other_input is None:
            continue

        trans_node = producer_map.get(other_input)
        if trans_node is None or trans_node.op_type != "Transpose":
            continue

        # Step 4: get static shape from Transpose
        trans_shape = shape_map.get(trans_node.output[0])
        if trans_shape is None or None in trans_shape:
            print(f"[WARN] Transpose output shape not static for Expand {node.name}")
            continue

        print(f"[INFO] Patching Expand '{node.name}' shape to {trans_shape}")

        # Step 5: create Constant for Expand shape
        const_name = node.name + "_shape_const"
        const_tensor = numpy_helper.from_array(
            np.array(trans_shape, dtype=np.int64), name=const_name
        )
        graph.initializer.append(const_tensor)

        # Step 6: replace Expand shape input
        original_shape_input = node.input[1]
        node.input[1] = const_name

        # Step 7: delete upstream nodes that produced original shape
        delete_upstream_chain(graph, original_shape_input, producer_map)

        # Step 8: rebuild maps after deletion to be safe
        shape_map = build_shape_map(graph)
        producer_map = build_producer_map(graph)
        node_map = build_node_map(graph)
        expand_nodes = [n for n in graph.node if n.op_type == "Expand"]

    return model


def fix_training_mode(model):
    for node in model.graph.node:
        new_attr = [a for a in node.attribute if a.name != "training_mode"]
        del node.attribute[:]
        node.attribute.extend(new_attr)
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load_model(args.model)
    model = fix_training_mode(model)
    onnx.checker.check_model(model)
    model = fix_expand_transpose_sub(model)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, args.out)

    print(f"[DONE] Saved patched model to {args.out}")
