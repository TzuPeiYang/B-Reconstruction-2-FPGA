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


def patch_expand_shape_by_eval(model):
    graph = model.graph

    # Build producer map
    producer_map = {}
    for node in graph.node:
        for out in node.output:
            producer_map[out] = node

    # Map initializers
    initializer_map = {init.name: numpy_helper.to_array(init)
                       for init in graph.initializer}

    expand_nodes = [n for n in graph.node if n.op_type == "Expand"]

    for expand in expand_nodes:

        shape_input = expand.input[1]

        try:
            shape_value = evaluate_tensor(
                shape_input,
                producer_map,
                initializer_map
            )
        except Exception:
            continue

        if shape_value is None:
            continue

        print(f"[INFO] Patching {expand.name} with shape {shape_value}")

        const_name = expand.name + "_static_shape"
        const_tensor = numpy_helper.from_array(
            shape_value.astype(np.int64),
            name=const_name
        )
        graph.initializer.append(const_tensor)

        expand.input[1] = const_name

    return model


def evaluate_tensor(tensor_name, producer_map, initializer_map):
    """
    Recursively evaluate tensor if it is constant-only subgraph.
    """

    # If initializer → return directly
    if tensor_name in initializer_map:
        return initializer_map[tensor_name]

    node = producer_map.get(tensor_name)
    if node is None:
        return None

    # Recursively evaluate inputs
    inputs = []
    for inp in node.input:
        val = evaluate_tensor(inp, producer_map, initializer_map)
        if val is None:
            return None
        inputs.append(val)

    # Evaluate supported ops
    if node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                return numpy_helper.to_array(attr.t)

    if node.op_type == "ConstantOfShape":
        shape = inputs[0]
        value = 0
        for attr in node.attribute:
            if attr.name == "value":
                value = numpy_helper.to_array(attr.t).item()
        return np.full(shape.astype(int), value)

    if node.op_type == "Mul":
        return inputs[0] * inputs[1]

    if node.op_type == "Equal":
        return inputs[0] == inputs[1]

    if node.op_type == "Where":
        return np.where(inputs[0], inputs[1], inputs[2])

    return None


def prune_disconnected_graph(model):
    graph = model.graph

    # Build producer map
    producer_map = {}
    for node in graph.node:
        for out in node.output:
            producer_map[out] = node

    # Start from graph outputs
    required_nodes = set()
    stack = [o.name for o in graph.output]

    while stack:
        tensor = stack.pop()
        node = producer_map.get(tensor)
        if node is None:
            continue

        node_id = id(node)
        if node_id in required_nodes:
            continue

        required_nodes.add(node_id)

        # Traverse inputs
        for inp in node.input:
            stack.append(inp)

    # Keep only required nodes
    new_nodes = [
        n for n in graph.node
        if id(n) in required_nodes
    ]

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    # Remove unused initializers
    used_inputs = set()
    for node in graph.node:
        for inp in node.input:
            used_inputs.add(inp)

    new_inits = [
        init for init in graph.initializer
        if init.name in used_inputs
    ]

    graph.ClearField("initializer")
    graph.initializer.extend(new_inits)

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
    model = patch_expand_shape_by_eval(model)
    onnx.checker.check_model(model)
    model = prune_disconnected_graph(model)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, args.out)

    print(f"[DONE] Saved patched model to {args.out}")
