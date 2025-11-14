import argparse
import os
import sys
import onnx
from onnx import helper, TensorProto, shape_inference, numpy_helper
from collections import defaultdict, deque

# Define supported operations for onnx2c
SUPPORTED_OPS = ['Abs', 'Acos', 'Acosh', 'Add', 'And', 'Asin', 'Asinh', 'Atan', 'Atanh', 'AveragePool', 
                 'BatchNormalization', 'BitShift', 'Cast', 'Ceil', 'Celu', 'Clip', 'Concat', 'Constant', 
                 'ConstantOfShape', 'Conv', 'Cos', 'Cosh', 'ConvInteger', 'ConvTranspose', 'DequantizeLinear', 
                 'Div', 'Dropout', 'DynamicQuantizeLinear', 'Flatten', 'Floor', 'Elu', 'Equal', 'Erf', 
                 'Exp', 'Expand', 'Gather', 'Gemm', 'GlobalAveragePool', 'GlobalMaxPool', 'Greater', 
                 'GreaterOrEqual', 'HardSigmoid', 'HardSwish', 'Identity', 'InstanceNormalization', 'LayerNormalization', 
                 'LeakyRelu', 'Less', 'LessOrEqual', 'Log', 'LogSoftmax', 'LRN', 'LSTM', 'MatMul', 'MatMulInteger', 'Max', 
                 'MaxPool', 'Mean', 'Min', 'Mod', 'Mul', 'Neg', 'Not', 'Or', 'Pad', 'Pow', 'PRelu', 'QuantizeLinear', 
                 'RandomUniform', 'Range', 'ReduceProd', 'ReduceMean', 'ReduceSumSquare', 'ReduceMax', 'ReduceMin', 
                 'ReduceSum', 'ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceLogSumExp', 'Reciprocal', 'Relu', 'Reshape', 
                 'Resize', 'Round', 'ScatterND', 'Selu', 'Shape', 'Shrink', 'Sigmoid', 'Sign', 'Sin', 'Sinh', 'Slice', 
                 'Softplus', 'Softsign', 'Softmax', 'Split', 'Squeeze', 'Sqrt', 'Sub', 'Sum', 'Tan', 'Tanh', 'Transpose', 
                 'TreeEnsembleClassifier', 'ThresholdedRelu', 'Unsqueeze', 'Upsample', 'Where', 'Xor'
                ]


def load_and_infer(path):
    model = onnx.load(path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[warning] shape_inference failed: {e}")
    return model


def build_tensor_producers_consumers(graph):
    producers = defaultdict(list) # tensor -> [node]
    consumers = defaultdict(list) # tensor -> [node]
    name_to_node = {}
    for node in graph.node:
        node_id = node.name if node.name else f"{node.op_type}_{id(node)}"
        name_to_node[node_id] = node
        for t in node.output:
            producers[t].append(node_id)
        for t in node.input:
            consumers[t].append(node_id)
    return producers, consumers, name_to_node


def supported_node_ids(graph, supported_ops):
    ids = []
    id_to_node = {}
    for node in graph.node:
        node_id = node.name if node.name else f"{node.op_type}_{id(node)}"
        id_to_node[node_id] = node
        if node.op_type in supported_ops:
            ids.append(node_id)
    return ids, id_to_node


def build_supported_components(graph, supported_ops):
    producers, consumers, name_to_node = build_tensor_producers_consumers(graph)
    supported_ids, id_to_node = supported_node_ids(graph, supported_ops)
    supported_set = set(supported_ids)

    # adjacency (undirected) among supported nodes
    adj = defaultdict(set)
    for t, prod_nodes in producers.items():
        cons_nodes = consumers.get(t, [])
        # for every producer and consumer that are both supported, connect them
        for p in prod_nodes:
            for c in cons_nodes:
                if p in supported_set and c in supported_set:
                    adj[p].add(c)
                    adj[c].add(p)

    # also consider nodes that share the same input (fanout) or share outputs? above covers producer-consumer
    visited = set()
    components = []
    for nid in supported_ids:
        if nid in visited:
            continue

        comp = []
        dq = deque([nid])
        visited.add(nid)
        while dq:
            cur = dq.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    dq.append(nb)
        components.append(comp)
    return components, id_to_node


def tensors_from_nodes(node_list, id_to_node):
    produced = set()
    consumed = set()
    for nid in node_list:
        node = id_to_node[nid]
        produced.update([t for t in node.output if t])
        consumed.update([t for t in node.input if t])
    return produced, consumed


def typed_names_set(model):
    names = set()
    for v in model.graph.value_info:
        names.add(v.name)
    for v in model.graph.input:
        names.add(v.name)
    for v in model.graph.output:
        names.add(v.name)
    return names


def find_initializer_shape(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            return list(arr.shape)
    return None


def find_value_info_shape(model, name):
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == name:
            try:
                dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                # if a dim is unknown (0), keep as None
                dims = [d if d != 0 else None for d in dims]
                return dims
            except Exception:
                return None
    return None


def add_value_info_if_missing(model, name, prefer_shape=None):
    # do not overwrite existing value_info
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == name and vi.type is not None and vi.type.HasField('tensor_type'):
            return # exists
    # determine shape: prefer prefer_shape, else initializer, else None
    shape = None
    if prefer_shape is not None:
        shape = prefer_shape
    else:
        shape = find_value_info_shape(model, name)
        if shape is None:
            shape = find_initializer_shape(model, name)
    # If still None, choose fallback: scalar if initializer exists and has [] shape, else [1]
    if shape is None:
        init_shape = find_initializer_shape(model, name)
        if init_shape == []:
            shape = [] # scalar
        else:
            shape = [1]
    # create value_info with TensorProto.FLOAT by default
    vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    model.graph.value_info.append(vi)


def extract_components(model, components, id_to_node, outdir, base_model_path):
    os.makedirs(outdir, exist_ok=True)
    producers, consumers, _ = build_tensor_producers_consumers(model.graph)
    typed_names = typed_names_set(model)

    extracted = []
    for i, comp in enumerate(components):
        produced, consumed = tensors_from_nodes(comp, id_to_node)
        seg_inputs = set()
        for nid in comp:
            node = id_to_node[nid]
            for t in node.input:
                if not t:
                    continue
                if t not in produced:
                    seg_inputs.add(t)

        # segment outputs are produced tensors consumed by nodes outside the component OR that are graph outputs
        seg_outputs = set()
        # any produced tensor that appears as input to a node not in comp
        for t in produced:
            outside_consumers = [c for c in consumers.get(t, []) if c not in comp]
            if outside_consumers:
                seg_outputs.add(t)

        # include produced tensors that are graph outputs
        graph_outputs = set(o.name for o in model.graph.output)
        seg_outputs |= (produced & graph_outputs)

        if not seg_outputs:
            print(f"[skip] component {i} has no external outputs (internal-only)")
            continue

        # Ensure every boundary tensor has value_info / type
        for t in list(seg_inputs) + list(seg_outputs):
            add_value_info_if_missing(model, t)

        start_names = list(seg_inputs)
        end_names = list(seg_outputs)


        out_path = os.path.join(outdir, f"segment_{i}.onnx")
        try:
            onnx.utils.extract_model(base_model_path, out_path, start_names, end_names)
            print(f"[ok] extracted segment_{i} -> {out_path} (nodes={len(comp)})")
            extracted.append(out_path)
        except Exception as e:
            print(f"[error] failed to extract segment_{i}: {e}")
    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--outdir", default="onnx_segments", help="Output directory for segments")
    parser.add_argument("--supported", default=None,
    help="Comma-separated list of supported ops (overrides built-in set)")
    args = parser.parse_args()

    if args.supported:
        supported = set([s.strip() for s in args.supported.split(",") if s.strip()])
    else:
        supported = SUPPORTED_OPS

    print(f"Loading model: {args.model}")
    model = load_and_infer(args.model)

    # Save inferred model copy we will reference for extraction
    inferred_path = os.path.join(args.outdir, "model_inferred.onnx")
    os.makedirs(args.outdir, exist_ok=True)
    onnx.save(model, inferred_path)
    print(f"Saved inferred model to: {inferred_path}")

    # Build components
    components, id_to_node = build_supported_components(model.graph, supported)
    print(f"Identified {len(components)} supported components")

    # Extract components
    extracted = extract_components(model, components, id_to_node, args.outdir, inferred_path)
    print(f"Extraction complete. {len(extracted)} segments written to {args.outdir}")


if __name__ == "__main__":
    main()