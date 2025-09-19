import onnx
import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from onnx import helper, numpy_helper
import numpy as np


def force_topk_k(onnx_in, onnx_out, fixed_k=16):
    model = onnx.load(onnx_in)
    graph = model.graph

    # 1. Create a single constant initializer for k
    k_name = "const_k{}".format(fixed_k)
    k_tensor = numpy_helper.from_array(
        np.array([fixed_k], dtype=np.int64), name=k_name
    )
    graph.initializer.append(k_tensor)

    # 2. Remove any Constant nodes that were feeding into TopK
    keep_nodes = []
    for node in graph.node:
        if node.op_type == "Constant" and any("output_0" in o for o in node.output):
            # skip this constant (candidate for k)
            print("Removing Constant node:", node.name or node.output)
            continue
        keep_nodes.append(node)

    # 3. Rewire TopK inputs to use our initializer
    new_nodes = []
    for node in keep_nodes:
        if node.op_type == "TopK":
            print("Rewiring TopK:", node.name or node.output)
            data_input = node.input[0]
            node.input[:] = [data_input, k_name]
        new_nodes.append(node)

    # Replace graph nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.save(model, onnx_out)
    print(f"Saved patched model to {onnx_out}")



def main(onnx_path, output_path="model.c"):
    # Load ONNX
    onnx_model = onnx.load(onnx_path)

    for node in onnx_model.graph.node:
        if node.op_type == "TopK":
            print(node)

    # Extract true input names + shapes
    shape_dict = {}
    for input_tensor in onnx_model.graph.input:
        dims = []
        for d in input_tensor.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                # unknown dim (batch size), set to 1
                dims.append(1)
        shape_dict[input_tensor.name] = tuple(dims)
    print("Detected input shapes:", shape_dict)

    shape_dict = {'pf_features': (1, 4, 35), 
                  'pf_points': (1, 3, 35), 
                  'pf_mask': (1, 1, 35),}
    # Convert ONNX → Relay
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # Build with AOT executor + CRT runtime
    executor = Executor("aot")   # ahead-of-time executor (not graph)
    runtime = Runtime("crt")     # C runtime
    target = "c"

    with tvm.transform.PassContext(opt_level=3,
                                   config={"tir.disable_vectorize": True}):
        lib = relay.build(mod, target=target, executor=executor,
                          runtime=runtime, params=params)

    # Dump the generated C source
    c_src = lib.get_source("c")
    with open(output_path, "w") as f:
        f.write(c_src)

    print(f"C source written to {output_path}")

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python tvm_export_to_c.py /path/to/model.onnx")
    #     sys.exit(1)
    # onnx_path = sys.argv[1]
    onnx_path = "pure_B_plus_B_minus/gen_level_1B/with_partial_vertex/training_log/particlenet_complete.onnx"
    fixed_onnx_path = "particlenet_fixed.onnx"
    force_topk_k(onnx_path, fixed_onnx_path, fixed_k=17)
    main(fixed_onnx_path)
