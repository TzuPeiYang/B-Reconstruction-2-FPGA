import onnx
import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from onnx import helper, numpy_helper
import numpy as np


def fix_topk_k(onnx_in, onnx_out, fixed_k=16):
    model = onnx.load(onnx_in)
    graph = model.graph

    new_nodes = []
    for node in graph.node:
        if node.op_type == "TopK":
            print("Patching TopK:", node.name or "(unnamed)")

            # Always keep 2 inputs, but make k a constant initializer
            data_input = node.input[0]
            k_const_name = node.name + "_k" if node.name else "const_k"

            # Create initializer tensor for k
            k_tensor = numpy_helper.from_array(
                np.array([fixed_k], dtype=np.int64),
                name=k_const_name
            )
            graph.initializer.append(k_tensor)

            # Replace inputs with [data_input, k_const_name]
            new_node = helper.make_node(
                "TopK",
                inputs=[data_input, k_const_name],
                outputs=node.output,
                name=node.name
            )
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    # Replace graph nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.save(model, onnx_out)
    print(f"Saved patched model with fixed k={fixed_k} to {onnx_out}")

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
    onnx_path = "particlenet_complete_simplified.onnx"
    fixed_onnx_path = "particlenet_fixed.onnx"
    fix_topk_k(onnx_path, fixed_onnx_path, fixed_k=16)
    main(fixed_onnx_path)
