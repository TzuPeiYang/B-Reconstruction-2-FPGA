import tvm
from tvm import relay, transform
import onnx
import numpy as np
from onnx import shape_inference


if __name__ == "__main__":
    # Load ONNX model
    onnx_path = "pure_B_plus_B_minus/gen_level_1B/with_vertex/training_log/particlenet_complete.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx_model = shape_inference.infer_shapes(onnx_model)

    # Input shape (from inspection/Netron, e.g., batch=1, seq=32, features=4 for Conv input)
    shape_dict = {"pf_points": (1, 3, 35),
                  "pf_features": (1, 4, 35),
                  "pf_mask": (1, 1, 35)}  # Adjust 'input_name' and shape

    # Convert to Relay IR
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # Apply optimizations with opt_level=2 (fusion + folding without vectorization)
    with transform.PassContext(opt_level=2, config={"tir.disable_vectorize": True}):
        optimized_mod = mod  # Optimizes in-place during build

    # Build for C codegen (plain C source, scalar to avoid vector errors)
    target = "c"
    with transform.PassContext(opt_level=2, config={"tir.disable_vectorize": True}):
        lib = relay.build(optimized_mod, target=target, params=params)

    # Export plain C source files (no compilation)
    lib.export_library("model", fcompile=False)  # Outputs model.c, model.h, etc.
    print("Plain C source generated as 'model.c' and 'model.h' (optimized with fusion, scalar C).")

    # Test inference (Python verification using LLVM for vector support)
    dev = tvm.device("llvm", 0)
    test_target = "llvm -mtriple=x86_64-linux-gnu"
    with transform.PassContext(opt_level=2):
        test_lib = relay.build(optimized_mod, target=test_target, params=params)
        test_module = tvm.runtime.load_module(test_lib.export_library("test_model.so"))
        ex = relay.create_executor("graph", mod=test_module, device=dev, params=params)
        input_data = tvm.nd.array(np.random.randn(*shape_dict["input_name"]).astype(np.float32), dev)
        output = ex.evaluate()(input_data)
    print("Test output shape:", output.shape)
    print("Sample output:", output.numpy()[0, 0, :5])  # First 5 elements