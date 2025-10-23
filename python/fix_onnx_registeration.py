import onnx
from onnx import helper
import onnx.shape_inference

# Load and infer shapes
model = onnx.load("/home/tpyang/B-Reconstruction-2-FPGA/python/pure_B_plus_B_minus/gen_level_1B/with_vertex/training_log/particlenet_complete.onnx")
try:
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    print("Initial shape inference succeeded.")
except Exception as e:
    print(f"Initial shape inference failed: {e}")

graph = model.graph

# Targets: Add_1 inputs (2 tensors), output, and ReLU output
add_output = "/mod/edge_convs.0/Add_1_output_0"
relu_output = "/mod/edge_convs.0/sc_act/Relu_output_0"

# Add_1 inputs: Replace these placeholders with actual names from inspect_onnx.py/Netron
add_inputs = [
    "/mod/edge_convs.0/sc/Conv_output_0",  # Placeholder for first input (e.g., Conv output)
    "/mod/edge_convs.0/ReduceMean_output_0"            # Placeholder for second input (e.g., bias)
]
print(f"Assuming Add_1 inputs: {add_inputs}—replace with actual names from inspection!")

# Function to add/update value_info for a tensor
def add_value_info(graph, tensor_name, elem_type=onnx.TensorProto.FLOAT, shape=None):
    if shape is None:
        shape = [1, -1, 64]  # Default for Add output
        print(f"Using shape {shape} for {tensor_name}—adjust based on Netron if needed!")

    tensor_type = helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
    value_info = helper.make_value_info(tensor_name, tensor_type)
    
    # Protobuf-safe removal: Clear the field and re-add all except the target
    new_value_info = []
    for vi in graph.value_info:
        if vi.name != tensor_name:
            new_value_info.append(vi)
    
    graph.ClearField('value_info')  # Clear the repeated field properly
    
    # Re-add existing ones
    for vi in new_value_info:
        graph.value_info.add().CopyFrom(vi)
    
    # Add the new/updated one
    graph.value_info.add().CopyFrom(value_info)
    print(f"Added/updated value_info for {tensor_name}: FLOAT, shape {shape}")

# Add for Add_1 inputs (2 tensors)
add_value_info(graph, add_inputs[0], shape=[1, -1, 4])  # First input (e.g., Conv output: input channels=4 from W [64, 4, 1])
add_value_info(graph, add_inputs[1], shape=[1, -1, 4], elem_type=onnx.TensorProto.FLOAT)  # Second input (e.g., bias: [64])

# Add for Add_1 output (ReLU input)
add_value_info(graph, add_output, shape=[1, -1, 64])  # Output channels=64 from W

# Add for ReLU output (same as input)
add_value_info(graph, relu_output, shape=[1, -1, 64])

# Re-run shape inference to propagate types after modifications
try:
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    print("Post-modification shape inference succeeded—types propagated.")
except Exception as e:
    print(f"Post-modification shape inference failed: {e}")

# Save fixed model
try:
    onnx.checker.check_model(model)
    onnx.save(model, "registered_model_fixed.onnx")
    print("Fixed model saved to 'registered_model_fixed.onnx'.")
except Exception as e:
    print(f"Validation failed: {e}")

# Quick re-inspection
print("\nQuick re-inspection of key tensors:")
for vi in graph.value_info:
    if any(t in vi.name for t in [add_output, relu_output] + add_inputs):
        shape_str = [dim.dim_value if dim.dim_value > 0 else f"({dim.dim_param})" for dim in vi.type.tensor_type.shape.dim]
        elem_name = onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
        print(f"Tensor: {vi.name}, Type: {elem_name}, Shape: {shape_str}")