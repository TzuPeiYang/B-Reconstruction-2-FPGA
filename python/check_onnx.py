import onnx
from onnx import helper
import onnx.shape_inference

# Load and infer shapes/types
model = onnx.load("/home/tpyang/B-Reconstruction-2-FPGA/python/pure_B_plus_B_minus/gen_level_1B/with_vertex/training_log/particlenet_complete.onnx")
try:
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)  # Enhanced inference with data prop
    print("Shape inference succeeded.")
except Exception as e:
    print(f"Shape inference failed: {e}")

graph = model.graph

# Check opset version
print(f"Model opset: {model.opset_import[0].version}")

# Inspect ReLU and TopK nodes
for node in graph.node:
    if node.op_type == "Relu":
        print(f"ReLU Node: {node.name or 'unnamed'}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
        print(f"  Attributes: {[(attr.name, attr.i) for attr in node.attribute]}")
        # Check input types and shapes
        for inp in node.input:
            type_str = "Not found"
            shape_str = "N/A"
            for value in graph.value_info:
                if value.name == inp:
                    type_str = f"elem_type: {value.type.tensor_type.elem_type} ({onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)})"
                    if value.type.tensor_type.HasField('shape'):
                        shape_str = f"Shape: {[dim.dim_value if dim.dim_value > 0 else f'({dim.dim_param})' for dim in value.type.tensor_type.shape.dim]}"
                    break
            print(f"  Input '{inp}' Type: {type_str}, {shape_str}")
    elif node.op_type == "TopK":
        print(f"TopK Node: {node.name or 'unnamed'}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
        print(f"  Attributes: {[(attr.name, attr.i) for attr in node.attribute]}")
        # Similar type/shape check for TopK inputs
        for inp in node.input:
            type_str = "Not found"
            shape_str = "N/A"
            for value in graph.value_info:
                if value.name == inp:
                    type_str = f"elem_type: {value.type.tensor_type.elem_type} ({onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)})"
                    if value.type.tensor_type.HasField('shape'):
                        shape_str = f"Shape: {[dim.dim_value if dim.dim_value > 0 else f'({dim.dim_param})' for dim in value.type.tensor_type.shape.dim]}"
                    break
            print(f"  Input '{inp}' Type: {type_str}, {shape_str}")
    print("---")

# Inspect all tensors
print("\nAll Inferred Tensors:")
for value in graph.value_info:
    type_str = f"elem_type: {value.type.tensor_type.elem_type} ({onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)})"
    shape_str = "N/A"
    if value.type.tensor_type.HasField('shape'):
        shape_str = f"Shape: {[dim.dim_value if dim.dim_value > 0 else f'({dim.dim_param})' for dim in value.type.tensor_type.shape.dim]}"
    print(f"Tensor: {value.name}, {type_str}, {shape_str}")

# Save inferred model for SOFIE
onnx.save(model, "inferred_model.onnx")
print("\nInferred model saved as 'inferred_model.onnx' for SOFIE testing.")