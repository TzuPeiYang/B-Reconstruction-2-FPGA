import onnx
import onnx.shape_inference
import onnx.utils
import onnxruntime as ort

model = onnx.load("./pure_B_plus_B_minus/gen_level_1B/with_vertex/training_log/particlenet_complete.onnx")

for node in model.graph.node:
    # Collect all attributes except training_mode
    keep_attrs = [a for a in node.attribute if a.name != "training_mode"]
    # Clear and re-append
    node.ClearField("attribute")
    node.attribute.extend(keep_attrs)

model = onnx.shape_inference.infer_shapes(model)

# Try constant folding: evaluate all subgraphs that can be precomputed
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)

onnx.save(model, "./test/model_folded.onnx")
print("✅ Saved model with static shapes to model_folded.onnx")

