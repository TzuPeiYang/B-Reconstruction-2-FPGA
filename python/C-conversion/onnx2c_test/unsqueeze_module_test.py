import torch
import torch.nn as nn
import os 
import time

class UnsqueezeModel(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.expand_shape = torch.tensor([1, 5, 10, k], dtype=torch.int64)

    def forward(self, x):
        shape = self.expand_shape.to(x.device)
        expanded = x.unsqueeze(3).expand(shape.tolist())
        return expanded


# Instantiate model
model = UnsqueezeModel(k=10)
model.eval()

# Example input
x = torch.randn(1, 5, 10)  # shape (batch, features, length)
print("Input:", x)

# Run once
output = model(x)
print("Output Shape:", output.shape)

# Export to ONNX
exported_program = torch.export.export(model, args=(x,))
torch.onnx.export(
    exported_program,
    x,
    "./rrrrrrrr.onnx",
    opset_version=18,  # Or 17—new exporter supports up to 19
    dynamic_shapes=None,  # ← Key: forces full static shapes, errors if impossible
    do_constant_folding=True,
    input_names=["Input"], output_names=["Output"]
)

for _ in range(10):
    if os.path.exists("./rrrrrrrr.onnx"):
        print("ONNX file created successfully!")
        print("Size:", os.path.getsize("./rrrrrrrr.onnx") / 1e6, "MB")
        break
    time.sleep(0.1)
else:
    print("File still not there → export failed silently")