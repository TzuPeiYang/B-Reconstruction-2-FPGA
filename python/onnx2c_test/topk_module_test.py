import torch
import torch.nn as nn

class TopKModel(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def forward(self, x):
        # Apply TopK along the last dimension
        values, indices = torch.topk(x, self.k, dim=-2)
        return values, indices


# Instantiate model
model = TopKModel(k=3)
model.eval()

# Example input
x = torch.randn(1, 5, 10)  # shape (batch, features, length)
print("Input:", x)

# Run once
values, indices = model(x)
print("Output values:", values)
print("Output indices:", indices)

# Export to ONNX
torch.onnx.export(
    model,
    x,
    "topk_test.onnx",
    input_names=["input"],
    output_names=["topk_values", "topk_indices"],
    opset_version=13,          # 13+ recommended for TopK
    do_constant_folding=True,
    dynamic_axes=None
)

print("✅ Exported model to topk_test.onnx")