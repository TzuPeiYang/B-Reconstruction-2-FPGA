import torch
import numpy as np
import sys


def tensor_to_c_array(name, tensor):
    arr = tensor.cpu().numpy()
    flat = arr.flatten()
    c_name = name.replace('.', '_')
    array_str = ", ".join([f"{v:.8e}f" for v in flat])  # <-- fixed
    return c_name, array_str, flat.size, arr.shape

def export_to_header(sd, header_path):
    with open(header_path, "w") as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write("// Auto-generated from PyTorch model\n\n")

        for name, tensor in sd.items():
            c_name, array_str, size, shape = tensor_to_c_array(name, tensor)

            # Write shape metadata
            f.write(f"#define {c_name.upper()}_SIZE {size}\n")
            f.write(f"#define {c_name.upper()}_DIMS {len(shape)}\n")
            for i, dim in enumerate(shape):
                f.write(f"#define {c_name.upper()}_SHAPE_{i} {dim}\n")

            # Write array
            f.write(f"static const float {c_name}[{size}] = {{ {array_str} }};\n\n")

        f.write("#endif // MODEL_WEIGHTS_H\n")


if __name__ == "__main__":
    MODEL_PATH = sys.argv[1]
    sd = torch.load(MODEL_PATH, map_location="cpu")

    HEADER_PATH = "../C/include/model_weights.h"
    export_to_header(sd, HEADER_PATH)
    print(f"Export complete → {HEADER_PATH}")