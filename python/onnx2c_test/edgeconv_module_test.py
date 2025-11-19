import torch
import torch.nn as nn
import torch.nn.functional as F
from weaver.nn.model.ParticleNet import ParticleNet
import numpy as np
import onnx
from .. import tensor_2_header


def format_nested_array(arr, indent=4):
    """
    Recursively formats a numpy array as nested C initializer lists.
    """
    if arr.ndim == 1:
        return ", ".join(f"{x:.9g}f" for x in arr)

    indent_str = " " * indent
    inner = []
    for sub in arr:
        formatted = format_nested_array(sub, indent + 4)
        inner.append(f"{{ {formatted} }}")

    return (",\n" + indent_str).join(inner)


def tensor_to_c_multidim_header(tensor: torch.Tensor, name: str, header_path: str):
    """
    Export a PyTorch tensor to a C header file as a multidimensional float array.
    """

    arr = tensor.detach().cpu().numpy().astype(np.float32)
    shape = arr.shape

    # C identifier names
    array_name = name.lower()
    macro_name = name.upper() + "_NDIMS"
    size_macro = name.upper() + "_TOTAL_SIZE"

    # Total number of elements
    total_elems = arr.size

    # Shape string for C
    shape_str = "".join(f"[{s}]" for s in shape)

    # Format the array content
    formatted_array = format_nested_array(arr)

    # Header file content
    header = f"""#ifndef {name.upper()}_H
    #define {name.upper()}_H

    #include <stdint.h>

    #define {macro_name} {len(shape)}
    #define {size_macro} {total_elems}

    static const float {array_name}{shape_str} = {{
        {formatted_array}
    }};

    #endif // {name.upper()}_H
    """

    with open(header_path, "w") as f:
        f.write(header)

    print(f"[OK] Wrote {header_path} (shape={shape}, total={total_elems})")


if __name__ == "__main__":
    pf_points = torch.randn(8, 3, 10)
    pf_features = torch.randn(8, 4, 10)

    tensor_to_c_multidim_header(pf_points, name="pf_points", header_path="../../C/include/pf_points.h")
    tensor_to_c_multidim_header(pf_features, name="pf_features", header_path="../../C/include/pf_features.h")

    edge_conv = EdgeConvBlock(3, 4, [8], cpu_mode=True)
    edge_conv.load_state_dict(torch.load("edge_conv_test.pt"))
    edge_conv.eval()
    # torch.save(edge_conv.state_dict(), "edge_conv_test.pt")
    output = edge_conv(pf_points, pf_features)
    for i in range(8):
        for j in range(8):
            for k in range(10):
                print(output[i][j][k].item())
    
    # export_program = torch.export.export(edge_conv, (pf_points, pf_features))
    torch.onnx.export(
        edge_conv,
        (pf_points, pf_features),
        "./edgeconv_test.onnx",
        input_names=["pf_points", "pf_features"],
        output_names=["output"],
        do_constant_folding=True,
        opset_version=18,
        export_params=True,
    )

    model = onnx.load_model("./edgeconv_test.onnx")
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, "./edgeconv_test.onnx")

    print("✅ Exported model to ./edgeconv_test.onnx")