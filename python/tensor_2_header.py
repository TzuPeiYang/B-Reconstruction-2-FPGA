import torch
import numpy as np


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
