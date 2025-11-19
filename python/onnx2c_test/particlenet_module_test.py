import torch
import torch.nn as nn
import numpy as np
import yaml
from weaver.nn.model.ParticleNet import ParticleNet


class ParticleNetWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        in_dim = kwargs['fc_params'][-1][0]
        num_classes = kwargs['num_classes']
        self.for_inference = kwargs['for_inference']
        fc_out_params = kwargs.pop('fc_out_params')

        # finetune the last FC layer
        layers = []
        layers = []
        for i in range(len(fc_out_params) - 1):
            in_channel, drop_rate = fc_out_params[i]
            out_channel, _ = fc_out_params[i + 1]
            if i == len(fc_out_params) - 2:
                layers += [nn.Linear(in_channel, out_channel)]
            else:
                layers += [nn.Linear(in_channel, out_channel),
                           nn.LeakyReLU(),
                           nn.Dropout(drop_rate, inplace=True)]
        
        self.fc_out = nn.Sequential(*layers)

        kwargs['for_inference'] = False
        self.mod = ParticleNet(**kwargs)
        self.mod.fc = self.mod.fc[:-1]

    def forward(self, points, features, mask):
        output = self.mod(points, features, mask)
        output = self.fc_out(output)
        return output


def get_model(data_config, **kwargs):
    # Define model configuration
    pf_features_dims = 4  # 4-momentum (px, py, pz, E)
    num_classes = 4
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
    ]
    fc_params = [(256, 0.1)]  # Fully connected layers with dropout
    fc_out_params = [(256, 0.0), (num_classes, 0.0)]  # Output layers

    # Initialize ParticleNet model
    model = ParticleNetWrapper(
        input_dims=pf_features_dims,
        num_classes=num_classes,
        fc_out_params=fc_out_params,
        conv_params=kwargs.get('conv_params', conv_params),
        fc_params=kwargs.get('fc_params', fc_params),
        use_fusion=kwargs.get('use_fusion', True),
        use_fts_bn=kwargs.get('use_fts_bn', True),
        use_counts=kwargs.get('use_counts', True),
        for_inference=kwargs.get('for_inference', False),
    )
    return model


def format_nested_array(arr, indent=4):
    """
    Recursively formats a numpy array as nested C initializer lists.
    """
    if arr.ndim == 1:
        return ", ".join((s if "." in (s := f"{x:.9g}") or "e" in s else s + ".0") + "f"for x in arr)

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
    pf_points = torch.randn(1, 3, 35)
    pf_features = torch.randn(1, 4, 35)
    pf_mask = torch.ones(1, 1, 35)

    tensor_to_c_multidim_header(pf_points, name="pf_points", header_path="../../C/include/pf_points.h")
    tensor_to_c_multidim_header(pf_features, name="pf_features", header_path="../../C/include/pf_features.h")
    tensor_to_c_multidim_header(pf_mask, name="pf_mask", header_path="../../C/include/pf_mask.h")

    with open('../pure_B_plus_B_minus/gen_level_1B/with_vertex/config/data_config_complete.yaml', 'r') as file:
        data_config = yaml.safe_load(file)

    particlenet = get_model(data_config)
    particlenet.load_state_dict(torch.load("../pure_B_plus_B_minus/gen_level_1B/with_vertex/training_log/particlenet_complete.pt"))
    particlenet.eval()

    output = particlenet(pf_points, pf_features, pf_mask)
    print(output)


