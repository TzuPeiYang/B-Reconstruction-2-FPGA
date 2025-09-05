import torch
import torch.nn as nn
import torch.nn.functional as F
from weaver.nn.model.ParticleNet import ParticleNet


class ParticleNetWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.in_dim = kwargs['fc_params'][-1][0]
        self.length = kwargs.pop('length')
        num_classes = kwargs['num_classes']
        self.for_inference = kwargs['for_inference']
        fc_out_params = kwargs.pop('fc_out_params')

        # finetune the last FC layer
        layers = []
        for i in range(len(fc_out_params) - 1):
            layers += [nn.Linear(fc_out_params[i], fc_out_params[i + 1]),
                       nn.LeakyReLU()]
        
        self.fc_out = nn.Sequential(*layers)

        kwargs['for_inference'] = False
        self.mod = ParticleNet(**kwargs)
        self.mod.fc = self.mod.fc[:-1]

    def forward(self, points, features, mask):
        output = self.mod(points, features, mask).view(-1, self.length, self.in_dim)
        output = self.fc_out(output).view(-1, 2, self.length)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output


def get_model(data_config, **kwargs):
    # Define model configuration
    pf_features_dims = len(data_config.input_dicts['pf_features'])  # 4-momentum (px, py, pz, E)
    length = 52
    num_classes = len(data_config.label_value)
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
    ]
    fc_params = [(128, 0.2)]  # Fully connected layers with dropout
    fc_out_params = [128, 64, 2]

    # Initialize ParticleNet model
    model = ParticleNetWrapper(
        input_dims=pf_features_dims,
        num_classes=num_classes,
        fc_out_params=fc_out_params,
        length=length,
        conv_params=kwargs.get('conv_params', conv_params),
        fc_params=kwargs.get('fc_params', fc_params),
        use_fusion=kwargs.get('use_fusion', True),
        use_fts_bn=kwargs.get('use_fts_bn', True),
        use_counts=kwargs.get('use_counts', True),
        for_inference=kwargs.get('for_inference', False),
        use_attention=kwargs.get('use_attention', False),
        for_segmentation=kwargs.get('for_segmentation', True),
    )

    # Define loss function and optimizer
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['MSE'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'MSE': {0: 'N'}}},
    }
    return model, model_info


    
def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()