import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        # print(outputs.size(), targets.size())
        loss = self.mse(outputs, targets)
        # print(loss)
        delta_mass = torch.mean(outputs[:, 0] ** 2 - targets[:, 0] ** 2)
        for i in range(1, 4):
            delta_mass += -torch.mean(outputs[:, i] ** 2 - targets[:, i] ** 2)
        return loss + torch.abs(delta_mass)


def get_model(data_config, **kwargs):
    # Define model configuration
    pf_features_dims = len(data_config.input_dicts['pf_features'])  # 4-momentum (px, py, pz, E)
    num_classes = len(data_config.label_value) 
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
    ]
    fc_params = [(256, 0.1)]  # Fully connected layers with dropout
    fc_out_params = [(256, 0.0), (128, 0.0), (64, 0.0), (16, 0.0), (num_classes, 0)]

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
        use_attention=kwargs.get('use_attention', False),
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
    return CustomLoss()