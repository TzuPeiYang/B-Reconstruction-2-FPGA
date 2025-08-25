import torch
import torch.nn as nn
import torch.nn.functional as F
from weaver.nn.model.ParticleNet import ParticleNet


def scaled_dot_product_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale by the square root of the key dimension
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply mask if provided (useful for masked self-attention in transformers)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax to normalize scores, producing attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute the final output as weighted values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        # Define linear transformations for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        # Generate Q, K, V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention using our scaled dot-product function
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return out


class ParticleNetWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        in_dim = kwargs['fc_params'][-1][0]
        num_classes = kwargs['num_classes']
        self.for_inference = kwargs['for_inference']

        # finetune the last FC layer
        self.fc_out = nn.Sequential(nn.Linear(in_dim, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 64),
                                    nn.LeakyReLU(),
                                    nn.Linear(64, 16),
                                    nn.LeakyReLU(),
                                    nn.Linear(16, num_classes))

        kwargs['for_inference'] = False
        self.mod = ParticleNet(**kwargs)
        self.mod.fc = self.mod.fc[:-1]

    def forward(self, points, features, mask):
        output = self.mod(points, features, mask)
        output = self.fc_out(output)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output


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

    # Initialize ParticleNet model
    model = ParticleNetWrapper(
        input_dims=pf_features_dims,
        num_classes=num_classes,
        conv_params=kwargs.get('conv_params', conv_params),
        fc_params=kwargs.get('fc_params', fc_params),
        use_fusion=kwargs.get('use_fusion', False),
        use_fts_bn=kwargs.get('use_fts_bn', True),
        use_counts=kwargs.get('use_counts', True),
        for_inference=kwargs.get('for_inference', False),
        use_attention=False
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
    return torch.nn.MSELoss()