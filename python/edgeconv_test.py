import torch
import torch.nn as nn
import torch.nn.functional as F
from weaver.nn.model.ParticleNet import ParticleNet, EdgeConvBlock


if __name__ == "__main__":
    pf_points = torch.arange(8 * 3 * 10, dtype=torch.float32).reshape(8, 3, 10)
    pf_features = torch.arange(8 * 4 * 10, dtype=torch.float32).reshape(8, 4, 10)

    edge_conv = EdgeConvBlock(3, 4, [8], cpu_mode=True)
    edge_conv.load_state_dict(torch.load("edge_conv_test.pt"))
    # torch.save(edge_conv.state_dict(), "edge_conv_test.pt")
    edge_conv(pf_points, pf_features)
    # print(edge_conv(pf_points, pf_features))