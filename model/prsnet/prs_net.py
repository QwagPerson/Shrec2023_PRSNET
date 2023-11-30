import math

import torch
from torch import nn


class PRSNet(nn.Module):
    def __init__(self, input_resolution, amount_of_heads, out_features=4, use_bn=True):
        super().__init__()
        if not (amount_of_heads > 0 and isinstance(amount_of_heads, int)):
            raise ValueError("Amount of heads should be a positive integer.")

        conv_layers = []
        n_conv_layers = int(math.log2(input_resolution))
        in_channels = 1
        out_channels = 4

        for i in range(n_conv_layers):
            conv_layer = [
                nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.MaxPool3d(2, stride=2),
                nn.LeakyReLU(),
            ]

            if use_bn and i != n_conv_layers - 1:
                conv_layer.append(nn.BatchNorm3d(out_channels))

            conv_layers += conv_layer
            in_channels = out_channels
            out_channels = out_channels * 2

        self.encoder = nn.Sequential(*conv_layers, nn.Flatten())

        self.heads = nn.ModuleList([])
        for i in range(amount_of_heads):
            if use_bn:
                self.heads.append(nn.Sequential(
                    nn.Linear(in_channels, 64),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(64),
                    nn.Linear(64, 32),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 16),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(16),
                    nn.Linear(16, out_features),
                ))
            else:
                self.heads.append(nn.Sequential(
                    nn.Linear(in_channels, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 16),
                    nn.LeakyReLU(),
                    nn.Linear(16, out_features),
                ))


    def forward(self, x):
        x = self.encoder(x)

        results = []
        for head in self.heads:
            results.append(head(x))
        return torch.stack(results, dim=1)
