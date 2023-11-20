import torch
from torch import nn


class PRSNet(nn.Module):
    def __init__(self, amount_of_heads, out_features=4):
        super().__init__()

        if not (amount_of_heads > 0 and isinstance(amount_of_heads, int)):
            raise ValueError("Amount of heads should be a positive integer.")

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 4, 3, stride=1, padding=1),
            nn.MaxPool3d(2, stride=2),
            nn.LeakyReLU(),
            nn.Conv3d(4, 8, 3, stride=1, padding=1),
            nn.MaxPool3d(2, stride=2),
            nn.LeakyReLU(),
            nn.Conv3d(8, 16, 3, stride=1, padding=1),
            nn.MaxPool3d(2, stride=2),
            nn.LeakyReLU(),
            nn.Conv3d(16, 32, 3, stride=1, padding=1),
            nn.MaxPool3d(2, stride=2),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, 3, stride=1, padding=1),
            nn.MaxPool3d(2, stride=2),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.heads = nn.ModuleList([])
        for i in range(amount_of_heads):
            self.heads.append(nn.Sequential(
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
