import torch
import torch.nn as nn

class RCL(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t=2,
    ):
        super().__init__()
        self.t = t

        self.weight_x = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

        self.weight_h = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.weight_x(x)
        h = torch.zeros_like(x)

        for _ in range(self.t):
            h = x + self.weight_h(h)

        return self.relu(h)
