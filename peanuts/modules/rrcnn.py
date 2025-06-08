import torch.nn as nn

class RRCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            ),
        )

        self.rcnn = nn.Sequential(
            RCL(channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            RCL(channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.rcnn(x) + x


class RCL(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        t=2,
    ):
        super().__init__()
        self.t = t
        
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
        )
        
    def forward(self, x):
        for _ in range(self.t):
            x = self.conv(x) + x
        return x
