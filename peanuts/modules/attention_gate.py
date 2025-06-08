import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, out_channels, hidden_channels):
        super().__init__()
        self.weight_x = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        self.weight_g = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Sigmoid(),
        )


    def forward(self, x, g):
        g = self.weight_g(g)
        g = nn.functional.interpolate(g, size=x.shape[2:])  # upsampling with [height, width]
        h = self.weight_x(x)
        
        alpha = self.layers(g + h)
        return alpha * x
