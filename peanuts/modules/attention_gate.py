import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, out_channels, hidden_channels):
        super().__init__()
        self.weight_x = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
        )

        self.weight_g = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
        ),

        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=1,
                kernel_size=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid(),
        )


    def forward(self, x, g):
        h = self.weight_x(x)
        g = self.weight_g(g)
        g = nn.functional.interpolate(g, size=h.shape[2:])  # resize with [height, width]
        
        alpha = self.layers(g + h)
        return alpha * x
