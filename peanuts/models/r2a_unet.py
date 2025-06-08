import torch
import torch.nn as nn
from torchvision.transforms.v2 import CenterCrop

from modules import Conv, ConvTranspose, RCL, AttentionGate

class R2AUNet(nn.Module):
    def __init__(self, kernel_size=(7, 1), stride=(4, 1)):
        super().__init__()
        self.input = Conv(3, 8, kernel_size)

        self.encoder = nn.ModuleList([
            DownConv(8, 8, kernel_size, stride),
            DownConv(8, 11, kernel_size, stride),
            DownConv(11, 16, kernel_size, stride),
            DownConv(16, 22, kernel_size, stride),
        ])

        self.decoder = nn.ModuleList([
            UpConv(22, 32, 22, kernel_size, stride),
            UpConv(44, 22, 16, kernel_size, stride),
            UpConv(32, 16, 11, kernel_size, stride),
            UpConv(22, 11, 8, kernel_size, stride),
        ])

        self.output = nn.Sequential(
            Conv(16, 8, kernel_size),
            nn.Conv2d(8, 3, kernel_size=1, padding="same"),
        )

    def forward(self, x):
        x = self.input(x)

        residuals = []
        for down_conv in self.encoder:
            x, residual = down_conv(x)
            residuals.append(residual)

        for up_conv in self.decoder:
            residual = residuals.pop()
            x = up_conv(x, residual)
            
        return self.output(x)
    

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.rcl = nn.Sequential(
            RCL(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            RCL(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )
        
        self.down_conv = Conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.residual = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
        )

    def forward(self, x):
        residual = self.rcl(x) + self.residual(x)
        x = self.down_conv(residual)
        return x, residual


class UpConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        stride,
    ):
        super().__init__()
        self.rcl = nn.Sequential(
            RCL(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
            ),
            RCL(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
            ),
        )
        
        self.up_conv = ConvTranspose(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.attention_gate = AttentionGate(
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )

    def forward(self, x, residual):
        x = self.rcl(x)
        residual = self.attention_gate(residual, x)

        x = self.up_conv(x)
        x = CenterCrop(residual.shape[2:])(x)  #  [height, width]
        x = torch.concat([residual, x], dim=1)  # channel-wise
        return x
