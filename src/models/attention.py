"""
Additive Attention Gate Module (Oktay et al., 2018)
Filters irrelevant matrix background noise along horizontal skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """
    Additive Attention Gate.
    Calculates spatial attention coefficients alpha in [0, 1] conditioned on
    a coarse gating signal g from the decoder and skip features x from the encoder.
    """

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_proj = self.W_g(g)
        x_proj = self.W_x(x)

        # Upsample gating projection if spatial dimensions differ
        if g_proj.shape[2:] != x_proj.shape[2:]:
            g_proj = F.interpolate(g_proj, size=x_proj.shape[2:], mode="bilinear", align_corners=True)

        alpha = self.relu(g_proj + x_proj)
        alpha = self.psi(alpha)

        # Element-wise multiplication: filter non-salient activations
        return x * alpha
