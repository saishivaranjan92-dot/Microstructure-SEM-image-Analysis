"""
Residual Attention U-Net Architecture with Deep Supervision
Customized for SEM Microstructural Particle Segmentation
"""

from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResConvBlock
from .attention import AttentionGate

class ResidualAttentionUNet(nn.Module):
    """
    Residual Attention U-Net with deep supervision auxiliary outputs.
    Channels: Encoder [16, 32, 64, 128], Bottleneck 256, Decoder [128, 64, 32, 16].
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: List[int] = None,
        bottleneck_channels: int = 256,
        dropout_bottleneck: float = 0.3,
        dropout_decoder: float = 0.1
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [16, 32, 64, 128]

        self.encoder_channels = encoder_channels
        self.decoder_channels = list(reversed(encoder_channels))

        # --- Contracting Encoder ---
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        curr_ch = in_channels
        for ch in self.encoder_channels:
            self.encoders.append(ResConvBlock(curr_ch, ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_ch = ch

        # --- Bottleneck ---
        self.bottleneck = ResConvBlock(curr_ch, bottleneck_channels, dropout=dropout_bottleneck)

        # --- Expanding Decoder ---
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()

        curr_ch = bottleneck_channels
        for ch in self.decoder_channels:
            self.upconvs.append(nn.ConvTranspose2d(curr_ch, ch, kernel_size=2, stride=2))
            self.attention_gates.append(AttentionGate(g_channels=ch, x_channels=ch, inter_channels=max(1, ch // 2)))
            self.decoders.append(ResConvBlock(ch * 2, ch, dropout=dropout_decoder))
            curr_ch = ch

        # --- Main 1x1 Convolution Head ---
        self.out_conv = nn.Conv2d(self.encoder_channels[0], out_channels, kernel_size=1)

        # --- Deep Supervision Heads (tapped from intermediate decoder stages) ---
        # Level 1 (64 channels) and Level 2 (32 channels)
        self.aux_head_1 = nn.Conv2d(self.decoder_channels[1], out_channels, kernel_size=1)
        self.aux_head_2 = nn.Conv2d(self.decoder_channels[2], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # Contracting path
        skips = []
        out = x
        for enc, pool in zip(self.encoders, self.pools):
            out = enc(out)
            skips.append(out)
            out = pool(out)

        # Bottleneck bridge
        out = self.bottleneck(out)

        # Expanding path with attention-gated skip connections
        skips = list(reversed(skips))
        aux_outputs = []

        for idx, (up, att, dec) in enumerate(zip(self.upconvs, self.attention_gates, self.decoders)):
            out = up(out)

            # Align spatial shape if needed
            if out.shape[2:] != skips[idx].shape[2:]:
                out = F.interpolate(out, size=skips[idx].shape[2:], mode="bilinear", align_corners=True)

            attended_skip = att(out, skips[idx])
            out = torch.cat([out, attended_skip], dim=1)
            out = dec(out)

            # Deep supervision output taps
            if idx == 1:
                aux_outputs.append(self.aux_head_1(out))
            elif idx == 2:
                aux_outputs.append(self.aux_head_2(out))

        main_logits = self.out_conv(out)

        if self.training and aux_outputs:
            return main_logits, aux_outputs
        return main_logits
