"""
Combined Loss with Deep Supervision
Balances pixel-wise Cross Entropy, region-based Dice loss, and intermediate auxiliary taps.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_loss import DiceLoss

class CombinedLoss(nn.Module):
    """
    Combined Loss: alpha * BCE + beta * Dice + gamma * sum(AuxLosses).
    Default weighting: 0.5 BCE + 0.5 Dice + 0.3 * Aux.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        aux_weight: float = 0.3,
        smooth: float = 1.0
    ):
        super().__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss(smooth=smooth)
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.aux_w = aux_weight

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits_list: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        loss_main = self.bce_w * self.bce_fn(main_logits, targets) + self.dice_w * self.dice_fn(main_logits, targets)

        if not aux_logits_list:
            return loss_main

        loss_aux = 0.0
        target_size = targets.shape[2:]
        for aux_logits in aux_logits_list:
            # Interpolate auxiliary outputs to match full patch size
            if aux_logits.shape[2:] != target_size:
                aux_logits = F.interpolate(aux_logits, size=target_size, mode="bilinear", align_corners=True)
            loss_aux += self.bce_w * self.bce_fn(aux_logits, targets) + self.dice_w * self.dice_fn(aux_logits, targets)

        return loss_main + self.aux_w * loss_aux
