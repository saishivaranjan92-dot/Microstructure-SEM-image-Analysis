"""
Model Training and Checkpoint Management Engine
"""

from typing import Dict, Any, Optional
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..losses.combined_loss import CombinedLoss
from ..losses.metrics import compute_segmentation_metrics

class MicrostructureTrainer:
    """
    Manages model training, validation, early stopping, and checkpoint export.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device = torch.device("cpu")
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        loss_cfg = config.get("loss", {})

        self.criterion = CombinedLoss(
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            aux_weight=loss_cfg.get("aux_weight", 0.3),
            smooth=loss_cfg.get("smooth", 1.0)
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4))
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=float(train_cfg.get("lr_factor", 0.5)),
            patience=int(train_cfg.get("lr_patience", 5)),
            verbose=True
        )

        self.early_stop_patience = int(train_cfg.get("early_stop_patience", 12))
        self.best_val_dice = 0.0
        self.epochs_without_improvement = 0

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0

        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            if isinstance(outputs, (tuple, list)):
                main_logits, aux_logits = outputs
                loss = self.criterion(main_logits, targets, aux_logits)
            else:
                loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / max(1, len(self.train_loader))

    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        dices, ious, precisions, recalls = [], [], [], []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                truths = targets.cpu().numpy().astype(int)

                for p, t in zip(preds, truths):
                    m = compute_segmentation_metrics(p[0], t[0])
                    dices.append(m["dice"])
                    ious.append(m["iou"])
                    precisions.append(m["precision"])
                    recalls.append(m["recall"])

        return {
            "val_dice": float(sum(dices) / max(1, len(dices))),
            "val_iou": float(sum(ious) / max(1, len(ious))),
            "val_precision": float(sum(precisions) / max(1, len(precisions))),
            "val_recall": float(sum(recalls) / max(1, len(recalls))),
        }

    def fit(self, num_epochs: int, save_dir: Path) -> Path:
        save_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = save_dir / "best_model.pth"

        print(f"Starting training for {num_epochs} epochs on {self.device}...")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.validate_epoch()
            val_dice = val_metrics["val_dice"]

            self.scheduler.step(val_dice)
            elapsed = time.time() - t0

            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"Val IoU: {val_metrics['val_iou']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpoint on best validation Dice
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.epochs_without_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_dice": val_dice,
                    "config": self.config
                }, best_ckpt_path)
                print(f"  -> Checkpoint saved to {best_ckpt_path} (Best Dice: {val_dice:.4f})")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stop_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        return best_ckpt_path
