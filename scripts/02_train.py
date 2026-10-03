"""
Step 2: Training Script for Residual Attention U-Net
Trains model using AdamW, ReduceLROnPlateau, and Combined Loss with Deep Supervision.
"""

import argparse
from pathlib import Path
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import MicrographPatchDataset
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.models.res_attention_unet import ResidualAttentionUNet
from src.engine.trainer import MicrostructureTrainer

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train Residual Attention U-Net on patches.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("training", {}).get("seed", 42))

    data_cfg = cfg.get("data", {})
    patches_dir = Path(data_cfg.get("patches_dir", "data/patches"))
    img_files = sorted(list(patches_dir.glob("*_img.npy")))

    if not img_files:
        print(f"No patch files found in {patches_dir}. Please run 01_prepare_data.py first.")
        return

    # Load all patches into memory
    patch_pairs = []
    for f in img_files:
        m_file = patches_dir / f.name.replace("_img.npy", "_msk.npy")
        if m_file.exists():
            patch_pairs.append((np.load(f), np.load(m_file)))

    print(f"Loaded {len(patch_pairs)} patch pairs.")

    val_split = float(data_cfg.get("val_split", 0.20))
    train_pairs, val_pairs = train_test_split(
        patch_pairs, test_size=val_split, random_state=42
    )
    print(f"Split: {len(train_pairs)} training patches, {len(val_pairs)} validation patches.")

    train_ds = MicrographPatchDataset(train_pairs, transform=get_training_transforms())
    val_ds = MicrographPatchDataset(val_pairs, transform=get_validation_transforms())

    train_cfg = cfg.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model_cfg = cfg.get("model", {})
    model = ResidualAttentionUNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        encoder_channels=model_cfg.get("encoder_channels", [16, 32, 64, 128]),
        bottleneck_channels=int(model_cfg.get("bottleneck_channels", 256)),
        dropout_bottleneck=float(model_cfg.get("dropout_bottleneck", 0.3)),
        dropout_decoder=float(model_cfg.get("dropout_decoder", 0.1))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MicrostructureTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device
    )

    epochs = int(train_cfg.get("epochs", 80))
    save_dir = Path("checkpoints")
    best_path = trainer.fit(num_epochs=epochs, save_dir=save_dir)
    print(f"Training complete. Best model saved at: {best_path}")

if __name__ == "__main__":
    main()
