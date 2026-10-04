"""
Extract 256x256 training patches from SEM images and masks.
Removes the footer info strip with inpainting before saving patches.
"""

import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.inpaint import inpaint_data_banner
from src.data.patch_extractor import extract_micrograph_patches

def main():
    parser = argparse.ArgumentParser(description="Prepare patches from SEM micrographs and masks.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    masks_dir = Path(data_cfg.get("masks_dir", "data/masks"))
    out_dir = Path(data_cfg.get("patches_dir", "data/patches"))
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_size = int(data_cfg.get("patch_size", 256))
    stride = int(data_cfg.get("patch_stride", 256))
    min_frac = float(data_cfg.get("min_particle_fraction", 0.005))

    image_files = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")) + list(raw_dir.glob("*.png")))
    print(f"Found {len(image_files)} micrographs in {raw_dir}")

    total_patches = 0
    for img_path in image_files:
        # Match mask filename
        mask_path = masks_dir / f"{img_path.stem}_mask.tif"
        if not mask_path.exists():
            mask_path = masks_dir / f"{img_path.stem}_Thresholded.tif"
        if not mask_path.exists():
            mask_path = masks_dir / img_path.name

        if not mask_path.exists():
            print(f"  [Skip] No matching mask found for {img_path.name}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        # Inpaint banner to prevent edge noise
        img_clean = inpaint_data_banner(img, banner_height=64)
        mask_binary = (mask > 127).astype(np.uint8)

        # Extract patches
        patches = extract_micrograph_patches(
            img_clean, mask_binary,
            patch_size=patch_size, stride=stride,
            min_particle_fraction=min_frac
        )

        for p_idx, (p_img, p_msk) in enumerate(patches):
            out_stem = f"{img_path.stem}_p{p_idx:03d}"
            np.save(out_dir / f"{out_stem}_img.npy", p_img)
            np.save(out_dir / f"{out_stem}_msk.npy", p_msk)
            total_patches += 1

    print(f"Successfully generated {total_patches} patches in {out_dir}")

if __name__ == "__main__":
    main()
