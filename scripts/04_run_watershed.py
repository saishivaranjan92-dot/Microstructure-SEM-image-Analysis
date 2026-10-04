"""
Clean binary masks and split touching particles using watershed.
"""

import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.postprocess.clean_mask import clean_binary_mask
from src.postprocess.watershed import separate_touching_particles

def main():
    parser = argparse.ArgumentParser(description="Separate touching particles using marker-controlled watershed.")
    parser.add_argument("--input", type=str, required=True, help="Path to binary mask or directory of masks")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/watershed_instances", help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    post_cfg = cfg.get("postprocess", {})
    morph_size = int(post_cfg.get("morph_element_size", 3))
    min_size = int(post_cfg.get("min_particle_size_px", 3))
    sigma = float(post_cfg.get("gaussian_sigma", 0.8))
    min_dist = int(post_cfg.get("peak_min_distance", 4))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_file():
        mask_files = [input_path]
    else:
        mask_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.tif")))

    print(f"Running watershed on {len(mask_files)} mask(s)...")

    for mf in mask_files:
        raw_mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            continue

        # Morphological clean
        cleaned = clean_binary_mask(
            prob_map=raw_mask.astype(np.float32) / 255.0,
            threshold=0.50,
            morph_size=morph_size,
            min_size_px=min_size
        )

        # Watershed separation
        labeled_instances, count = separate_touching_particles(
            binary_mask=cleaned,
            gaussian_sigma=sigma,
            min_peak_distance=min_dist
        )

        # Save labeled instances as 16-bit PNG (to support > 255 particles)
        out_path = out_dir / f"{mf.stem}_instances.png"
        cv2.imwrite(str(out_path), labeled_instances.astype(np.uint16))
        print(f"  Processed {mf.name} -> Separated into {count} individual particles.")

    print(f"Watershed complete. Instance maps saved to {out_dir}")

if __name__ == "__main__":
    main()
