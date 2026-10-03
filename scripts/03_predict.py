"""
Step 3: Sliding-Window Inference Script
Executes full-frame particle segmentation with 50% window overlap.
"""

import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
import torch
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.res_attention_unet import ResidualAttentionUNet
from src.engine.evaluator import predict_sliding_window
from src.data.inpaint import inpaint_data_banner

def main():
    parser = argparse.ArgumentParser(description="Run sliding-window prediction on micrographs.")
    parser.add_argument("--input", type=str, required=True, help="Path to micrograph image or directory")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pth", help="Path to model weights")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_cfg = cfg.get("model", {})
    model = ResidualAttentionUNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        encoder_channels=model_cfg.get("encoder_channels", [16, 32, 64, 128]),
        bottleneck_channels=int(model_cfg.get("bottleneck_channels", 256)),
    )

    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()

    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.png")))

    out_prob_dir = Path(args.output_dir) / "probability_maps"
    out_mask_dir = Path(args.output_dir) / "binary_masks"
    out_prob_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    inf_cfg = cfg.get("inference", {})
    patch_size = int(inf_cfg.get("patch_size", 256))
    stride = int(inf_cfg.get("stride", 128))
    threshold = float(inf_cfg.get("threshold", 0.50))

    print(f"Running inference on {len(image_files)} image(s)...")

    for img_p in image_files:
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Inpaint banner
        clean_img = inpaint_data_banner(img, banner_height=64)

        # Sliding window prediction
        prob_map = predict_sliding_window(
            model=model,
            image=clean_img,
            patch_size=patch_size,
            stride=stride,
            device=device
        )

        binary_mask = (prob_map >= threshold).astype(np.uint8) * 255

        # Save continuous probability map and binary mask
        prob_uint8 = (prob_map * 255.0).astype(np.uint8)
        cv2.imwrite(str(out_prob_dir / f"{img_p.stem}_prob.png"), prob_uint8)
        cv2.imwrite(str(out_mask_dir / f"{img_p.stem}_mask.png"), binary_mask)
        print(f"  Processed {img_p.name} -> Mask area fraction: {np.mean(binary_mask > 0)*100:.2f}%")

    print(f"Inference complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
