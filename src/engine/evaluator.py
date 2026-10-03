"""
Full-Frame Sliding-Window Inference Module
Computes seamless probability maps across full-resolution micrographs using overlapping crops.
"""

import numpy as np
import torch
import torch.nn as nn

def predict_sliding_window(
    model: nn.Module,
    image: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    device: torch.device = torch.device("cpu")
) -> np.ndarray:
    """
    Runs sliding-window inference on a large micrograph.
    Accumulates overlapping predictions and averages them to suppress border stitching artifacts.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch segmentation model.
    image : np.ndarray
        Grayscale input image (H, W) in range [0, 255] or [0, 1].
    patch_size : int
        Window size (default 256).
    stride : int
        Step size (default 128 for 50% overlap).
    device : torch.device
        Inference device (CPU or CUDA).

    Returns
    -------
    np.ndarray
        Full-frame continuous probability map in range [0, 1].
    """
    model.eval()
    h, w = image.shape[:2]

    # Normalize to [0, 1] float32
    if image.max() > 1.0:
        norm_img = image.astype(np.float32) / 255.0
    else:
        norm_img = image.astype(np.float32)

    prob_accum = np.zeros((h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)

    # Pre-generate coordinates
    y_starts = list(range(0, h - patch_size + 1, stride))
    if (h - patch_size) % stride != 0:
        y_starts.append(h - patch_size)

    x_starts = list(range(0, w - patch_size + 1, stride))
    if (w - patch_size) % stride != 0:
        x_starts.append(w - patch_size)

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                patch = norm_img[y:y + patch_size, x:x + patch_size]
                t_patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

                logits = model(t_patch)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                patch_prob = torch.sigmoid(logits).squeeze().cpu().numpy()

                prob_accum[y:y + patch_size, x:x + patch_size] += patch_prob
                weight_accum[y:y + patch_size, x:x + patch_size] += 1.0

    # Avoid division by zero
    weight_accum[weight_accum == 0] = 1.0
    full_prob = prob_accum / weight_accum
    return np.clip(full_prob, 0.0, 1.0)
