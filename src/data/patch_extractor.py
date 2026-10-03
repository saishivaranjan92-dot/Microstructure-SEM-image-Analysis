"""
Patch Extraction Module
Extracts uniform patches from paired SEM micrographs and binary masks.
"""

from typing import List, Tuple
import numpy as np

def extract_micrograph_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
    min_particle_fraction: float = 0.005
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts square patches from an image-mask pair using a sliding grid.
    Filters out near-empty patches containing less than min_particle_fraction foreground.

    Parameters
    ----------
    image : np.ndarray
        Grayscale micrograph (H, W).
    mask : np.ndarray
        Binary reference mask (H, W) with values {0, 1} or {0, 255}.
    patch_size : int
        Width and height of square crop (default 256).
    stride : int
        Step size between adjacent crops (default 256 for non-overlapping).
    min_particle_fraction : float
        Minimum fraction of particle pixels required to keep patch (default 0.005 = 0.5%).

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (image_patch, mask_patch) arrays.
    """
    h, w = image.shape[:2]
    binary_mask = (mask > 0).astype(np.uint8)
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_crop = image[y:y + patch_size, x:x + patch_size]
            mask_crop = binary_mask[y:y + patch_size, x:x + patch_size]

            particle_frac = float(mask_crop.sum()) / float(mask_crop.size)
            if particle_frac >= min_particle_fraction:
                patches.append((img_crop.copy(), mask_crop.copy()))

    return patches
