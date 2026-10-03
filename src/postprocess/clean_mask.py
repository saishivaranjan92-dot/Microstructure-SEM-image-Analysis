"""
Morphological Cleaning and Small Artifact Filter
"""

import numpy as np
import cv2

def clean_binary_mask(
    prob_map: np.ndarray,
    threshold: float = 0.50,
    morph_size: int = 3,
    min_size_px: int = 3
) -> np.ndarray:
    """
    Cleans raw probability maps into high-quality binary masks:
      1. Binarizes probability map at fixed threshold (0.50).
      2. Applies morphological closing (bridges internal voids).
      3. Applies morphological opening (erodes single-pixel spikes).
      4. Discards connected regions smaller than min_size_px (camera read-out noise).

    Parameters
    ----------
    prob_map : np.ndarray
        Continuous probability heatmap in range [0, 1].
    threshold : float
        Decision threshold (default 0.50).
    morph_size : int
        Structuring element diameter (default 3 for 3x3 ellipse).
    min_size_px : int
        Minimum area in pixels to retain a particle (default 3 px).

    Returns
    -------
    np.ndarray
        Cleaned binary particle mask with values {0, 1} as uint8.
    """
    binary = (prob_map >= threshold).astype(np.uint8)

    # 3x3 elliptical structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))

    # Closing then opening
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Filter regions smaller than min_size_px
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cleaned = np.zeros_like(opened)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_size_px:
            cleaned[labels == label_id] = 1

    return cleaned
