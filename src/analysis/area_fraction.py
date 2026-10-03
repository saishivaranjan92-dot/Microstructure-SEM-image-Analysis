"""
Delesse Principle Area Fraction (A_A) Calculation
Computes particle area fraction representing 3D volume fraction (V_V).
"""

from typing import List, Dict, Any
import numpy as np

def compute_area_fraction(
    binary_mask: np.ndarray,
    banner_height: int = 64
) -> float:
    """
    Computes total particle area fraction across the active micrograph region.
    Note: Boundary-intersecting particles are strictly INCLUDED in total area fraction
    so that overall phase volume fraction is not underestimated.

    Parameters
    ----------
    binary_mask : np.ndarray
        Cleaned binary mask (H, W).
    banner_height : int
        Height of the microscope data banner (excluded from active analysis area).

    Returns
    -------
    float
        Area fraction percentage (A_A %) in range [0, 100].
    """
    h, w = binary_mask.shape[:2]
    active_h = h - banner_height if banner_height > 0 else h
    active_area_px = active_h * w

    # Foreground pixels in active region
    active_mask = binary_mask[:active_h, :]
    particle_pixels = np.sum(active_mask > 0)

    area_fraction_pct = (float(particle_pixels) / float(active_area_px)) * 100.0
    return float(area_fraction_pct)
