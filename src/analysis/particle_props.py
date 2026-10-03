"""
Quantitative Particle Sizing and Morphological Extraction
Measures equivalent circular diameter (d_eq), ellipse aspect ratio, and boundary contact.
"""

from typing import List, Dict, Any
import numpy as np
from skimage.measure import regionprops

def measure_particle_properties(
    labeled_mask: np.ndarray,
    nm_per_pixel: float,
    min_ellipse_size_px: int = 10
) -> List[Dict[str, Any]]:
    """
    Measures morphological attributes for each labeled precipitate particle.

    Parameters
    ----------
    labeled_mask : np.ndarray
        2D labeled image (int32) where 0 is matrix and 1..N are particles.
    nm_per_pixel : float
        Nanometers per pixel scale factor from microscope calibration.
    min_ellipse_size_px : int
        Minimum area in pixels to reliably fit second-moment inertia ellipse (default 10 px).

    Returns
    -------
    List[Dict[str, Any]]
        List of particle records containing area, d_eq (nm), aspect ratio, and touches_border flag.
    """
    h, w = labeled_mask.shape
    props = regionprops(labeled_mask)
    particles = []

    for prop in props:
        area_px = prop.area
        # Scale to physical units
        area_um2 = area_px * ((nm_per_pixel / 1000.0) ** 2)
        d_eq_nm = 2.0 * np.sqrt(area_px / np.pi) * nm_per_pixel

        # Aspect ratio: minor_axis / major_axis of equivalent inertia ellipse (in (0, 1])
        if area_px >= min_ellipse_size_px and prop.major_axis_length > 0:
            aspect_ratio = float(prop.minor_axis_length / prop.major_axis_length)
            aspect_ratio = min(1.0, max(0.01, aspect_ratio))
        else:
            aspect_ratio = np.nan

        # Check if particle intersects the image boundary
        minr, minc, maxr, maxc = prop.bbox
        touches_border = (minr <= 0 or minc <= 0 or maxr >= h or maxc >= w)

        particles.append({
            "label_id": prop.label,
            "area_px": area_px,
            "area_um2": float(area_um2),
            "d_eq_nm": float(d_eq_nm),
            "aspect_ratio": aspect_ratio,
            "touches_border": bool(touches_border),
            "centroid": (float(prop.centroid[0]), float(prop.centroid[1]))
        })

    return particles
