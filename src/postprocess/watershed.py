"""
Marker-Controlled Watershed Module for Touching Particles
Splits clustered semantic regions into distinct individual instances.
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label

def separate_touching_particles(
    binary_mask: np.ndarray,
    gaussian_sigma: float = 0.8,
    min_peak_distance: int = 4
) -> Tuple[np.ndarray, int]:
    """
    Separates touching precipitates using marker-controlled watershed segmentation:
      1. Computes Euclidean Distance Transform (EDT) from background.
      2. Smooths distance transform using Gaussian filter (sigma = 0.8).
      3. Identifies local geometric peaks (min_distance = 4 px) as seeds.
      4. If a region has no peak, assigns centroid as single marker.
      5. Floods the inverted distance landscape constrained to foreground mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        Cleaned binary particle mask with values {0, 1}.
    gaussian_sigma : float
        Standard deviation for Gaussian smoothing (default 0.8 px).
    min_peak_distance : int
        Minimum distance between detected seed peaks (default 4 px).

    Returns
    -------
    Tuple[np.ndarray, int]
        Labeled instance image (int32, background=0, particles=1..N),
        and the total count of separated individual particles.
    """
    foreground = (binary_mask > 0).astype(bool)
    if not np.any(foreground):
        return np.zeros_like(binary_mask, dtype=np.int32), 0

    # 1. Euclidean distance transform to nearest background
    dist_map = distance_transform_edt(foreground)

    # 2. Gaussian smoothing to avoid over-segmentation on rugged particle contours
    smoothed_dist = gaussian_filter(dist_map, sigma=gaussian_sigma)

    # 3. Peak local maxima as seed markers
    peaks = peak_local_max(
        smoothed_dist,
        min_distance=min_peak_distance,
        labels=foreground,
        exclude_border=False
    )

    markers = np.zeros_like(binary_mask, dtype=np.int32)
    if len(peaks) > 0:
        for idx, (r, c) in enumerate(peaks, start=1):
            markers[r, c] = idx
    else:
        # Fallback to connected component labeling if no peaks found
        markers = label(foreground).astype(np.int32)

    # Ensure every disconnected foreground island has at least one seed
    cc_labels = label(foreground)
    for cc_id in range(1, cc_labels.max() + 1):
        island_mask = (cc_labels == cc_id)
        if np.max(markers[island_mask]) == 0:
            coords = np.argwhere(island_mask)
            center = coords[len(coords) // 2]
            markers[center[0], center[1]] = markers.max() + 1

    # 4. Invert distance map and execute watershed flooded from seeds
    labeled_instances = watershed(-smoothed_dist, markers, mask=foreground)
    total_particles = int(labeled_instances.max())

    return labeled_instances, total_particles
