"""
Area-Weighted Particle Size Distribution (PSD) Module
Computes metallurgically meaningful area-weighted histograms using uniform 25 nm bins.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

def compute_area_weighted_psd(
    particles: List[Dict[str, Any]],
    bin_width_nm: float = 25.0,
    exclude_borders: bool = True,
    max_diameter_nm: float = None
) -> pd.DataFrame:
    """
    Constructs an area-weighted particle size distribution:
      - Bins particles by equivalent circular diameter (d_eq).
      - Primary parameter Y_k = sum(Area_i) in um^2 for particles in bin k.
      - Secondary parameter AF_k (%) = (Y_k / Total_Particle_Area) * 100.

    Parameters
    ----------
    particles : List[Dict[str, Any]]
        List of particle property records.
    bin_width_nm : float
        Uniform bin width in nanometers (default 25.0 nm).
    exclude_borders : bool
        If True, excludes particles touching the micrograph edge from sizing.
    max_diameter_nm : float, optional
        Upper bound for binning. If None, calculated from max observed particle.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
          - bin_start_nm, bin_center_nm, bin_end_nm
          - particle_count (frequency)
          - area_parameter_um2 (Y_k)
          - area_fraction_pct (AF_k %)
    """
    # Filter particles
    valid_particles = [
        p for p in particles
        if not (exclude_borders and p.get("touches_border", False))
    ]

    if not valid_particles:
        return pd.DataFrame()

    diameters = np.array([p["d_eq_nm"] for p in valid_particles])
    areas_um2 = np.array([p["area_um2"] for p in valid_particles])

    max_d = max_diameter_nm if max_diameter_nm is not None else float(np.max(diameters))
    bins = np.arange(0.0, max_d + bin_width_nm, bin_width_nm)

    bin_indices = np.digitize(diameters, bins) - 1
    total_area_um2 = np.sum(areas_um2)

    records = []
    for k in range(len(bins) - 1):
        b_start = bins[k]
        b_end = bins[k + 1]
        b_center = (b_start + b_end) / 2.0

        mask_k = (bin_indices == k)
        count_k = int(np.sum(mask_k))
        area_sum_k = float(np.sum(areas_um2[mask_k])) if count_k > 0 else 0.0

        af_pct_k = (area_sum_k / total_area_um2 * 100.0) if total_area_um2 > 0 else 0.0

        records.append({
            "bin_start_nm": b_start,
            "bin_center_nm": b_center,
            "bin_end_nm": b_end,
            "particle_count": count_k,
            "area_parameter_um2": area_sum_k,
            "area_fraction_pct": af_pct_k
        })

    return pd.DataFrame(records)
