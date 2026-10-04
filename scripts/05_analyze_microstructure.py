"""
Measure particle sizes, calculate area fraction, and plot area-weighted PSD.
"""

import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.analysis.particle_props import measure_particle_properties
from src.analysis.area_fraction import compute_area_fraction
from src.analysis.area_weighted_psd import compute_area_weighted_psd
from src.analysis.gmm_fit import fit_two_component_gmm

def main():
    parser = argparse.ArgumentParser(description="Analyze microstructure and compute area-weighted PSD.")
    parser.add_argument("--instances", type=str, required=True, help="Path to watershed instance image or directory")
    parser.add_argument("--mag", type=str, default="3000x", help="Microscope magnification (e.g. 2000x, 3000x, 20000x)")
    parser.add_argument("--calibration", type=str, default="configs/calibration.yaml", help="Path to calibration.yaml")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/psd_results", help="Output directory")
    args = parser.parse_args()

    # Load calibration metadata
    with open(args.calibration, "r", encoding="utf-8") as f:
        calib_cfg = yaml.safe_load(f)

    mag_data = calib_cfg.get("magnifications", {}).get(args.mag, {})
    if not mag_data:
        print(f"Magnification {args.mag} not found in calibration.yaml. Defaulting to 33.07 nm/px (3000x).")
        nm_per_px = 33.07
    else:
        nm_per_px = float(mag_data.get("nm_per_pixel", 33.07))

    print(f"Using scale calibration for {args.mag}: {nm_per_px:.2f} nm/pixel")

    with open(args.config, "r", encoding="utf-8") as f:
        pipe_cfg = yaml.safe_load(f)

    bin_width = float(pipe_cfg.get("analysis", {}).get("psd_bin_width_nm", 25.0))
    exclude_borders = bool(pipe_cfg.get("analysis", {}).get("exclude_boundary_particles", True))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.instances)
    if input_path.is_file():
        instance_files = [input_path]
    else:
        instance_files = sorted(list(input_path.glob("*_instances.png")) + list(input_path.glob("*.png")))

    all_particles = []
    total_area_fractions = []

    for inst_p in instance_files:
        inst_img = cv2.imread(str(inst_p), cv2.IMREAD_UNCHANGED)
        if inst_img is None:
            continue

        # Area fraction on entire active image (including borders)
        af = compute_area_fraction((inst_img > 0).astype(np.uint8), banner_height=64)
        total_area_fractions.append(af)

        # Measure particles
        particles = measure_particle_properties(
            labeled_mask=inst_img,
            nm_per_pixel=nm_per_px,
            min_ellipse_size_px=10
        )
        all_particles.extend(particles)

    if not all_particles:
        print("No particles detected for analysis.")
        return

    # Compute PSD
    psd_df = compute_area_weighted_psd(
        particles=all_particles,
        bin_width_nm=bin_width,
        exclude_borders=exclude_borders
    )

    csv_path = out_dir / f"psd_{args.mag}_area_weighted.csv"
    psd_df.to_csv(csv_path, index=False)
    print(f"Saved PSD table to {csv_path}")

    # Fit GMM
    valid_diams = np.array([p["d_eq_nm"] for p in all_particles if not (exclude_borders and p["touches_border"])])
    valid_areas = np.array([p["area_um2"] for p in all_particles if not (exclude_borders and p["touches_border"])])
    gmm_results = fit_two_component_gmm(valid_diams, weights=valid_areas)

    # Plot Dual-Axis Figure
    fig, ax1 = plt.subplots(figsize=(7.5, 5.0), dpi=300)

    bin_centers = psd_df["bin_center_nm"].values
    area_params = psd_df["area_parameter_um2"].values
    af_pcts = psd_df["area_fraction_pct"].values

    # Primary axis: Area Parameter Y_k (um^2)
    bar_width = bin_width * 0.85
    bars = ax1.bar(bin_centers, area_params, width=bar_width, color="#3B82F6", edgecolor="#1D4ED8", alpha=0.75, label="Area Parameter $Y_k$")
    ax1.set_xlabel("Equivalent Circular Diameter $d_{\\mathrm{eq}}$ (nm)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Area Parameter $Y_k = \\sum A_i$ ($\\mu\\mathrm{m}^2$)", fontsize=11, fontweight="bold", color="#1D4ED8")
    ax1.tick_params(axis="y", labelcolor="#1D4ED8")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Secondary axis: Area Fraction (%)
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, af_pcts, color="#DC2626", lw=2.0, marker="o", markersize=4, label="Normalized $AF_k$ (%)")
    ax2.set_ylabel("Area Fraction $AF_k$ (%)", fontsize=11, fontweight="bold", color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")

    mean_af = np.mean(total_area_fractions) if total_area_fractions else 0.0
    title_str = f"Area-Weighted PSD ({args.mag}) | Mean Area Fraction: {mean_af:.2f}%\n"
    if gmm_results:
        title_str += f"GMM Modes: $\\mu_1 = {gmm_results['mu1_nm']:.1f}$ nm, $\\mu_2 = {gmm_results['mu2_nm']:.1f}$ nm"

    plt.title(title_str, fontsize=10.5, fontweight="bold", pad=12)
    plt.tight_layout()

    plot_path = out_dir / f"psd_{args.mag}_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated PSD plot: {plot_path}")

if __name__ == "__main__":
    main()
