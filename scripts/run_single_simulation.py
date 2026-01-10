#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Simulation Runner

This script runs a single OpenSees-Py nonlinear pushover simulation
with sampled parameters and prints the capacity point.
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from BridgeModeling.Pushover import run_single_pushover_simulation
from src.postprocessing.bilinear_fit import fit_bilinear_profile


def main():
    """Main function to run single simulation."""
    parser = argparse.ArgumentParser(
        description="Run a single OpenSees-Py pushover simulation with sampled parameters"
    )
    parser.add_argument(
        "--scenario",
        choices=["missouri", "colorado", "extreme"],
        default="missouri",
        help="Climate scenario (default: missouri)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling"
    )

    args = parser.parse_args()

    print(f"🏃 Running single simulation for {args.scenario} scenario...")

    # Run the simulation
    result = run_single_pushover_simulation(
        scenario=args.scenario,
        random_seed=args.seed
    )

    if result is None:
        print("❌ Simulation failed")
        sys.exit(1)
    else:
        capacity_point, displacement_mm, base_shear_kN, rotation_rad, base_moment_kNm = result
        vy, dy, my, thy = capacity_point
        print("✅ Simulation completed successfully!")
        print(f"📊 Capacity Point:")
        print(f"   Vy (Yield Base Shear): {vy:.1f} kN")
        print(f"   Dy (Yield Displacement): {dy:.1f} mm")
        print(f"   My (Yield Moment): {my:.1f} kNm")
        print(f"   Thy (Yield Rotation): {thy:.4f} rad")

        # Create plots
        plots_dir = Path('./Plots/single_simulation/')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Fit bilinear for D-V
        k1_v, k2_v, dy_fit, vy_fit = fit_bilinear_profile(displacement_mm, base_shear_kN)
        d_range = np.linspace(displacement_mm.min(), displacement_mm.max(), 500)
        v_model = np.where(d_range < dy_fit, k1_v * d_range, k2_v * d_range + (k1_v - k2_v) * dy_fit)

        # Plot D-V
        plt.figure(figsize=(8,6))
        plt.plot(displacement_mm, base_shear_kN, 'b-', linewidth=1.5, label='Pushover Curve')
        plt.plot(d_range, v_model, 'r--', linewidth=2, label='Bilinear Fit')
        plt.plot([dy], [vy], 'go', markersize=8, label='Yield Point (Dy, Vy)')
        plt.xlabel('Displacement (mm)', fontsize=14)
        plt.ylabel('Base Shear (kN)', fontsize=14)
        plt.title(f'Pushover Curve - Displacement vs Base Shear ({args.scenario.capitalize()})', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'pushover_D_V.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot Th-M (no bilinear fit, as capacity is derived from D-V fit)
        plt.figure(figsize=(8,6))
        plt.plot(rotation_rad, base_moment_kNm, 'b-', linewidth=1.5, label='Pushover Curve')
        plt.plot([thy], [my], 'go', markersize=8, label='Yield Point (Thy, My)')
        plt.xlabel('Rotation (rad)', fontsize=14)
        plt.ylabel('Base Moment (kNm)', fontsize=14)
        plt.title(f'Pushover Curve - Rotation vs Base Moment ({args.scenario.capitalize()})', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'pushover_Th_M.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()