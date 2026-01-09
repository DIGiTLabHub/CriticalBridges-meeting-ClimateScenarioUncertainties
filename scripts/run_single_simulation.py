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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from BridgeModeling.Pushover import run_single_pushover_simulation


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
    capacity_point = run_single_pushover_simulation(
        scenario=args.scenario,
        random_seed=args.seed
    )

    if capacity_point is None:
        print("❌ Simulation failed")
        sys.exit(1)
    else:
        vy, dy, my, thy = capacity_point
        print("✅ Simulation completed successfully!")
        print(f"📊 Capacity Point:")
        print(f"   Vy (Yield Base Shear): {vy:.1f} kN")
        print(f"   Dy (Yield Displacement): {dy:.1f} mm")
        print(f"   My (Yield Moment): {my:.1f} kNm")
        print(f"   Thy (Yield Rotation): {thy:.4f} rad")


if __name__ == "__main__":
    main()