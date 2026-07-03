#!/usr/bin/env python
"""Manual smoke runner for one OpenSees pushover simulation.

This invokes the real OpenSees model and can take several minutes.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from BridgeModeling.Pushover import run_single_pushover_simulation


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one real OpenSees pushover simulation and print the capacity tuple."
    )
    parser.add_argument(
        "--scenario",
        default="missouri",
        choices=("missouri", "colorado", "extreme"),
        help="Climate/scour scenario label for output organization.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for omitted parameters.")
    parser.add_argument(
        "--scour-depth-m",
        type=float,
        default=0.5,
        help="Explicit scour depth in meters, measured from the original riverbed.",
    )
    parser.add_argument("--fc-mpa", type=float, default=27.0, help="Concrete strength in MPa.")
    parser.add_argument("--fy-mpa", type=float, default=420.0, help="Steel yield strength in MPa.")
    return parser


def main():
    args = build_parser().parse_args()

    print("Testing single pushover simulation...")
    result = run_single_pushover_simulation(
        scenario=args.scenario,
        random_seed=args.seed,
        scour_depth_m=args.scour_depth_m,
        fc_MPa=args.fc_mpa,
        fy_MPa=args.fy_mpa,
    )

    if result:
        capacity_point, *_ = result
        Vy, Dy, My, Thy = capacity_point
        print(
            "Success! Capacity point: "
            f"Vy={Vy:.1f}kN, Dy={Dy:.1f}mm, My={My:.1f}kNm, Thy={Thy:.4f}rad"
        )
        return 0

    print("Simulation failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
