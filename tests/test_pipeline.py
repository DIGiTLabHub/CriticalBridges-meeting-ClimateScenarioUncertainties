#!/usr/bin/env python
"""Lightweight pipeline smoke test (Phase 1 hazard/config only).

This script intentionally validates imports, configuration wiring, and hazard sampling
without invoking OpenSees simulations or claiming end-to-end pipeline execution.
"""

import argparse
import sys
from pathlib import Path


def _bootstrap_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_bootstrap_repo_root()

from config.logger_setup import get_logger, setup_logging
from config.parameters import SCOUR
from src.scour import LHS_scour_hazard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quick smoke test for Phase-1 hazard generation and configuration imports. "
            "Does NOT run OpenSees or full pipeline phases."
        )
    )
    parser.add_argument(
        "--scenario",
        default="missouri",
        choices=sorted(SCOUR["scenarios"].keys()),
        help="Climate scenario to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of LHS hazard samples for smoke testing (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic hazard sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_to_file=False)
    logger = get_logger(__name__)

    scenario_params = SCOUR["scenarios"][args.scenario]
    logger.info("Running lightweight pipeline smoke test (Phase 1/config only).")
    logger.info("Scenario: %s", args.scenario)
    logger.info("Samples: %d", args.samples)
    logger.info("Seed: %s", args.seed if args.seed is not None else "None")

    hazard_result = LHS_scour_hazard(
        lhsN=args.samples,
        vel=scenario_params["velocity_m_s"],
        dPier=1.5,
        gama=SCOUR["kinematic_viscosity_m2_s"],
        zDot=scenario_params["erosion_rate_mm_hr"],
        random_seed=args.seed,
    )

    logger.info("Hazard smoke test complete.")
    logger.info("Mean scour depth: %.3f m", hazard_result["z50Mean"])
    logger.info("Std scour depth: %.3f m", hazard_result["z50std"])


if __name__ == "__main__":
    main()
