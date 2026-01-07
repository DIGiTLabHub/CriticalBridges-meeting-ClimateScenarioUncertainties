#!/usr/bin/env python
# Test pipeline script

import sys
from pathlib import Path
import argparse

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

# Imports
from scour import LHS_scour_hazard
from config.parameters import SCOUR
from config.logger_setup import setup_logging, get_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True, choices=['missouri', 'colorado', 'extreme'])
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()
    
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Testing pipeline...")
    scenario_params = SCOUR['scenarios'][args.scenario]
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Velocity: {scenario_params['velocity_m_s']} m/s")
    
    result = LHS_scour_hazard(lhsN=args.samples, vel=scenario_params['velocity_m_s'], dPier=1.5, gama=1e-6, zDot=scenario_params['erosion_rate_mm_hr'])
    logger.info(f"Mean scour: {result['z50Mean']:.3f} m")

if __name__ == '__main__':
    main()
