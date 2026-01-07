#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from scour import LHS_scour_hazard

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run scour pipeline")
    parser.add_argument('--scenario', required=True, choices=['missouri', 'colorado', 'extreme'])
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()

    logging.info("=" * 50)
    logging.info("SCOUR BRIDGE SIMULATION PIPELINE")
    logging.info("=" * 50)
    logging.info(f"Scenario: {args.scenario}")
    logging.info(f"Samples: {args.samples}")

    # Scenario parameters
    scenarios = {
        'missouri': {'vel': 2.9, 'zDot': 100},
        'colorado': {'vel': 6.5, 'zDot': 500},
        'extreme': {'vel': 10.0, 'zDot': 1000}
    }
    params = scenarios[args.scenario]
    logging.info(f"Velocity: {params['vel']} m/s, Erosion: {params['zDot']} mm/hr")
    logging.info("")

    # Generate scour samples
    logging.info("Phase 1: Scour hazard generation...")
    result = LHS_scour_hazard(
        lhsN=args.samples, vel=params['vel'], dPier=1.5, gama=1e-6, zDot=params['zDot']
    )
    logging.info(f"Generated {args.samples} scour samples")
    logging.info(f"   Mean 50-year scour: {result['z50Mean']:.3f} m")
    logging.info(f"   Max scour: {np.max(result['z50Final']):.3f} m")
    logging.info("")

    # Generate material samples
    logging.info("Phase 2: Material property sampling...")
    np.random.seed(42)
    fc_samples = np.random.normal(27.0, 1.0, args.samples)
    fy_samples = np.random.lognormal(np.log(420.0), 4.2/420.0, args.samples)

    df = pd.DataFrame({
        'Sample_ID': range(1, args.samples + 1),
        'Scour_Depth_mm': result['z50Final'] * 1000,
        'fc_MPa': fc_samples,
        'fy_MPa': fy_samples,
        'Scenario': [args.scenario] * args.samples
    })

    # Save
    input_dir = PROJECT_ROOT / 'data' / 'input'
    input_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = input_dir / f"Scour_Materials_{args.scenario}_{timestamp}.xlsx"
    df.to_excel(filepath, index=False, sheet_name=args.scenario.capitalize())
    logging.info(f"Material samples saved to: {filepath}")
    logging.info("")
    logging.info("=" * 50)
    logging.info("AUTOMATED PHASES COMPLETE!")
    logging.info("=" * 50)
    logging.info("Next steps:")
    logging.info("  1. Run: python BridgeModeling/Pushover.py")
    logging.info("  2. Run: python src/postprocessing/processing.py")
    logging.info("  3. Run: python src/surrogate_modeling/training.py")

if __name__ == '__main__':
    main()
