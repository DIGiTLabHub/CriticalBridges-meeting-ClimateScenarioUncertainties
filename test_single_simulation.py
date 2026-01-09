#!/usr/bin/env python
"""
Test script for single pushover simulation
"""

import sys
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'BridgeModeling'))
sys.path.insert(0, str(PROJECT_ROOT / 'config'))

# Test the function
from Pushover import run_single_pushover_simulation

if __name__ == "__main__":
    print("Testing single pushover simulation...")
    result = run_single_pushover_simulation(scenario='missouri', random_seed=42)

    if result:
        Vy, Dy, My, Thy = result
        print(f"Success! Capacity point: Vy={Vy:.1f}kN, Dy={Dy:.1f}mm, My={My:.1f}kNm, Thy={Thy:.4f}rad")
    else:
        print("Simulation failed")