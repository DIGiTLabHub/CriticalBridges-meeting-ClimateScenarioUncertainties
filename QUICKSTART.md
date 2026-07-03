# Quick Start Guide

> **Snapshot note:** this repository still mixes fast smoke-testable stages with long-running OpenSees workflows, but the user-facing CLI surface is now more coherent. `experiments/run_full_pipeline.py` supports reproducible Phase 1-2 generation, `experiments/run_single_simulation.py` is the preferred single-run OpenSees wrapper, and the postprocessing / training / visualization stages now have import-safe CLIs.

## ✅ Supported Workflow at a Glance

| Stage | Current Support | Primary Entry Point |
|-------|-----------------|---------------------|
| Phase 1-2: Hazard + material sampling | ✅ Smoke-testable / scripted | `python experiments/run_full_pipeline.py --scenario missouri --samples 1000 --seed 42` |
| Phase 3a: Single OpenSees run | ⚠️ Long-running | `python experiments/run_single_simulation.py --scenario missouri --seed 42` |
| Phase 3b: Batch OpenSees runs | ⏳ Legacy / manual | `BridgeModeling/Pushover.py` |
| Phase 4: Post-processing | ✅ Available CLI | `python src/postprocessing/processing.py` |
| Phase 5-6: Surrogate training + bootstrap | ✅ Available CLI | `python -m src.surrogate_modeling.training` |
| Phase 7: Visualization | ✅ Available CLI | `python -m src.visualization.visualization` |

## 📦 Installation

### Prerequisites
- Python 3.10+ recommended for new environments (this repo is also used in a legacy Python 3.8 setup)
- pip package manager

### Install from Source

```bash
# Clone repository
git clone <repository-url>
cd CriticalBridges-meeting-ClimateScenarioUncertainties

# Install dependencies
pip install -r requirements.txt

# Install in development mode (recommended)
pip install -e .

# Verify installation
python -c "from src.scour import LHS_scour_hazard; print('✅ Installation successful!')"
```

### Install from PyPI (Future)

This project is not currently documented as a published PyPI package release. Prefer source installation for now.

---

## 🚀 Running the Pipeline

### Phase 1-2: Automated (Scour Hazard + Material Sampling)

```bash
# Run complete automated workflow
python experiments/run_full_pipeline.py --scenario missouri --samples 1000 --seed 42

# This creates:
# - Phase 1: Scour hazard samples
# - Phase 2: Material property Excel file (ready for pushover)
# Output: data/input/Scour_Materials_{scenario}_{timestamp}.xlsx
```

**Current status:** this command now works as the supported Phase 1-2 entry point and supports `--seed` for reproducibility.

**Available Scenarios:**
- `missouri` - Moderate flow (2.9 m/s, 100 mm/hr erosion)
- `colorado` - Fast flow (6.5 m/s, 500 mm/hr erosion)
- `extreme` - Extreme flow (10.0 m/s, 1000 mm/hr erosion)

### Phase 3-7: Sequential Execution

After Phase 1-2, continue with the remaining stages in order:

```bash
# Step 3: Single OpenSees simulation (long-running)
python experiments/run_single_simulation.py --scenario missouri --seed 42
# Output: RecorderData/{scenario}/scour_{depth}/*.out
# Note: for batch simulation, the legacy/manual BridgeModeling path still applies.

# Step 4: Post-Processing & Capacity Extraction
python src/postprocessing/processing.py
# Uses: RecorderData/{scenario}/scour_{depth}/*.out
# Output: RecorderData/Yield_Results_by_Scenario.xlsx

# Step 5: Surrogate Model Training
python -m src.surrogate_modeling.training
# Uses: RecorderData/Yield_Results_by_Scenario.xlsx
# Output: RecorderData/results/Tuple_Data_Process/*

# Step 6: Bootstrap Uncertainty Quantification (included in Step 5)
# Uses: RecorderData/results/Tuple_Data_Process/*
# Output: Credal bounds (2.5% - 97.5% intervals)

# Step 7: Visualization
python -m src.visualization.visualization
# Uses: Credal bounds from Step 6
# Output: RecorderData/results/visualizations/**/*.png
```

---

## 📁 Project Structure Overview

```
CriticalBridges-meeting-ClimateScenarioUncertainties/
├── config/                      # Configuration management
├── src/                         # Main package
│   ├── scour/                 # Scour hazard modeling
│   ├── bridge_modeling/        # OpenSees modeling
│   ├── postprocessing/          # Post-processing utilities
│   ├── surrogate_modeling/       # ML surrogates
│   └── visualization/           # Plotting tools
├── BridgeModeling/              # Legacy (backward compatible)
├── data/                        # All data files
│   ├── geometry/              # JSON geometry data
│   ├── input/                 # Material sample inputs
│   └── output/                # Simulation results
├── experiments/                     # Automation
└── README.md, project_overview.md, PIPELINE_WORKFLOW.md
```

---

## 🧪 Usage Examples

### Generate Scour Samples

```python
from src.scour import LHS_scour_hazard

# Generate 1000 samples for Missouri scenario
result = LHS_scour_hazard(lhsN=1000, vel=2.9, dPier=1.5, gama=1e-6, zDot=100)
print(f"Mean scour: {result['z50Mean']:.3f} m")
print(f"Max scour: {result['z50Final'].max():.3f} m")
```

### Load Geometry Data

```python
from src.bridge_modeling.geometry.geometry_loader import GeometryLoader

loader = GeometryLoader()
nodes = loader.load_nodes()  # Returns 1,892 nodes
restraints = loader.load_restraints()  # Returns 1,152 restraints
constraints = loader.load_constraints()  # Returns 615 constraints
masses = loader.load_masses()  # Returns 49 masses

# Load all at once
geometry = loader.load_all()
```

### Run Configuration

```python
from config.parameters import SCOUR, MATERIALS, ANALYSIS

# Access scenario parameters
missouri_params = SCOUR['scenarios']['missouri']
print(f"Velocity: {missouri_params['velocity_m_s']} m/s")

# Access material distributions
concrete = MATERIALS["concrete"]
print(f"Concrete: {concrete['mean_MPa']} ± {concrete['std_MPa']} MPa")

# Access analysis parameters
pushover = ANALYSIS["pushover"]
print(f"Max drift: {pushover['max_drift_ratio']}")
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** ModuleNotFoundError when running scripts
```bash
# Run from project root, not from experiments/ directory
cd CriticalBridges-meeting-ClimateScenarioUncertainties
python experiments/run_full_pipeline.py --scenario missouri --help
```

If imports still fail, verify you are using the project root and that dependencies were installed into the active environment.

**Issue:** ImportError: No module named 'openpyxl'
```bash
pip install openpyxl
```

**Issue:** OpenSees functions not found
```bash
pip install openseespy
```

**Issue:** Scikit-learn not installed
```bash
pip install scikit-learn
```

---

## 📚 Additional Documentation

- **Technical Overview:** See [project_overview.md](project_overview.md)
- **Pipeline Workflow:** See [PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)
- **Docs Status:** `docs/api_reference.md` is not present in the current snapshot

---

## 🎓 Citation

If you use this code in research, please cite:

```bibtex
@software{scour_bridge_simulator,
  title={Scour-Critical Bridge Capacity Learning with Climate Scenarios Deep Uncertainties},
  author={Chen, Z. and others},
  year={2025},
  version={0.3.0},
  url={https://github.com/<owner>/CriticalBridges-meeting-ClimateScenarioUncertainties},
  note={A probabilistic framework for assessing bridge fragility under progressive scour conditions using OpenSeesPy and machine learning surrogates}
}
```
