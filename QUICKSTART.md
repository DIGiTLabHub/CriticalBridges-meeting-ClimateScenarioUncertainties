# Quick Start Guide

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
# Clone repository
git clone <repository-url>
cd ScourCriticalBridgeSimulators

# Install dependencies
pip install -r requirements.txt

# Install in development mode (recommended)
pip install -e .

# Verify installation
python -c "from src.scour import LHS_scour_hazard; print('✅ Installation successful!')"
```

### Install from PyPI (Future)

```bash
pip install scour-critical-bridge-simulator
```

---

## 🚀 Running the Pipeline

### Phase 1-2: Automated (Scour Hazard + Material Sampling)

```bash
# Run complete automated workflow
python scripts/run_full_pipeline.py --scenario missouri --samples 1000

# This creates:
# - Phase 1: Scour hazard samples
# - Phase 2: Material property Excel file (ready for pushover)
# Output: data/input/Scour_Materials_{scenario}_{timestamp}.xlsx
```

**Available Scenarios:**
- `missouri` - Moderate flow (2.9 m/s, 100 mm/hr erosion)
- `colorado` - Fast flow (6.5 m/s, 500 mm/hr erosion)
- `extreme` - Extreme flow (10.0 m/s, 1000 mm/hr erosion)

### Phase 3-7: Manual (Requires Sequential Execution)

After Phase 1-2, run the remaining phases manually:

```bash
# Step 3: Bridge Modeling & Pushover Analysis (~3.5 hours for 1000 samples)
python BridgeModeling/Pushover.py
# Uses: data/input/Scour_Materials_{scenario}_{timestamp}.xlsx
# Output: RecorderData/{scenario}/scour_{depth}/*.out

# Step 4: Post-Processing & Capacity Extraction (~5 minutes)
python src/postprocessing/processing.py
# Uses: RecorderData/{scenario}/scour_{depth}/*.out
# Output: RecorderData/Yield_Results_by_Scenario.xlsx

# Step 5: Surrogate Model Training (~5 minutes)
python src/surrogate_modeling/training.py
# Uses: RecorderData/Yield_Results_by_Scenario.xlsx
# Output: data/output/models/*.pkl

# Step 6: Bootstrap Uncertainty Quantification (included in Step 5)
# Uses: data/output/models/*.pkl
# Output: Credal bounds (2.5% - 97.5% intervals)

# Step 7: Visualization (~2 minutes)
python src/visualization/visualization.py
# Uses: Credal bounds from Step 6
# Output: data/output/plots/*.pdf
```

---

## 📁 Project Structure Overview

```
ScourCriticalBridgeSimulators/
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
├── scripts/                     # Automation
└── README.md, project_overview.md, PIPELINE_WORKFLOW.md
```

---

## 🧪 Usage Examples

### Generate Scour Samples

```python
from scour import LHS_scour_hazard

# Generate 1000 samples for Missouri scenario
result = LHS_scour_hazard(lhsN=1000, vel=2.9, dPier=1.5, gama=1e-6, zDot=100)
print(f"Mean scour: {result['z50Mean']:.3f} m")
print(f"Max scour: {result['zMax']:.3f} m")
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
# Run from project root, not from scripts/ directory
cd ScourCriticalBridgeSimulators
python scripts/run_full_pipeline.py --scenario missouri

# OR use python -m
cd scripts
python -m run_full_pipeline --scenario missouri
```

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
- **API Reference:** See [docs/api_reference.md](docs/api_reference.md) (coming soon)

---

## 🎓 Citation

If you use this code in research, please cite:

```bibtex
@software{scour_bridge_simulator,
  title={Scour-Critical Bridge Capacity Learning with Climate Scenarios Deep Uncertainties},
  author={Chen, Z. and others},
  year={2025},
  version={0.3.0},
  url={https://github.com/yourusername/ScourCriticalBridgeSimulators},
  note={A probabilistic framework for assessing bridge fragility under progressive scour conditions using OpenSeesPy and machine learning surrogates}
}
```
