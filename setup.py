"""
Setup script for installing Scour-Critical Bridge Simulator.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
readme_desc = ""
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        # Extract first paragraph for short description
        for line in f:
            if line.startswith('# '):
                readme_desc += line[1:].strip() + ' '
                continue
            break

# Read requirements for dependencies
requirements_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        for line in f:
            if line and not line.startswith('#'):
                pkg_spec = line.strip().split('>=')[0].strip()
                install_requires.append(pkg_spec)

setup(
    name="scour-critical-bridge-simulator",
    version="0.2.0",
    description=readme_desc or "A probabilistic framework for assessing bridge fragility under scour conditions using OpenSeesPy and machine learning surrogates.",
    long_description="""
# Scour-Critical Bridge Simulator
# =============================

A probabilistic framework for assessing bridge fragility under progressive scour conditions using OpenSeesPy and machine learning surrogates.

## Features

* 🔬 Scour Hazard Modeling
  - Latin Hypercube Sampling (LHS) for probabilistic scour depth prediction
  - Three pre-defined flow scenarios (Missouri, Colorado, Extreme)
  - Lognormal distribution fitting

* 🌉 3D Bridge Modeling (OpenSeesPy)
  - Nonlinear finite element model
  - 4-span continuous bridge (140m total)
 3 piers at 35m, 70m, 105m
  - Fiber sections with confining reinforcement
  - PySimple1, QzSimple1, TzSimple1 soil springs

* 📊 Pushover Analysis
  - Monotonic lateral load to 5% drift
  Multiple solver fallback algorithms
  Automatic batch processing

* 📈 Post-Processing
  - Bilinear capacity curve fitting (energy criterion)
  - Extract: Vy, Dy, My, Thy
  Excel consolidation of results

* 🤖 Machine Learning Surrogates
  Gradient Boosting Regressor (GBR) + Support Vector Regression (SVR)
 30-model bootstrap ensembles for uncertainty quantification
  Credal bounds [min, median, max]

* 📉 Visualization
  Hazard curves
  Capacity vs. scour plots
  ML model performance metrics
  Sensitivity analysis with dominance coloring

* ⚙️ Configuration Management
  Centralized parameter system (config/parameters.py)
  File path management (config/paths.py)
  Logging configuration (config/logging.py)

## Project Structure

```
scour-critical-bridge-simulator/
├── config/                   # Central configuration
│   ├── parameters.py           # All project parameters
│   ├── paths.py                # File paths
│   └── logging.py              # Logging setup
├── data/                      # All data files
│   └── geometry/                # Bridge geometry (JSON)
├── src/                        # Main package
│   ├── scour/                 # Scour hazard
│   ├── bridge_modeling/        # OpenSees modeling
│   ├── postprocessing/          # Post-processing
│   ├── surrogate_modeling/       # ML models
│   └── visualization/           # Visualization
├── scripts/                    # Automation
└── docs/                       # Documentation
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Run Full Pipeline

```bash
# Run complete workflow for Missouri scenario
python scripts/run_full_pipeline.py --scenario missouri --samples 100

# Run for extreme case
python scripts/run_full_pipeline.py --scenario extreme --samples 1000
```

### Run Individual Modules

```python
# Generate scour samples
from src.scour.hazard import LHS_scour_hazard

# Build bridge model
from src.bridge_modeling.model_setup import build_model

# Run pushover analysis
from src.bridge_modeling.analysis.pushover import run_pushover_batch

# Train surrogate models
from src.surrogate_modeling.training import train_surrogates

# Create visualizations
from src.visualization.visualization import create_all_plots
```

## Documentation

- [Project Overview](docs/project_overview.md) - Technical overview
- [README.md](README.md) - User-facing documentation
- [API Reference](docs/api_reference.md) - Function documentation (TODO)

## Citation

If you use this code in research, please cite:

> Chen, Z., et al. (2025). Scour-Critical Bridge Simulator: 
> A probabilistic framework for bridge fragility assessment. 
> Available at: https://github.com/yourusername/ScourCriticalBridgeSimulators

## License

MIT License - see [LICENSE](LICENSE) file for details
""",
    long_description_content_type="text/markdown",
    author="Research Team",
    python_requires=">=3.8",
    packages=find_packages(where="src", exclude=["test*"]),
    install_requires=install_requires or [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'joblib>=1.3.0',
        'openpyxl>=3.0.0',
        'openseespy>=3.5.0',
    ],
    include_package_data=True,
    package_data={
        'src.scour': ['scour_hazard.py'],
        'src.bridge_modeling': [
            'model_setup.py',
            'os_model_functions.py',
            'geometry/*.py',
            'materials/*.py',
            'components/*.py',
            'analysis/*.py',
        ],
        'src.postprocessing': [
            'processing.py',
            'bilinear_fit.py',
        ],
        'src.surrogate_modeling': [
            'training.py',
        ],
        'src.visualization': [
            'visualization.py',
        ],
        'config': [
            'parameters.py',
            'paths.py',
            'logging.py',
        ],
        'scripts': [
            'run_full_pipeline.py',
        ],
    },
    entry_points={
        'console_scripts': [
            'scripts.run_full_pipeline:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
    ],
    keywords=[
        'scour',
        'bridge',
        'fragility',
        'openseespy',
        'finite element',
        'surrogate',
        'machine learning',
        'probabilistic analysis',
        'structural engineering',
        'structural reliability',
    ],
    project_urls={
        'Bug Tracker': 'https://github.com/yourusername/ScourCriticalBridgeSimulators/issues',
        'Documentation': 'https://github.com/yourusername/ScourCriticalBridgeSimulators/docs',
        'Source Code': 'https://github.com/yourusername/ScourCriticalBridgeSimulators',
    },
)
