# Scour-Critical Bridge Capacity Learning with Climate Scenarios Deep Uncertainties

## 🎯 **Project Overview**

A distribution-agnostic framework for predicting transverse capacities and quantifying uncertainties of scour-critical bridges under climate-driven deep uncertainties. Scour-critical bridges are among the most vulnerable transportation infrastructure systems, with failure risks increasingly uncertain due to climate change and climate-driven flood hazards.

**Core Innovation:** Combines surrogate modeling of bridge capacities from nonlinear pushover simulations with credal-set uncertainty quantification, providing bounded predictions without assuming precise probabilistic distributions.

### **Supported Workflow at a Glance**

| Stage | Current Support | Primary Entry Point |
|-------|-----------------|---------------------|
| Phase 1-2: Hazard + material sampling | ✅ Smoke-testable / scripted | `python experiments/run_full_pipeline.py --scenario missouri --samples 1000 --seed 42` |
| Phase 3a: Single OpenSees run | ⚠️ Long-running | `python experiments/run_single_simulation.py --scenario missouri --seed 42` |
| Phase 3b: Batch OpenSees runs | ⏳ Legacy / manual | `BridgeModeling/Pushover.py` |
| Phase 4: Post-processing | ✅ Available CLI | `python src/postprocessing/processing.py` |
| Phase 5-6: Surrogate training + bootstrap | ✅ Available CLI | `python -m src.surrogate_modeling.training` |
| Phase 7: Visualization | ✅ Available CLI | `python -m src.visualization.visualization` |

---

## 🌊 **Problem Statement**

Scour-critical bridges represent a dominant failure mode in transportation infrastructure. Traditional approaches often overlook the effects of deep uncertainties from climate change, computing scour hazards assuming a single probabilistic distribution to analyze bridge capacity and collapse potential. This is increasingly unjustified under nonstationary climate conditions where flood magnitudes and frequencies are rising dramatically.

### **Key Limitations Addressed**

1. **Single-valued estimates** - Traditional fragility produces point estimates (e.g., "5% drift = 85% capacity") with no uncertainty quantification
2. **Climate scenario divergence** - Different flow regimes (Missouri vs. Colorado vs. Extreme) produce vastly different hazard distributions, which should not be averaged
3. **Model uncertainty accumulation** - Material properties, scour rates, structural responses all have uncertainties that compound
4. **Tail risk hidden** - Low-probability, high-consequence events are missed by single-point estimates
5. **Non-decision-relevant outputs** - Resilience planning needs uncertainty bands, not point estimates

---

## 🔬 **Core Contribution: Credal-Bounded Fragility**

This project introduces a **distribution-agnostic framework** for assessing scour-critical bridge capacity under climate-driven deep uncertainty, featuring:

1. **Surrogate modeling** of bridge capacities trained on nonlinear pushover simulations with climate-scenario-based scour profiles
2. **Credal-set uncertainty quantification** providing bounded predictions without assuming precise probabilities
3. **Bootstrap ensembles** (30 GBR/SVR models) to quantify model uncertainty and separate aleatoric vs. epistemic uncertainty
4. **Climate scenario modeling** across three distinct hydrologic regimes (Missouri, Colorado, Extreme)
5. **Scenario comparison** enabling direct comparison of risk profiles for resilient decision-making

### **Climate Scenarios**

| Scenario    | Velocity (m/s) | Erosion (mm/hr) | Mean Scour (m) | Max Scour (m) | Description                     |
|-------------|----------------|------------------|---------------|---------------|---------------------------------|
| **Missouri** | 2.9            | 100              | 2–4           | 3–4           | Moderate river with slow erosion |
| **Colorado** | 6.5            | 500              | 4–8           | 10–12         | Fast-flow river                  |
| **Extreme**  | 10.0           | 1000             | 8–15          | 15–17         | Extreme flood events             |


These scenarios represent **fundamentally different hazard regimes** with divergent stochastic properties, not merely different parameter sets.

---

## 🤖 **Computational Framework**

### **Physics-Based Modeling (OpenSeesPy)**
- 3D nonlinear finite element bridge model (4-span, 140m total)
- Fiber sections for reinforced concrete columns
- Pushover analysis to 5% drift ratio
- ~60 seconds per simulation

### **Machine Learning Surrogates**
- **Gradient Boosting Regressor (GBR)** - 700 estimators, max depth 3
- **Support Vector Regression (SVR)** - RBF kernel, C=100
- **Bootstrap ensembles** - 30 models per target (Vy, Dy, My, Thy)
- **Speedup:** ~60,000x faster than FEM (0.001s vs. 121s per sample)

### **Uncertainty Quantification**
- **Credal sets** - 95% intervals from bootstrap percentiles
- **Scenario separation** - Prevents inappropriate averaging of climate-specific responses
- **Epistemic uncertainty** - Model variance quantified via bootstrap variance
- **Aleatoric uncertainty** - Inherent variability from data

**Capacity Output Variables:**
- **Vy** - Yield base shear (kN)
- **Dy** - Yield displacement (mm)
- **My** - Yield moment (kNm)
- **Thy** - Yield rotation (rad)

---

## 📊 **Key Features**

### **Hazard Modeling**
- **Latin Hypercube Sampling (LHS)** for probabilistic scour depth prediction
- **Lognormal distribution fitting** for scour depth variability
- **Scenario-specific parameters** - Velocity, erosion rate, kinematic viscosity
- **Reynolds number calculation** - Hydraulic shear stress estimation

### **Bridge Modeling**
- **Nonlinear 3D finite element analysis** with OpenSeesPy
- **Fiber section modeling** for reinforced concrete columns
- **Zero-length elements** for soil springs at pier foundations
- **4-span continuous bridge** with 3 piers at 35m, 70m, 105m

### **Post-Processing**
- **Bilinear capacity curve fitting** with energy criterion
- **Yield point extraction** (Vy, Dy, My, Thy)
- **Single simulation support** - Run individual analyses with immediate capacity results
- **Batch processing** - Excel consolidation of results by scenario

### **Surrogate Modeling**
- **Feature engineering** - Polynomial terms, logarithms, square roots of scour depth
- **Multi-target prediction** - Vy, Dy, My, Thy from scour depth
- **Bootstrap ensembles** - 30 models for uncertainty quantification
- **Model persistence** - Saved models for rapid evaluation

### **Uncertainty Quantification**
- **Credal bounds** - 2.5th and 97.5th percentiles from bootstrap
- **Scenario comparison** - Side-by-side fragility curves with uncertainty bands
- **Tail risk analysis** - Which scenario dominates failure probability at high scour
- **Time-evolution foundation** - Framework for nonstationary risk assessment

---

## 🏗️ **Architecture**

### **Project Structure**

```
CriticalBridges-meeting-ClimateScenarioUncertainties/
├── config/                      # Centralized configuration
│   ├── parameters.py            # All project parameters
│   ├── paths.py                # File path management
│   └── logger_setup.py         # Logging configuration
│
├── src/                         # Main package
│   ├── scour/                 # Scour hazard modeling
│   ├── bridge_modeling/        # OpenSees bridge modeling
│   ├── postprocessing/          # Post-processing utilities
│   ├── surrogate_modeling/     # ML surrogates
│   └── visualization/          # Plotting tools
│
├── BridgeModeling/              # Legacy bridge modeling (backward compatible)
│   ├── geometry/              # Geometry data (JSON-based)
│   └── Pushover.py            # Pushover analysis
│
├── data/                        # Data files
│   ├── geometry/              # JSON geometry data (5,400+ lines)
│   ├── input/                 # Material sample inputs
│   └── output/                # Simulation results
│
├── experiments/                     # Automation
│   ├── run_full_pipeline.py   # Pipeline orchestrator
│   ├── run_single_simulation.py # Single simulation runner
│   └── test_surrogate_modeling.py # Surrogate model testing tool
│
├── RecorderData/              # Legacy simulation outputs
├── archive/old_scripts/        # Archived Jupyter notebooks
└── README.md                  # This file
```

### **Modular Design**
- **Separation of concerns** - Each module has single responsibility
- **Clear interfaces** - Well-defined public APIs
- **Independent updates** - Modules can be modified without affecting others
- **Easy testing** - Individual modules can be unit tested

---

## 🚀 **Installation & Usage**

### **Installation**

```bash
# Clone repository
git clone <repository-url>
cd CriticalBridges-meeting-ClimateScenarioUncertainties

# Install dependencies
pip install -r requirements.txt

# Install in development mode (recommended)
pip install -e .
```

### **Dependencies**
- **Core:** numpy>=1.21.0, pandas>=1.3.0, scipy>=1.7.0
- **Visualization:** matplotlib>=3.4.0, seaborn>=0.11.0
- **Finite Element:** openseespy>=3.5.0
- **Machine Learning:** scikit-learn>=1.0.0, joblib>=1.2.0
- **File I/O:** openpyxl>=3.0.0
- **Development (optional):** jupyter>=1.0.0, ipykernel>=6.0.0
- **Python:** >=3.8

> **Environment note (current snapshot):** this repository is presently being used in a legacy Python 3.8 environment, but current OpenSeesPy wheels on PyPI may require Python 3.10+ for a fresh install. For new environments, prefer Python 3.10+ unless you are intentionally reproducing an older setup.

### **Quick Start**

```bash
# Option 1: Smoke-test the main imports
python -c "from src.scour import LHS_scour_hazard; from src.bridge_modeling.geometry.geometry_loader import GeometryLoader; from config.parameters import SCOUR, MATERIALS, ANALYSIS; print('imports-ok')"

# Option 2: Generate reproducible scour + material samples (Phases 1-2)
python experiments/run_full_pipeline.py --scenario missouri --samples 10 --seed 42

# Option 3: Inspect the single-simulation CLI before launching OpenSees
python experiments/run_single_simulation.py --help

# Option 4: Run a single OpenSees simulation (long-running; not a smoke test)
python experiments/run_single_simulation.py --scenario missouri

# Option 5: Aggregate recorder outputs after simulations finish
python src/postprocessing/processing.py --help

# Option 6: Train surrogate models from Yield_Results_by_Scenario.xlsx
python -m src.surrogate_modeling.training --help

# Option 7: Generate plots from postprocessing / surrogate outputs
python -m src.visualization.visualization --help
```

### **Current Script Status (verified in this review)**

| Script | Status | Notes |
|--------|--------|-------|
| `experiments/run_full_pipeline.py` | ✅ Available | CLI works; generates Phase 1-2 scour/material inputs and supports `--seed` for reproducibility |
| `experiments/run_single_simulation.py` | ⚠️ Long-running | CLI works; full OpenSees execution is intentionally not treated as a lightweight smoke test |
| `src/postprocessing/processing.py` | ✅ Available | Import-safe CLI for aggregating `RecorderData/<scenario>/scour_*` into `Yield_Results_by_Scenario.xlsx` |
| `src/surrogate_modeling/training.py` | ✅ Available | Import-safe CLI for GBR/SVR bootstrap surrogate training |
| `src/visualization/visualization.py` | ✅ Available | Import-safe CLI for supported postprocessing / surrogate plots |
| `experiments/test_surrogate_modeling.py` | ✅ Compatibility wrapper | CLI/help path works; requires existing yield-results data and is secondary to `src.surrogate_modeling.training` |

### **What the workflow is intended to do**

```bash
# Phase 1-2: Generate scour samples and material inputs
python experiments/run_full_pipeline.py --scenario missouri --samples 1000 --seed 42

# Output: data/input/Scour_Materials_missouri_TIMESTAMP.xlsx
# Next: run OpenSees simulations, then aggregate + train + visualize
```

### **Module Import Examples**

```python
# Scour hazard modeling
from src.scour import LHS_scour_hazard

result = LHS_scour_hazard(lhsN=1000, vel=2.9, dPier=1.5, gama=1e-6, zDot=100)
print(f"Mean scour: {result['z50Mean']:.3f} m")

# Load geometry data
from src.bridge_modeling.geometry.geometry_loader import GeometryLoader

loader = GeometryLoader()
nodes = loader.load_nodes()  # 1,892 nodes from JSON

# Configuration access
from config.parameters import SCOUR, MATERIALS, ANALYSIS

missouri_params = SCOUR['scenarios']['missouri']
print(f"Velocity: {missouri_params['velocity_m_s']} m/s")
```

### **Surrogate Model Testing**

The repository includes `experiments/test_surrogate_modeling.py`, intended to train/test bootstrap GBR/SVR surrogate models on `RecorderData/Yield_Results_by_Scenario.xlsx`.

**Current status:** the script now exposes a working CLI/help path and can still be used as a compatibility/testing wrapper. For the cleaned primary workflow, prefer `python -m src.surrogate_modeling.training`.

### **Pipeline Workflow**

| Phase | Status | Description |
|--------|--------|-------------|
| **1: Hazard** | ✅ Automated | Scour depth samples |
| **2: Sample** | ✅ Automated | Material Excel file |
| **3a: Single Simulate** | ⚠️ Available | `experiments/run_single_simulation.py` exists, but is a long-running OpenSees job |
| **3b: Batch Simulate** | ⏳ Manual / legacy | Driven by `BridgeModeling/Pushover.py` rather than a clean scripted batch entry point |
| **4: Post-process** | ✅ Available | `src/postprocessing/processing.py` is an import-safe CLI for recorder aggregation |
| **5: Train** | ✅ Available | `src/surrogate_modeling/training.py` provides the primary surrogate-training CLI |
| **6: Bootstrap** | ✅ Integrated | Credal/bootstrap workflow is built into the training module |
| **7: Visualize** | ✅ Available | `src/visualization/visualization.py` provides a clean plotting CLI |

---

## 📖 **Documentation**

- **[README.md](README.md)** - This file (quick start, installation, features)
- **[project_overview.md](project_overview.md)** - Technical architecture reference
- **[PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)** - Complete workflow documentation
<!--- - **[SimulationCodeRefactoring.md](SimulationCodeRefactoring.md)** - Research to production guide
--->

---

## 📈 **Performance**

| Process | Traditional Time | Surrogate Time | Speedup |
|----------|-----------------|----------------|---------|
| Build Model | ~60s | 60s | 1x |
| Pushover | ~60s | 60s | 1x |
| Post-process | ~1s | 1s | 1x |
| **Total FEM** | ~121s | 121s | 1x |
| **Surrogate eval** | ~121s | **0.001s** | **60,000x faster** |

**Impact:** Enables evaluation of 90,000 scenarios (3 scenarios × 30,000 samples) in **~30 seconds** vs. **~3,000 hours** with FEM.

---

## 🎓 **Scientific Significance**

### **Research Questions Addressed**

1. **How does fragility vary across flow regimes?**
   - Framework provides direct comparison of Missouri, Colorado, Extreme scenarios
   - Scenario-specific credal bounds enable quantification of divergence

2. **Which scenario dominates risk in tail regions?**
   - Bootstrap ensembles identify scenario dominance at high scour depths
   - Tail risk quantification via 2.5% - 97.5% credal intervals

3. **What is the probability that capacity exceeds threshold under different climate conditions?**
   - Scenario-specific fragility functions
   - Credal bounds provide uncertainty quantification
   - Enables risk envelope calculations

4. **Should retrofit prioritize a specific scenario?**
   - Scenario-to-scenario comparison with uncertainty bands
   - Decision-relevant risk envelopes for resilience planning

### **Key Scientific Contributions**

1. **Replaces false probabilistic certainty** with defensible uncertainty bounds
2. **Demonstrates climate-driven variability dominates fragility estimates**
3. **Establishes rigorous bridge** between climate science and structural reliability
4. **Provides decision-relevant uncertainty envelopes** for resilience planning
5. **Enables large-scale scenario comparison** previously infeasible with FEM

---

## 📋 **Decision Support**

| **Stakeholder** | **Challenge Addressed** | **Framework Provides** |
|-----------------|-------------------------|------------------------|
| **State DOTs** | Which scenarios dominate tail risk? | Scenario-to-scenario comparison with credal bounds |
| **FEMA** | How does climate affect fragility? | Climate-driven hazard scenarios |
| **Bridge Owners** | What's the retrofit priority under uncertainty? | Scenario-specific risk envelopes |
| **Transportation Agencies** | Decision-relevant risk bounds | Uncertainty envelopes vs. fragile point estimates |
| **Asset Managers** | Prioritization under deep uncertainty | Risk profiles without subjective probability assignments |
| **Researchers** | Reproducible methodology | Modular, version-controllable codebase |


---

## 🔬 **Conclusion**

This framework transforms **experimental research code** into **scientific software** that:

- ✅ **Addresses climate scenario divergence** via three distinct hydrologic regimes
- ✅ **Quantifies deep uncertainty** with bootstrap credal bounds
- ✅ **Enables scenario comparison** for climate pathway assessment
- ✅ **Reduces computational time** by 60,000x for large-scale analysis
- ✅ **Provides decision-relevant risk bounds** for resilience planning under deep uncertainty
- ✅ **Is reproducible** with clear documentation and modular structure

**The framework establishes a precedent for climate-aware structural risk assessment that accounts for scenario dominance and tail uncertainty - a critical need for infrastructure planning under nonstationary climate. Results reveal how variable uncertainties propagate through the nonlinear soil–foundation–structure bridge system when expressing the capacity tuple (Vy, Dy, My, Thy).**

---

**Version:** 0.3.0 (Climate-Enhanced)
**Date:** January 6, 2025
**Keywords:** Scour hazard, bridge fragility, climate scenarios, credal sets, imprecise probability, uncertainty quantification, scenario dominance, risk assessment
