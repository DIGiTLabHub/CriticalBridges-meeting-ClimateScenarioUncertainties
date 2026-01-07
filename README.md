# Scour-Critical Bridge Capacity Learning with Climate Scenarios Deep Uncertainties

## 🎯 **Project Overview**

A probabilistic framework for assessing bridge fragility under **progressive scour conditions** across three distinct climate scenarios (Missouri River, Colorado River, Extreme Floods).

**Core Innovation:** Replaces traditional point-estimate fragility with **scenario-bounded credal sets** that quantify model uncertainty without assuming precise probabilities.

---

## 🌊 **Problem Statement**

Scour-critical bridges represent a dominant failure mode in transportation infrastructure, but existing fragility frameworks face a critical limitation: **they assume static or stationary climate conditions**. This is increasingly unjustified under nonstationary climate change where flood magnitudes and frequencies are rising dramatically.

### **Key Limitations Addressed**

1. **Single-valued estimates** - Traditional fragility produces point estimates (e.g., "5% drift = 85% capacity") with no uncertainty quantification
2. **Climate scenario divergence** - Different flow regimes (Missouri vs. Colorado vs. Extreme) produce vastly different hazard distributions, which should not be averaged
3. **Model uncertainty accumulation** - Material properties, scour rates, structural responses all have uncertainties that compound
4. **Tail risk hidden** - Low-probability, high-consequence events are missed by single-point estimates
5. **Non-decision-relevant outputs** - Resilience planning needs uncertainty bands, not point estimates

---

## 🔬 **Core Contribution: Credal-Bounded Fragility**

This project introduces a **surrogate-based, imprecise-probability fragility framework** that:

1. **Models three distinct climate scenarios** - Each representing different hydrologic regimes
2. **Quantifies model uncertainty** via bootstrap ensembles (30 GBR/SVR models)
3. **Provides credal bounds** (2.5% - 97.5% intervals) instead of confidence intervals
4. **Separates aleatoric vs. epistemic uncertainty** - Distinguishes inherent variability from knowledge uncertainty
5. **Enables scenario comparison** - Direct comparison of risk profiles without inappropriate averaging

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
├── scripts/                     # Automation
│   └── run_full_pipeline.py   # Pipeline orchestrator
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

### **Quick Start**

```bash
# Generate scour samples and material inputs (automated phases)
python scripts/run_full_pipeline.py --scenario missouri --samples 1000

# This creates data/input/Scour_Materials_missouri_TIMESTAMP.xlsx
# Ready for bridge modeling (Phase 3)
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

### **Pipeline Workflow**

| Phase | Status | Description |
|--------|--------|-------------|
| **1: Hazard** | ✅ Automated | Scour depth samples |
| **2: Sample** | ✅ Automated | Material Excel file |
| **3: Simulate** | ⏳ Manual | OpenSees pushover analysis |
| **4: Post-process** | ⏳ Manual | Yield point extraction |
| **5: Train** | ⏳ Manual | Surrogate models |
| **6: Bootstrap** | ⏳ Manual | Credal bounds |
| **7: Visualize** | ⏳ Manual | Plots & figures |

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

**The framework establishes a precedent for climate-aware structural risk assessment that accounts for scenario dominance and tail uncertainty - a critical need for infrastructure planning under nonstationary climate.**

---

**Version:** 0.3.0 (Climate-Enhanced)
**Date:** January 6, 2025
**Keywords:** Scour hazard, bridge fragility, climate scenarios, credal sets, imprecise probability, uncertainty quantification, scenario dominance, risk assessment
