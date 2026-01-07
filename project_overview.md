# Scour-Critical Bridge Simulator - Project Overview

## 🏗️ **Architecture**

This project implements a climate-aware bridge fragility assessment framework using:
- **Physics-based modeling** (OpenSeesPy)
- **Machine learning surrogates** (GBR + SVR)
- **Bootstrap uncertainty quantification** (credal sets)
- **Scenario-based hazard modeling** (Missouri, Colorado, Extreme)

### **Project Structure**

```
ScourCriticalBridgeSimulators/
├── config/                      # Centralized configuration
│   ├── parameters.py            # All project parameters (materials, scour, analysis)
│   ├── paths.py                # File path management
│   └── logger_setup.py         # Logging configuration
│
├── src/                         # Main source modules
│   ├── scour/                 # Scour hazard modeling
│   │   ├── __init__.py
│   │   └── scour_hazard.py   # LHS sampling for scour depths
│   │
│   ├── bridge_modeling/        # Bridge model building
│   │   ├── __init__.py
│   │   ├── model_setup.py       # Main model building function
│   │   ├── geometry/            # Geometry data (JSON-based)
│   │   │   ├── __init__.py
│   │   │   └── geometry_loader.py
│   │   ├── materials/           # Material definitions
│   │   ├── components/          # Structural components
│   │   └── analysis/            # Analysis utilities
│   │
│   ├── postprocessing/         # Post-processing utilities
│   │   ├── __init__.py
│   │   ├── processing.py         # Extract yield points
│   │   └── bilinear_fit.py      # Bilinear fitting
│   │
│   ├── surrogate_modeling/     # Machine learning
│   │   ├── __init__.py
│   │   └── training.py          # GBR/SVR training
│   │
│   └── visualization/          # Plotting tools
│       ├── __init__.py
│       └── visualization.py
│
├── BridgeModeling/              # Legacy bridge modeling (backward compatible)
│   ├── model_setup.py           # Uses geometry_loader from JSON
│   ├── geometry/
│   │   └── geometry_loader.py  # Duplicate for compatibility
│   ├── Pushover.py             # Pushover analysis
│   ├── ColSection.py           # Column sections
│   ├── SectionMat.py           # Section materials
│   ├── Element.py              # Element definitions
│   ├── Node.py                 # Node definitions
│   ├── Constraint.py           # Constraints
│   ├── Restraint.py           # Restraints
│   ├── Mass.py                # Masses
│   └── os_model_functions.py   # OpenSees utilities
│
├── data/                        # Data files
│   ├── geometry/              # JSON geometry data (5,400+ lines)
│   │   ├── nodes.json         # 1,892 nodes
│   │   ├── elements.json      # 797 elements
│   │   ├── restraints.json   # 1,152 restraints
│   │   ├── constraints.json  # 615 constraints
│   │   └── masses.json       # 49 masses
│   ├── input/                 # Input data (material samples)
│   ├── output/                # Output data (simulation results)
│   └── logs/                  # Simulation logs
│
├── scripts/                     # Automation scripts
│   └── run_full_pipeline.py   # Main workflow orchestrator
│
├── RecorderData/              # Legacy simulation outputs
├── archive/old_scripts/        # Archived Jupyter notebooks
├── tests/                     # Unit tests (future)
├── docs/                      # Documentation (future)
├── requirements.txt             # Python dependencies
├── setup.py                  # Package setup
├── README.md                  # Project documentation
└── project_overview.md         # This file
```

---

## 📊 **Data Flow**

### **Phase 1: Scour Hazard Generation**
```
Input:
  - Scenario parameters (velocity, erosion rate)
  - LHS sample count

Process:
  - Latin Hypercube Sampling
  - Hydraulic calculations
  - Lognormal distribution fitting

Output:
  - Scour depth samples (z50Final)
  - Mean, std, lognormal parameters
  - Hazard curves
```

### **Phase 2: Material Sampling**
```
Input:
  - Scour depths from Phase 1
  - Material property distributions (fc, fy)

Process:
  - Random sampling from distributions
  - Pair with scour depths
  - Generate Excel input file

Output:
  - Excel file: data/input/Scour_Materials_{scenario}_{timestamp}.xlsx
  - Columns: Sample_ID, Scour_Depth_mm, fc_MPa, fy_MPa, Scenario
```

### **Phase 3: Bridge Modeling & Pushover**
```
Input:
  - Material samples from Phase 2
  - Geometry data from JSON files
  - Scour depth (applied to pier supports)

Process:
  - Build OpenSeesPy model
  - Apply gravity loads
  - Run pushover analysis
  - Record responses

Output:
  - RecorderData/{scenario}/scour_{depth}/*.out
  - Displacements, forces, moments
  - Time history results
```

### **Phase 4: Post-Processing**
```
Input:
  - OpenSees output files from Phase 3

Process:
  - Extract base shear vs. displacement
  - Extract column rotation
  - Bilinear fit to identify yield point
  - Extract yield parameters: Vy, Dy, My, Thy

Output:
  - Excel file: Yield_Results_by_Scenario.xlsx
  - Summary table by scenario
```

### **Phase 5: Surrogate Training**
```
Input:
  - Yield results from Phase 4

Process:
  - Feature engineering (polynomial terms, log, sqrt)
  - Train GBR models
  - Train SVR models
  - Bootstrap ensembles (30 models)
  - Generate credal bounds

Output:
  - Trained models: data/output/models/
  - Ensemble predictions: credal bounds
  - R2 scores, MSE metrics
```

### **Phase 6: Uncertainty Quantification**
```
Input:
  - Bootstrap ensembles from Phase 5
  - Credal bounds

Process:
  - Calculate 2.5% and 97.5% percentiles
  - Separate aleatoric vs epistemic uncertainty
  - Scenario comparison

Output:
  - Credal intervals per scenario
  - Uncertainty statistics
  - Risk envelopes
```

### **Phase 7: Visualization**
```
Input:
  - Credal bounds
  - Hazard curves
  - Fragility relationships

Process:
  - Plot hazard curves by scenario
  - Plot fragility with uncertainty bands
  - Generate comparison plots

Output:
  - Figures: data/output/plots/
  - Publication-quality figures
```

---

## 🔄 **Configuration Management**

### **Central Parameters** (`config/parameters.py`)

```python
GEOMETRY = {
    "span_length_m": 35,
    "deck_width_m": 12,
    "num_spans": 4,
    "column_height_m": 13.05,
    "pier_diameter_m": 1.5,
}

MATERIALS = {
    "concrete": {
        "mean_MPa": 27.0,
        "std_MPa": 1.0,
        "distribution": "normal"
    },
    "steel": {
        "mean_MPa": 420.0,
        "std_MPa": 4.2,
        "distribution": "lognormal"
    }
}

SCOUR = {
    "kinematic_viscosity_m2_s": 1.0e-6,
    "density_water_kg_m3": 1000,
    "years_of_exposure": 50,
    "num_lhs_samples_per_scenario": 1000,
    "scenarios": {
        "missouri": {"velocity_m_s": 2.9, "erosion_rate_mm_hr": 100},
        "colorado": {"velocity_m_s": 6.5, "erosion_rate_mm_hr": 500},
        "extreme": {"velocity_m_s": 10.0, "erosion_rate_mm_hr": 1000}
    }
}

ANALYSIS = {
    "pushover": {
        "max_drift_ratio": 0.05,
        "tolerance": 1.0e-6,
        "max_iterations": 100
    }
}

SURROGATE = {
    "features": ["Sz", "Sz_sq", "Sz_cu", "log_Sz", "inv_Sz", "sqrt_Sz"],
    "targets": ["Vy", "Dy", "My", "Thy"],
    "gbr": {"n_estimators": 700, "max_depth": 3},
    "svr": {"kernel": "rbf", "C": 100},
    "bootstrap_ensemble_size": 30
}
```

### **Path Management** (`config/paths.py`)

All paths centralized in `PATHS` dictionary:
- `project_root`: Project root directory
- `data_root`: data/ directory
- `geometry_data`: data/geometry/ directory
- `input_data`: data/input/ directory
- `output_data`: data/output/ directory
- `recorder_data`: RecorderData/ directory (legacy)

---

## 🎯 **Key Innovations**

### **1. Geometry JSON-Based Data**
- **Before:** 5,400+ lines hardcoded in Python files
- **After:** Clean JSON files with metadata
- **Benefits:** Version-controllable, human-readable, modular

### **2. Climate Scenario Architecture**
- **Traditional:** Single scenario, single hazard curve
- **Refactored:** Three scenarios (Missouri, Colorado, Extreme)
- **Benefits:** Scenario comparison, tail risk analysis, climate sensitivity

### **3. Bootstrap Credal Sets**
- **Traditional:** Confidence intervals (assumes normality)
- **Refactored:** Bootstrap ensembles (non-parametric)
- **Benefits:** Robust to non-normality, conservative bounds, quantifies epistemic uncertainty

### **4. Modular Package Structure**
- **Traditional:** Monolithic Jupyter notebooks
- **Refactored:** Package hierarchy with clear interfaces
- **Benefits:** Testability, maintainability, installation via pip

---

## 🚀 **Workflow Automation**

The `scripts/run_full_pipeline.py` orchestrates the full workflow:

### **Usage**

```bash
# Run all automated phases
python scripts/run_full_pipeline.py --scenario missouri --samples 1000

# Run specific phase
python scripts/run_full_pipeline.py --scenario colorado --samples 500 --phases hazard

# Run all phases (automated + manual)
python scripts/run_full_pipeline.py --scenario extreme --phases all
```

### **Pipeline Phases**

| Phase | Status | Description | Output |
|--------|----------|-------------|--------|
| 1: Hazard | ✅ Automated | Scour depth samples |
| 2: Sample | ✅ Automated | Material Excel file |
| 3: Simulate | ⏳ Manual | OpenSees results |
| 4: Post-process | ⏳ Manual | Yield results |
| 5: Train | ⏳ Manual | Surrogate models |
| 6: Bootstrap | ⏳ Manual | Credal bounds |
| 7: Visualize | ⏳ Manual | Plots & figures |

### **Phase 1-2 Flow**

```
run_full_pipeline.py
  ↓
LHS_scour_hazard() → scour depths
  ↓
generate_material_samples() → paired with fc, fy
  ↓
Save to Excel → data/input/Scour_Materials_{scenario}_{timestamp}.xlsx
```

### **Phase 3-7 Flow**

Each phase requires manual execution with specific scripts:
- `BridgeModeling/Pushover.py`: Runs pushover analysis
- `src/postprocessing/processing.py`: Extracts yield points
- `src/surrogate_modeling/training.py`: Trains ML models
- `src/visualization/visualization.py`: Generates plots

---

## 📈 **Performance Metrics**

| Process | Time (per sample) | Total (1000 samples) | Speedup |
|----------|-------------------|------------------------|---------|
| Scour hazard | < 1s | ~1s | 1x |
| Material sampling | < 1s | ~1s | 1x |
| Model building | ~60s | ~16.7 hours | 1x |
| Pushover analysis | ~60s | ~16.7 hours | 1x |
| Total FEM | ~121s | ~33.5 hours | 1x |
| Surrogate eval | < 0.001s | ~1s | 60,000x |
| 30,000 evals | ~30s | ~30s | 60,000x |

**Key Insight:** Surrogate models enable scenario comparison (30,000 × 3 = 90,000 evaluations) in minutes vs. months with FEM.

---

## 🔧 **Dependencies**

### **Core Dependencies**
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `openseespy`: Structural analysis
- `scipy`: Statistical functions
- `matplotlib`: Plotting

### **ML Dependencies**
- `scikit-learn`: GBR, SVR, cross-validation
- `joblib`: Model persistence

### **File I/O**
- `openpyxl`: Excel file I/O
- `json`: JSON geometry data

---

## 📝 **Development Guidelines**

### **Adding New Features**

1. **Define parameters** in `config/parameters.py`
2. **Add paths** in `config/paths.py` if needed
3. **Implement logic** in appropriate `src/` module
4. **Add to exports** in module `__init__.py`
5. **Update pipeline** if automating workflow

### **Testing**

1. Unit tests: `tests/test_*.py` (future)
2. Integration tests: Run pipeline with small sample count
3. Validation: Compare with published results

### **Documentation**

1. Update `README.md` for user-facing changes
2. Update `project_overview.md` for architecture changes
3. Add docstrings to public APIs
4. Add examples to `docs/` (future)

---

## 🎓 **Research Context**

### **Problem Statement**

Scour-critical bridges represent a dominant failure mode, but existing fragility frameworks assume stationary climate. This is increasingly unjustified under nonstationary climate change.

### **Solution Approach**

1. **Climate Scenarios:** Model distinct hydrologic regimes
2. **Bootstrap Uncertainty:** Quantify model uncertainty without assuming normality
3. **Credal Sets:** Provide uncertainty bounds, not point estimates
4. **Surrogate Acceleration:** Enable rapid scenario comparison

### **Scientific Contributions**

- Replaces false probabilistic certainty with defensible uncertainty bounds
- Demonstrates climate-driven variability dominates fragility estimates
- Establishes rigorous bridge between climate science and structural reliability
- Sets precedent for climate-aware structural risk assessment

---

## 📊 **Data Schemas**

### **Geometry JSON Schema**

```json
{
  "metadata": {
    "description": "...",
    "total_items": N,
    "created_date": "YYYY-MM-DD",
    "units": "..."
  },
  "items": {
    "id": {
      "property": value,
      ...
    }
  }
}
```

### **Material Samples Schema**

```python
{
    "Sample_ID": int,
    "Scour_Depth_mm": float,
    "fc_MPa": float,
    "fy_MPa": float,
    "Scenario": str
}
```

### **Yield Results Schema**

```python
{
    "Sample_ID": int,
    "Scour_Depth_mm": float,
    "Vy_kN": float,      # Yield base shear
    "Dy_mm": float,       # Yield displacement
    "My_kNm": float,     # Yield moment
    "Thy_rad": float,    # Yield rotation
    "Scenario": str
}
```

---

## 🔗 **Module Dependencies**

```
scour/
  └── LHS_scour_hazard()

bridge_modeling/
  ├── model_setup.build_model()
  │   └── GeometryLoader.load_*()
  │
  └── os_model_functions.*
      ├── define_nodes()
      ├── apply_restraints()
      ├── apply_constraints()
      └── apply_masses()

postprocessing/
  └── processing.extract_yield_points()

surrogate_modeling/
  └── training.train_surrogates()

visualization/
  └── visualization.generate_plots()
```

---

## 📖 **Version History**

- **v0.3.0** (2025-01-06): Climate-enhanced framework
  - JSON-based geometry
  - Modular package structure
  - Pipeline automation
  - Scenario-based hazard modeling
  - Bootstrap uncertainty quantification

- **v0.2.0** (Earlier): Initial refactoring
  - Moved to package structure
  - Extracted geometry data
  - Centralized configuration

- **v0.1.0** (Original): Research code
  - Jupyter notebooks
  - Hardcoded parameters
  - Manual workflow
