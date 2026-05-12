# Pipeline Workflow Documentation

This document describes the complete workflow for running the scour bridge simulation pipeline.

> **Snapshot note:** the workflow surface is now more coherent than earlier repo snapshots. `scripts/run_full_pipeline.py` supports reproducible Phase 1-2 generation, `scripts/run_single_simulation.py` is the preferred single-run OpenSees wrapper, and the postprocessing / training / visualization stages now have import-safe CLIs. Full OpenSees runtime is still long-running and was not re-verified in this documentation pass.

## 🚀 **Quick Start**

```bash
# Generate scour samples and material inputs
python scripts/run_full_pipeline.py --scenario missouri --samples 1000 --seed 42

# This creates:
# - Phase 1: Scour hazard samples
# - Phase 2: Material property Excel file (ready for pushover analysis)
```

---

## 📋 **Complete 7-Phase Workflow**

### **Phase 1: Scour Hazard Generation** ✅ AUTOMATED

**Purpose:** Generate probabilistic scour depth samples using Latin Hypercube Sampling

**Script:** `scripts/run_full_pipeline.py --scenario <scenario> --samples <N> [--seed <seed>]`

**Implementation note:** the current script does not expose a `--phases` argument; Phase 1 and Phase 2 are both part of the same intended run.

**Input Parameters:**
- `--scenario`: One of [missouri, colorado, extreme]
- `--samples`: Number of LHS samples (default: 1000)
- `--seed`: Optional random seed for reproducible hazard/material sampling
- Scenario parameters from `config/parameters.py`:
  - Velocity (m/s): Missouri=2.9, Colorado=6.5, Extreme=10.0
  - Erosion rate (mm/hr): Missouri=100, Colorado=500, Extreme=1000

**Process:**
1. Calculate Reynolds number: `Re = velocity * pier_diameter / kinematic_viscosity`
2. Calculate maximum scour: `zMax = 0.18 * Re^0.635`
3. Calculate 50-year scour: `z50 = tEq / (1/zDot + tEq/zMax)`
4. Generate LHS samples using standard normal distribution
5. Apply lognormal uncertainty: `z50Final = z50 * exp(0.407 * error)`
6. Calculate statistics (mean, std, lognormal parameters)

**Output:**
- Scour depth samples (z50Final) in meters
- Sorted scour depths (z50Final_sort)
- Cumulative distribution (zP)
- Statistics: z50Mean, z50std, z50LogMean, z50LogStd

**Example Output:**
```
Phase 1/7: Scour hazard generation...
Generated 1000 scour samples
   Mean 50-year scour: 2.831 m
   Max scour: 6.892 m
   Std dev: 1.234 m
```

**Files Created:**
- None (data is in-memory for Phase 2)

---

### **Phase 2: Material Property Sampling** ✅ AUTOMATED

**Purpose:** Generate concrete and steel strength samples paired with scour depths

**Script:** `scripts/run_full_pipeline.py --scenario <scenario> --samples <N> [--seed <seed>]`

**Implementation note:** material sampling is coupled to the same script execution as hazard generation in the current code.

**Input Parameters:**
- Scour depths from Phase 1
- Material distributions from `config/parameters.py`:
  - Concrete (fc): mean=27.0 MPa, std=1.0 MPa (normal)
  - Steel (fy): mean=420.0 MPa, std=4.2 MPa (lognormal)

**Process:**
1. Generate fc samples: `fc = normal(mean=27.0, std=1.0, n=samples)`
2. Generate fy samples: `fy = lognormal(log(420.0), std=4.2/420.0, n=samples)`
3. Create DataFrame with columns:
   - Sample_ID: 1 to n
   - Scour_Depth_mm: z50Final * 1000
   - fc_MPa: concrete samples
   - fy_MPa: steel samples
   - Scenario: scenario name

**Output:**
- Excel file: `data/input/Scour_Materials_{scenario}_{timestamp}.xlsx`
- Sheet name: Scenario capitalization (e.g., "Missouri")
- Columns: Sample_ID, Scour_Depth_mm, fc_MPa, fy_MPa, Scenario

**Example Output:**
```
Phase 2/7: Material property sampling...
Generated material samples
   fc range: 24.12 - 29.87 MPa
   fy range: 408.34 - 431.68 MPa
Material samples saved to: data/input/Scour_Materials_missouri_20250106_220842.xlsx
```

**Files Created:**
- `data/input/Scour_Materials_{scenario}_{timestamp}.xlsx`

---

### **Phase 3: Bridge Modeling & Pushover** ⚠️ PARTIALLY SCRIPTED

**Purpose:** Build OpenSeesPy bridge model and run pushover analysis

**Preferred single-run script:** `python scripts/run_single_simulation.py --scenario <scenario> [--seed <seed>]`

**Batch path:** legacy/manual via `BridgeModeling/Pushover.py`

**Input (preferred single-run path):**
- Scenario name and optional seed passed to `scripts/run_single_simulation.py`
- Geometry from JSON files: `data/geometry/*.json`

**Input (legacy batch path):**
- Excel file from Phase 2: `data/input/Scour_Materials_{scenario}_{timestamp}.xlsx`
- Geometry from JSON files: `data/geometry/*.json`

**Process (for each sample):**
1. **Model Setup:**
   - `build_model(fc, fy, scourDepth)`
   - Load geometry via `GeometryLoader.load_all()`
   - Apply restraints, constraints, masses
   - Define sections, materials

2. **Gravity Analysis:**
   - Apply gravity loads
   - Solve for equilibrium
   - Apply load constant

3. **Pushover Analysis:**
   - Apply lateral load incrementally
   - Record displacements, forces, moments
   - Continue until max drift ratio (0.05) or convergence

4. **Output Recording:**
    - Save to: `RecorderData/{scenario}/scour_{depth}/*.out`
    - Files include: `Displacement.5201.out`, `ColDisplacement.3201.out`, `ColLocForce.3101.out`, `ColLocForce.3201.out`, `ColLocForce.3301.out`

**Output Files:**
```
RecorderData/
├── missouri/
│   ├── scour_2500.0/
│   │   ├── Displacement.5201.out
│   │   ├── ColDisplacement.3201.out
│   │   └── ColLocForce.3201.out
│   ├── scour_3125.0/
│   │   └── ...
│   └── ...
├── colorado/
│   └── ...
└── extreme/
    └── ...
```

**Files Created:**
- `RecorderData/{scenario}/scour_{depth}/*.out` (per sample)

**Time:** long-running OpenSees job; wall-clock time depends on convergence and environment

---

### **Phase 4: Post-Processing** ✅ AVAILABLE CLI

**Purpose:** Extract yield points from pushover results

**Script:** `python src/postprocessing/processing.py`

**Input:**
- OpenSees output files from Phase 3

**Process (for each sample):**
1. **Load Data:**
   - Displacement: `Displacement.5201.out` → uy (deck displacement)
   - Column rotation: `ColDisplacement.3201.out` → θy (rotation)
   - Base shear: `ColLocForce.3201.out` → V2_j (base shear)

2. **Bilinear Fit:**
   - Fit bilinear curve to identify yield point
   - Two segments: elastic (k1) and post-yield (k2)
   - Yield displacement: dy (intersection point)
   - Yield base shear: Vy = k1 * dy

3. **Extract Yield:**
   - Vy = yield base shear (kN)
   - Dy = yield displacement (mm)
   - My = yield moment (kNm) from moment file
   - Thy = yield rotation (rad)

**Output:**
- Excel file: `RecorderData/Yield_Results_by_Scenario.xlsx`
- Summary table with columns: Sample_ID, Scour_Depth_mm, Vy_kN, Dy_mm, My_kNm, Thy_rad

**Files Created:**
- `RecorderData/Yield_Results_by_Scenario.xlsx`

---

### **Phase 5: Surrogate Model Training** ✅ AVAILABLE CLI

**Purpose:** Train GBR and SVR surrogate models for rapid capacity evaluation

**Preferred script:** `python -m src.surrogate_modeling.training`

**Input:**
- Excel file from Phase 4: `RecorderData/Yield_Results_by_Scenario.xlsx`

**Process:**
1. **Feature Engineering:**
   - Base features: Scour_Depth_mm (rename to Scour)
   - Polynomials: Scour², Scour³
   - Logarithmic: log(Scour)
   - Inverse: 1/Scour
   - Square root: sqrt(Scour)

2. **Training per Target:**
   - Targets: Vy, Dy, My, Thy
   - Features: [Scour, Scour², Scour³, log(Scour), 1/Scour, sqrt(Scour)]
   - Train GBR (Gradient Boosting): 700 estimators, max_depth=3
   - Train SVR (Support Vector Regression): RBF kernel, C=100

3. **Bootstrap Ensembles:**
   - 30 bootstrap models per target
   - Generate 30 different training datasets via resampling
   - Train 30 models per algorithm (GBR, SVR)

4. **Credal Bounds:**
   - 2.5th percentile: lower bound
   - 97.5th percentile: upper bound
   - Separate aleatoric (inherent variance) vs epistemic (model uncertainty)

**Output:**
- Trained models, credal bounds, performance metrics, and plots under `RecorderData/results/Tuple_Data_Process/`

**Files Created:**
- `RecorderData/results/Tuple_Data_Process/ML_Surrogate_Credal/` (GBR outputs)
- `RecorderData/results/Tuple_Data_Process/ML_Surrogate_Credal_SVR/` (SVR outputs)
- `RecorderData/results/Tuple_Data_Process/Credal_Model_Performance_*.xlsx`

**Speedup:** Surrogate evaluation ~0.001s vs. ~120s FEM (60,000x faster)

---

### **Phase 6: Bootstrap Uncertainty Quantification** ✅ INTEGRATED

**Purpose:** Generate credal bounds and quantify uncertainty

**Script:** Included in `src/surrogate_modeling/training.py`

**Input:**
- Bootstrap ensembles from Phase 5

**Process:**
1. **Ensemble Prediction:**
   - Predict using all 30 models
   - Get 30 predictions per sample

2. **Credal Statistics:**
   - 2.5th percentile: lower bound
   - Median prediction: central estimate
   - 97.5th percentile: upper bound
   - 95% credibility interval

3. **Scenario Comparison:**
   - Compare credal bounds across scenarios
   - Identify scenario dominance regions
   - Quantify tail risk differences

**Output:**
- Credal interval tables
- Uncertainty quantification statistics
- Scenario comparison plots

**Files Created:**
- Credal-bound workbooks and summaries under `RecorderData/results/Tuple_Data_Process/`

---

### **Phase 7: Visualization** ✅ AVAILABLE CLI

**Purpose:** Generate publication-quality plots and figures

**Preferred script:** `python -m src.visualization.visualization`

**Input:**
- Credal bounds from Phase 6
- Hazard curves from Phase 1
- Fragility relationships

**Process:**
1. **Hazard Curves:**
   - Plot P(Exceedance) vs. Scour Depth
   - Three curves: Missouri, Colorado, Extreme
   - Lognormal fits

2. **Fragility with Credal Bounds:**
   - Plot Vy vs. Scour Depth with uncertainty bands
   - Plot Dy vs. Scour Depth with uncertainty bands
   - Credal intervals: 2.5% - 97.5%

3. **Scenario Comparison:**
   - Side-by-side fragility comparison
   - Tail risk analysis
   - Scenario crossover points

**Output:**
- Figures: `RecorderData/results/visualizations/`
  - Hazard curves
  - Fragility with uncertainty
  - Credal bounds
  - Scenario comparisons

**Files Created:**
- `RecorderData/results/visualizations/**/*.png`

---

## 🔄 **End-to-End Example**

```bash
# Step 1: Generate input data (automated)
python scripts/run_full_pipeline.py --scenario missouri --samples 100 --seed 42

# Output: data/input/Scour_Materials_missouri_20250106_HHMMSS.xlsx

# Step 2: Run a single pushover analysis (long-running OpenSees job)
python scripts/run_single_simulation.py --scenario missouri --seed 42
# Note: this single-run wrapper samples its own parameters; it does not directly
# consume the Phase-2 Excel workbook. The workbook is mainly for legacy/manual
# batch simulation flows.
# Output: RecorderData/missouri/scour_*/Displacement.5201.out

# Step 3: Extract yield points (manual, ~1 minute)
python src/postprocessing/processing.py
# Uses: RecorderData/*/ColLocForce.*.out
# Output: RecorderData/Yield_Results_by_Scenario.xlsx

# Step 4: Train surrogate models
python -m src.surrogate_modeling.training
# Uses: RecorderData/Yield_Results_by_Scenario.xlsx
# Output: RecorderData/results/Tuple_Data_Process/*

# Step 5: Generate credal bounds (included in training)
# Output: Credal bounds, 95% intervals

# Step 6: Generate visualizations
python -m src.visualization.visualization
# Output: RecorderData/results/visualizations/**/*.png
```

**Total Time:**
- Automated (Phases 1-2): ~10 seconds
- Manual (Phases 3-7): ~4-5 hours (100 samples)

**With Surrogates:**
- Scenario comparison (30,000 × 3 = 90,000 evaluations): ~30 seconds

---

## 📊 **Data Sizes**

| Scenario | Samples | Scour Range (m) | Material Range (fc, fy) |
|----------|---------|-------------------|------------------------|
| Missouri | 1000 | 0.5 - 8.0 | fc: 24-30 MPa, fy: 400-440 MPa |
| Colorado | 1000 | 2.0 - 15.0 | fc: 24-30 MPa, fy: 400-440 MPa |
| Extreme | 1000 | 5.0 - 20.0 | fc: 24-30 MPa, fy: 400-440 MPa |

**Typical 1000-sample dataset size:**
- Excel file: ~50 KB
- OpenSees outputs: ~5 MB per sample
- Yield results: ~20 KB

---

## 🔧 **Troubleshooting**

### **Common Issues**

**Issue: ModuleNotFoundError when running scripts**
- **Solution:** Run from project root, not from nested subdirectories
- **Check:** `python scripts/run_full_pipeline.py --help`

**Issue: openpyxl not found**
- **Solution:** `pip install openpyxl`

**Issue: OpenSees wipe() not found**
- **Solution:** Verify openseespy installation: `pip install openseespy`

**Issue: Pipeline stalls at Phase 3**
- **Check:** Verify Excel file exists from Phase 2
- **Check:** OpenSees model convergence (non-zero exit codes)
- **Note:** Phase 3 is still the slowest and least smoke-test-friendly part of the workflow

**Issue: Memory errors with large samples**
- **Solution:** Reduce sample count, use chunk processing
- **Recommendation:** Start with 10 samples to test workflow

---

## 📚 **References**

1. **Configuration:** `config/parameters.py`
2. **Pipeline Script:** `scripts/run_full_pipeline.py`
3. **Architecture:** `project_overview.md`
4. **Scientific Context:** `README.md`
5. **Research:** Preprint/PreprintStatement.md
