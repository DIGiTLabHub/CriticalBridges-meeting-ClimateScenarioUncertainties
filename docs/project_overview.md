# Project Overview

## System Description

<<<<<<< HEAD
The Scour-Critical Bridge Simulator is a distribution-agnostic framework for predicting transverse capacities and quantifying uncertainties of scour-critical bridges under climate-driven deep uncertainties. It develops a novel analytics toolkit supporting engineers and transportation agencies in making resilient asset management decisions.
=======
The 'CriticalBridges-meeting-ClimateScenarioUncertainties' project provides a probabilistic framework for assessing bridge fragility under progressive scour conditions. It combines:
>>>>>>> fb11340784eac9829381763022e351ea5d229dc5

**Key Components:**
1. **Structural mechanics** - OpenSeesPy 3D nonlinear finite element analysis for bridge capacity simulation
2. **Hazard modeling** - Climate-scenario-based scour profiles using Latin Hypercube Sampling
3. **Machine learning** - Surrogate models trained on nonlinear pushover simulations (GBR & SVR)
4. **Uncertainty quantification** - Credal-set bounds ensuring objective capacity estimation without precise probabilistic assumptions

## Research Highlights

- **Distribution-agnostic framework** for scour-critical bridge capacity assessment under climate-driven deep uncertainty
- **Surrogate models trained** using nonlinear pushover simulations with climate-scenario-based scour inputs
- **Credal-set uncertainty quantification** introduced to provide bounded predictions, ensuring objective capacity estimation
- **Results support resilient decision-making** for infrastructure adaptation and asset management under uncertain climate futures

---

## Bridge Model

The model represents a 4-span continuous bridge with:
- **Total length**: 140m (4 spans × 35m each)
- **Deck width**: 12m
- **Number of bents**: 3 (piers at 35m, 70m, 105m)
- **Column height**: 13.05m (above ground)
- **Pier diameter**: 1.5m

### Finite Element Model

- **Element type**: Nonlinear beam-column elements with fiber sections
- **Materials**:
  - Concrete: Concrete01 (confined, crushing + cracking)
  - Steel: Steel01 (bilinear with hardening)
- **Soil-structure interaction**: ZeroLength elements with PySimple1, QzSimple1, TzSimple1 springs

### Analysis Type

- **Gravity analysis**: Apply self-weight loads
- **Pushover analysis**: Monotonic lateral load to 5% drift
- **DOFs**: 6 DOFs per node (3 translation + 3 rotation)

---

## Scour Hazard Model

### Scour Mechanism

Local scour at bridge piers due to flow around obstacles.

### Governing Equation

Uses Reynolds number-based empirical relationship:

```
z_max = f(Re, d_pier, ...)
z_50 = z_max * f(t_50)
```

Where:
- `Re` = Reynolds number
- `d_pier` = Pier diameter
- `z_max` = Maximum possible scour depth
- `z_50` = 50-year scour depth

### Probabilistic Approach

**Latin Hypercube Sampling (LHS)** generates spatially-uniform random samples:
- Input: Flow velocity, pier diameter, kinematic viscosity
- Output: Scour depth distribution
- Distribution: Lognormal (based on physics)

### Scenarios

| Scenario | Velocity (m/s) | Erosion Rate (mm/h) | Mean Depth (m) | Max Depth (m) |
|----------|----------------|---------------------|---------------|-------------|
| Missouri | 2.9 | 100 | 1-2 | 3-4 |
| Colorado | 6.5 | 500 | 4-8 | 10-12 |
| Extreme | 10.0 | 1000 | 8-15 | 15-17 |

---

## Surrogate Modeling

### Why Surrogates?

OpenSees pushover analysis is computationally expensive:
- Single analysis: ~30-60 seconds
- 1000 samples × 3 scenarios = 60,000 analyses
- Total time: ~500-1000 hours

Surrogates enable rapid evaluation (<0.001 seconds per prediction).

### ML Algorithms

**Gradient Boosting Regressor (GBR)**
- Ensemble of weak learners (decision trees)
- Hyperparameters: 700 trees, max depth 3, learning rate 0.015
- Advantages: Handles non-linear relationships, robust to outliers

**Support Vector Regression (SVR)**
- RBF kernel with epsilon-insensitive loss
- Hyperparameters: C=100, epsilon=0.01
- Advantages: Good for small datasets, strong generalization

### Bootstrap Ensemble

- **Purpose**: Quantify model uncertainty
- **Method**: 30 bootstrap resamples, train 30 GBR and 30 SVR models
- **Output**: Credal bounds [min, median, max]
- **Coverage**: ~95% for credible intervals

### Feature Engineering

Polynomial and transformation features to capture non-linear scour-capacity relationship:

```
X = [Sz, Sz², Sz³, log(Sz), 1/Sz, √Sz]
```

Where Sz = scour depth

### Performance Metrics

- **R²** - Coefficient of determination (higher is better)
- **RMSE** - Root Mean Squared Error (lower is better)
- **MAPE** - Mean Absolute Percentage Error

---

## Workflow Summary

```
Input: Flow parameters → Scour samples
                              ↓
                    Material sampling
                              ↓
                  Bridge model building (OpenSeesPy)
                              ↓
                 Pushover analysis (slow: ~60s each)
                              ↓
             Bilinear fitting → Capacity tuples
                              ↓
           Surrogate training (GBR + SVR)
                              ↓
        Fast predictions + uncertainty bounds
```

**Time savings**: ~60,000 hours → ~1 hour (after training)

---

## Key Publications & References

1. Johnson, P. A., & M. E. Schiff. (1992). *Scour at Bridge Piers*. National Highway Cooperative Research Program Report.

2. McKenna, F., & R. A. (1985). *Bridge Scour: A Review of Methods for Prediction and Assessment*. US Department of Transportation.

3. Arulmoli, K., & R. A. (2009). *Bridge scour countermeasures: A review*. Structure and Infrastructure Engineering.

---

## Development History

### Version 0.2.0 (January 2025) - **MAJOR REFACTORING**
- Extracted ~5,400 lines of hardcoded geometry data to JSON
- Created modular package structure
- Centralized configuration system
- Added comprehensive documentation
- Improved maintainability

### Version 0.1.0 - Initial version
- Original monolithic structure
- Mixed notebook and Python files
- Hardcoded data in Python files
