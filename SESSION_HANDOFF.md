# Session Handoff

## Goal
Continue reviewing and refactoring the scour-critical bridge simulation code so it matches the manuscript workflow: climate-scenario scour selection, OpenSees pushover simulation, bilinear capacity extraction, plotting, and optional surrogate modeling.

## Current State
- Workspace: `/Users/chenzhiq/MyProjects/CriticalBridges-meeting-ClimateScenarioUncertainties`
- Manuscript reviewed: `Manuscript/revision_submission_02.tex`
- Main refactor target completed: `single_pushover_simulation.py` is now a modular scenario runner instead of a mock.
- Real OpenSees single-run was verified once with:
  `python scripts/run_single_simulation.py --scenario missouri --seed 42 --scour-depth-m 0.5 --fc-mpa 27.0 --fy-mpa 420.0`
- That run completed in about 402 seconds and generated:
  - `RecorderData/missouri/scour_500.0/*.out`
  - `Plots/single_simulation/pushover_D_V.pdf`
  - `Plots/single_simulation/pushover_Th_M.pdf`

## Important Code Changes
- `single_pushover_simulation.py`
  - Replaced placeholder/mock implementation.
  - Adds scenario case generation using `avg - 2 std` and `avg + 2 std` scour depths from LHS hazard statistics.
  - Runs explicit OpenSees pushover cases.
  - Saves per-case plots with pushover scatter, bilinear fit, rotation-moment scatter, and capacity tuple annotation.
  - Writes selected-case capacity workbook to `RecorderData/Yield_Results_selected_scenarios.xlsx`.
  - Supports optional surrogate training via `--train-surrogates`.
- `BridgeModeling/Pushover.py`
  - `run_single_pushover_simulation(...)` now accepts explicit `scour_depth_m`, `fc_MPa`, and `fy_MPa`.
  - Keeps legacy random sampling when explicit values are omitted.
  - Extracts `My` and `Thy` from recorder data instead of lever-arm approximations.
  - Fixes base moment conversion to kNm using `/1e6`.
- `scripts/run_single_simulation.py`
  - Adds CLI flags `--scour-depth-m`, `--fc-mpa`, `--fy-mpa`.
- `src/postprocessing/processing.py`
  - Fixes yield displacement unit conversion: recorder displacement is meters, output `dy_mm` is now millimeters.
- `src/scour/scour_hazard.py`
  - Moved `matplotlib.pyplot` import into the script block so importing hazard utilities does not initialize Matplotlib.
- `test_single_simulation.py`
  - Updated unpacking to match the real return value from `run_single_pushover_simulation`.

## Selected Scenario Cases
With `--hazard-samples 1000 --seed 42`, the runner selected:

| Scenario | Case | Scour Depth (m) | Mean (m) | Std (m) |
|---|---:|---:|---:|---:|
| missouri | avg_minus_2std | 0.411431 | 2.678607 | 1.133588 |
| missouri | avg_plus_2std | 4.945782 | 2.678607 | 1.133588 |
| colorado | avg_minus_2std | 0.745374 | 4.866423 | 2.060525 |
| colorado | avg_plus_2std | 8.987472 | 4.866423 | 2.060525 |
| extreme | avg_minus_2std | 1.017459 | 6.441517 | 2.712029 |
| extreme | avg_plus_2std | 11.865575 | 6.441517 | 2.712029 |

Note: the extreme scenario emitted a warning that some LHS scour samples exceed the 20 m pier-height cap in `src/scour/scour_hazard.py`.

## Commands To Run
- Six selected scenario pushover cases:
  ```bash
  python single_pushover_simulation.py --scenarios missouri colorado extreme --hazard-samples 1000 --seed 42
  ```
- Same, with selected-case surrogate training afterward:
  ```bash
  python single_pushover_simulation.py --scenarios missouri colorado extreme --hazard-samples 1000 --seed 42 --train-surrogates
  ```
- Explicit one-off pushover:
  ```bash
  python scripts/run_single_simulation.py --scenario missouri --scour-depth-m 0.5 --fc-mpa 27.0 --fy-mpa 420.0
  ```

## Verification
- Passed syntax checks:
  ```bash
  python -m py_compile src/scour/scour_hazard.py single_pushover_simulation.py BridgeModeling/Pushover.py src/postprocessing/processing.py
  ```
- Passed lightweight Phase-1 smoke test:
  ```bash
  python test_pipeline.py --scenario missouri --samples 5 --seed 1
  ```
- Verified `python single_pushover_simulation.py --help` works.
- Verified one real OpenSees run completes, but it is slow.

## Known Issues
- Real OpenSees runs are long. One Missouri 0.5 m scour case took about 402 seconds, so six selected cases may take roughly 40 minutes or more.
- The manuscript equations for shear stress, deterministic 50-year scour, stochastic final scour, and exceedance probability were corrected to match the implemented Briaud-style workflow.
- The manuscript Table 1 scenario statistics still do not match the current code/configuration. With `N=1000`, `seed=42`, and `dPier=1.5 m`, the current uncapped statistics are:
  - missouri: mean 2.678607 m, std 1.133588 m, p95 4.805227 m, max 9.021979 m
  - colorado: mean 4.866487 m, std 2.059500 m, p95 8.730127 m, max 16.391113 m
  - extreme: mean 6.445312 m, std 2.727659 m, p95 11.562425 m, max 21.708850 m
- `src/scour/scour_hazard.py` no longer caps samples at 20 m by default. Use `max_scour_depth_m=20.0` only when an explicitly truncated bridge-geometry sample set is desired.
- The selected six-case surrogate training option is mostly diagnostic. The manuscript-level surrogate claims require a much larger dataset, nominally 1000 simulations per scenario.
- The repo is dirty and includes generated artifacts and cache files:
  - modified `.DS_Store`
  - modified `__pycache__` files
  - modified generated plots
  - untracked `RecorderData/`
  - untracked `Manuscript/`
  Do not clean or revert these without user approval.

## Next Steps
1. Review `src/scour/scour_hazard.py` against manuscript equations and Table 1 statistics; decide whether code or manuscript needs correction.
2. Run the six selected scenario cases if runtime is acceptable.
3. Inspect generated plots under `Plots/scenario_pushover_cases/` and workbook `RecorderData/Yield_Results_selected_scenarios.xlsx`.
4. If results look physically plausible, decide whether to scale up toward the manuscript dataset size or keep this as a targeted diagnostic workflow.
