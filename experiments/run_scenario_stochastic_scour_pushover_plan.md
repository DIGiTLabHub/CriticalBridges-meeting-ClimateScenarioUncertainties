# Plan: Scenario-Based Stochastic Scour Pushover Experiment

## Objective

Create `experiments/run_scenario_stochastic_scour_pushover.py` as the scenario-level stochastic experiment driver. The script will generate Latin-hypercube input samples for scour and material uncertainties, optionally stop after input-distribution simulation/plotting, or continue to nonlinear pushover and capacity tuple extraction for every sampled row.

## Inputs

- Scenarios come from `config.parameters.SCOUR["scenarios"]`.
- Scour sampling uses the existing `src.scour.LHS_scour_hazard` implementation.
- Material sampling uses Latin-hypercube normal-space samples:
  - concrete strength `fc_MPa`: normal distribution from `MATERIALS["concrete"]`;
  - steel yield strength `fy_MPa`: lognormal distribution using the existing project convention, `log(mean_MPa)` and `std_MPa / mean_MPa`.
- Feasible scour bins come from `SCOUR["spring_removal_depths_m"]`.

## Scour Discretization

For both histograms and FE pushover runs, a continuous sampled scour depth `s_z_continuous_m` will be repositioned to a feasible modeled depth `s_z_model_m` by ceiling to the next configured spring-removal depth. Values below zero map to `0 m`; values above the configured maximum map to the terminal `20 m` bin.

This keeps the stochastic input plots and the nonlinear model inputs aligned. The `20 m` bin remains the physical pile-tip terminal case and is recorded as zero capacity without launching OpenSees.

## Phase 1: Stochastic Input Simulation

For each selected scenario:

1. Generate `N` Latin-hypercube samples of scour depth, `fc_MPa`, and `fy_MPa`.
2. Save the scenario sample table with both continuous and model-discretized scour depths.
3. Save a combined sample table across all scenarios.
4. Plot and save:
   - PDF scour-depth histogram with selected scenarios shown as semi-transparent colored columns using shared feasible-depth bin edges in one Nature-style figure;
   - PDF scour-depth cumulative histogram/empirical CDF with the selected scenarios overlaid in one Nature-style figure;
   - PDF material histograms for `fc_MPa` and `fy_MPa`, with scenarios shown as colored columns in one Nature-style figure;
   - PDF material cumulative histograms/empirical CDFs, with scenarios overlaid in one Nature-style figure.

The CLI will include an input-only flag so this phase can be run without OpenSees.

## Phase 2: Nonlinear Pushover

Unless the input-only flag is passed:

1. Iterate over the sampled rows for all selected scenarios.
2. Pass `s_z_model_m`, `fc_MPa`, and `fy_MPa` into `BridgeModeling.Pushover.run_single_pushover_simulation`.
3. Reuse the existing adaptive DisplacementControl pushover settings and expose the same CLI overrides used by deterministic experiments.
4. Save each raw pushover curve with sample metadata.
5. Extract and save the capacity tuple:
   - `V_y_kN`
   - `Delta_y_mm`
   - `M_x_kNm`
   - `theta_x_rad`
6. Save a single capacity tuple CSV for downstream surrogate/nonparametric modeling.
7. Save `capacity_tuples_progress.csv` after every processed sample so a multi-day run leaves a usable checkpoint if interrupted.

## Outputs

Default root: `experiments/output/scenario_stochastic_scour_pushover`

- `samples/scenario_<name>_samples.csv`
- `samples/all_scenario_samples.csv`
- `capacity_tuples.csv`
- `capacity_tuples_progress.csv`
- `raw_curves/<scenario>/sample_<index>_scour_<depth>m.csv`
- `plots/<scenario>/*`
- `plots/materials/*`
- `manifest.json`

## Verification

Before long OpenSees sweeps:

1. Run `python -m py_compile` on the new script.
2. Run the new script with `--input-only` and a small sample count.
3. Run `--dry-run` to confirm the planned stochastic pushover workload.
4. Run `git diff --check`.
