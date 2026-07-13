#!/usr/bin/env python
# coding: utf-8

"""
Pushover analysis - Automated batch pushover analysis for multiple samples.
Refactored to use central configuration and new module structure.
"""

try:
    import openseespy.opensees as op
except ImportError as e:
    raise ImportError(
        "OpenSeesPy is required but not installed. "
        "Install with: pip install openseespy>=3.5.0\n"
        f"Original error: {e}"
    ) from e
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import tempfile
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from .RecorderColFiber import *
from .model_setup import build_model
from config.paths import get_latest_excel_file, get_simulation_output_folder
from config.parameters import ANALYSIS, MATERIALS, SCOUR
from src.postprocessing.bilinear_fit import fit_bilinear_profile


FORCE_COLUMNS = [
    "time",
    "P_i",
    "V2_i",
    "V3_i",
    "T_i",
    "M2_i",
    "M3_i",
    "P_j",
    "V2_j",
    "V3_j",
    "T_j",
    "M2_j",
    "M3_j",
]


def _load_force_response(force_file):
    force_data = pd.read_csv(force_file, sep=r"\s+", header=None, names=FORCE_COLUMNS)
    shear = np.abs(force_data["V2_j"].to_numpy()) / 1000.0
    moment = np.abs(force_data["M3_j"].to_numpy()) / 1e6
    return shear, moment


def _load_column_rotation(rotation_file):
    col_disp_data = pd.read_csv(rotation_file, sep=r"\s+", header=None)
    return np.abs(col_disp_data.iloc[:, 10].to_numpy())


def _close_recorders():
    """Flush and close OpenSees recorders before reading recorder files."""
    try:
        op.record()
    except Exception:
        pass
    try:
        op.remove("recorders")
    except Exception:
        pass


def _set_pushover_strategy(
    load_node,
    ctrl_dof,
    step_mm,
    num_iter,
    min_step_mm,
    max_step_mm,
    test_type,
    tol,
    max_iter,
    algorithm,
):
    op.test(test_type, tol, max_iter)
    op.algorithm(*algorithm)
    op.integrator(
        "DisplacementControl",
        load_node,
        ctrl_dof,
        step_mm,
        num_iter,
        min_step_mm,
        max_step_mm,
    )
    op.analysis("Static")


def _run_adaptive_pushover(
    load_node,
    ctrl_dof,
    target_displacement_mm,
    initial_increment_mm,
    min_increment_mm,
    max_increment_mm,
    num_iter,
):
    """Run pushover using OpenSees adaptive DisplacementControl bounds."""
    strategies = [
        ("EnergyIncr", 1.0e-6, 100, ("Newton",)),
        ("NormDispIncr", 1.0e-6, 100, ("NewtonLineSearch",)),
        ("EnergyIncr", 1.0e-5, 200, ("KrylovNewton",)),
        ("NormDispIncr", 1.0e-5, 200, ("Broyden", 8)),
        ("NormDispIncr", 1.0e-4, 300, ("ModifiedNewton",)),
    ]
    completed_displacement_mm = abs(op.nodeDisp(load_node, ctrl_dof))
    step_count = 0
    current_increment_mm = min(initial_increment_mm, max_increment_mm)
    chunk_steps = 25

    while completed_displacement_mm < target_displacement_mm - 1.0e-9:
        converged = False
        remaining_mm = target_displacement_mm - completed_displacement_mm
        current_increment_mm = min(current_increment_mm, remaining_mm, max_increment_mm)
        remaining_steps = max(1, int(np.ceil(remaining_mm / max_increment_mm - 1.0e-9)))
        trial_steps = min(chunk_steps, remaining_steps)

        for test_type, tol, max_iter, algorithm in strategies:
            _set_pushover_strategy(
                load_node,
                ctrl_dof,
                current_increment_mm,
                num_iter,
                min_increment_mm,
                max_increment_mm,
                test_type,
                tol,
                max_iter,
                algorithm,
            )
            before_displacement_mm = abs(op.nodeDisp(load_node, ctrl_dof))
            ok = op.analyze(trial_steps)
            after_displacement_mm = abs(op.nodeDisp(load_node, ctrl_dof))
            completed_in_chunk = max(
                0,
                int(
                    round(
                        (after_displacement_mm - before_displacement_mm)
                        / max(min(current_increment_mm, max_increment_mm), min_increment_mm)
                    )
                ),
            )
            step_count += completed_in_chunk
            completed_displacement_mm = after_displacement_mm

            if ok == 0:
                last_increment_mm = max(
                    (after_displacement_mm - before_displacement_mm) / trial_steps,
                    min_increment_mm,
                )
                current_increment_mm = min(
                    max(last_increment_mm, min_increment_mm), max_increment_mm
                )
                converged = True
                break

            if completed_displacement_mm > before_displacement_mm + 1.0e-9:
                current_increment_mm = max(
                    min(completed_displacement_mm - before_displacement_mm, max_increment_mm),
                    min_increment_mm,
                )
                converged = True
                break

        if not converged and current_increment_mm > min_increment_mm + 1.0e-12:
            reduced_increment_mm = max(0.5 * current_increment_mm, min_increment_mm)
            print(
                "⚠️ Adaptive pushover retrying with smaller displacement "
                f"increment: {current_increment_mm:.6g} -> "
                f"{reduced_increment_mm:.6g} mm"
            )
            current_increment_mm = reduced_increment_mm
            continue

        if not converged:
            print(
                "⚠️ Adaptive pushover could not converge beyond "
                f"{completed_displacement_mm:.3f} mm "
                f"(OpenSees DisplacementControl bounds: "
                f"{min_increment_mm:.3f} to {max_increment_mm:.3f} mm)."
            )
            return -1, step_count, completed_displacement_mm

    return 0, step_count, completed_displacement_mm


def _sample_material_properties(random_seed=None):
    rng = np.random.default_rng(random_seed)
    fc_mean = MATERIALS["concrete"]["mean_MPa"]
    fc_std = MATERIALS["concrete"]["std_MPa"]
    fy_mean = MATERIALS["steel"]["mean_MPa"]
    fy_std = MATERIALS["steel"]["std_MPa"]

    fc_MPa = float(rng.normal(fc_mean, fc_std))
    fy_MPa = float(rng.lognormal(np.log(fy_mean), fy_std / fy_mean))
    return fc_MPa, fy_MPa


def _sample_scour_depth_m(scenario, random_seed=None):
    rng = np.random.default_rng(random_seed)
    scenario_params = SCOUR["scenarios"][scenario]
    vel = scenario_params["velocity_m_s"]

    # Compatibility sampler for the legacy single-run path. Scenario sweep tools
    # should pass an explicit scour_depth_m from the hazard model instead.
    scour_mean_m = 0.5 + 0.1 * vel
    scour_std_m = 0.2
    return float(rng.lognormal(np.log(scour_mean_m), scour_std_m / scour_mean_m))


def run_single_pushover_simulation(
    scenario="missouri",
    random_seed=None,
    *,
    scour_depth_m=None,
    fc_MPa=None,
    fy_MPa=None,
):
    """
    Run a single OpenSees-Py nonlinear pushover simulation with sampled parameters.

    This function:
    1. Samples one data point of key parameters (scour depth, fc, fy)
    2. Builds the OpenSees-Py bridge model
    3. Runs gravity analysis
    4. Runs pushover analysis with displacement control
    5. Extracts displacement and base shear data
    6. Performs bilinear regression on the pushover curve
    7. Returns the capacity point as a tuple (Vy, Dy, My, Thy)

    Parameters
    ----------
    scenario : str, optional
        Climate scenario ('missouri', 'colorado', 'extreme'), default='missouri'
    random_seed : int, optional
        Random seed for reproducible sampling, default=None

    Returns
    -------
    tuple
        Capacity point (Vy_kN, Dy_mm, My_kNm, Thy_rad)
        Returns None if simulation fails
    """
    start_time = time.time()
    print("⏱️  Starting simulation...")

    # === 1. Sample parameters ===
    if scenario not in SCOUR["scenarios"]:
        raise ValueError(f"Unknown scenario '{scenario}'. Expected one of {sorted(SCOUR['scenarios'])}.")

    if scour_depth_m is None:
        scour_depth_m = _sample_scour_depth_m(scenario, random_seed=random_seed)

    sampled_fc, sampled_fy = _sample_material_properties(random_seed=random_seed)
    if fc_MPa is None:
        fc_MPa = sampled_fc
    if fy_MPa is None:
        fy_MPa = sampled_fy

    print(
        f"🔄 Sampled parameters: Scour={scour_depth_m:.3f}m, fc={fc_MPa:.1f}MPa, fy={fy_MPa:.1f}MPa"
    )

    # === 2. Build model ===
    model_start = time.time()
    print("🏗️  Building OpenSees model...")
    scour_depth_mm = scour_depth_m * 1000.0
    op.wipe()
    build_model(fc_MPa, fy_MPa, scour_depth_mm)
    print("✅ Model built")

    # === 3. Gravity analysis ===
    print("⚖️  Running gravity analysis...")
    op.constraints("Transformation")
    op.numberer("RCM")
    op.system("BandGeneral")
    op.algorithm("Newton")
    op.test("NormDispIncr", 1.0e-6, 1000)
    op.integrator("LoadControl", 1.0)
    op.analysis("Static")

    result = op.analyze(1)
    if result != 0:
        print("❌ Gravity analysis failed")
        return None

    op.loadConst("-time", 0.0)
    print("✅ Gravity analysis complete")

    # === 4. Pushover setup ===
    # Define pushover parameters (from config)
    pushover_config = ANALYSIS["pushover"]
    max_drift = pushover_config["max_drift_ratio"]  # 0.05
    effective_bridge_height_m = pushover_config.get("effective_bridge_height_m", 13.05)
    default_increment_mm = 0.05 * effective_bridge_height_m * 1000.0 / 100.0
    displacement_increment_mm = pushover_config.get(
        "displacement_increment_mm", default_increment_mm
    )
    displacement_increment_min_mm = pushover_config.get(
        "displacement_increment_min_mm",
        min(displacement_increment_mm, 0.02),
    )
    displacement_increment_max_mm = pushover_config.get(
        "displacement_increment_max_mm",
        displacement_increment_mm,
    )
    displacement_increment_mm = min(
        displacement_increment_mm, displacement_increment_max_mm
    )
    displacement_control_num_iter = pushover_config.get(
        "displacement_control_num_iter", 20
    )
    ctrl_dof = pushover_config["control_dof"]  # 2 (Y-direction)

    # Pushover load pattern
    load_node = 5201  # Top node
    pattern_tag = 200
    weight_per_node = 976786.705  # From GravityLoad.py
    total_weight = 3 * weight_per_node  # 3 nodes
    load_factor = 0.1  # 10% of weight as lateral load

    op.timeSeries("Linear", 2)
    op.pattern("Plain", pattern_tag, 2)
    op.load(load_node, 0.0, load_factor, 0.0, 0.0, 0.0, 0.0)

    # === 5. Analysis setup ===
    output_dir = get_simulation_output_folder(scenario, scour_depth_mm)
    output_dir.mkdir(parents=True, exist_ok=True)

    define_recorders(folder=str(output_dir))
    define_displacement_recorders(folder=str(output_dir))

    op.wipeAnalysis()
    op.constraints("Transformation")
    op.numberer("RCM")
    op.system("BandGeneral")
    # Run pushover analysis
    target_displacement_mm = max_drift * effective_bridge_height_m * 1000.0
    estimated_steps = int(
        np.ceil(target_displacement_mm / displacement_increment_max_mm - 1.0e-9)
    )
    print(
        f"🏗️  Running pushover analysis (~{estimated_steps} steps minimum to "
        f"{max_drift * 100:.1f}% drift of effective bridge height, "
        f"{target_displacement_mm:.1f} mm target displacement, "
        f"adaptive step {displacement_increment_min_mm:.3f}-"
        f"{displacement_increment_max_mm:.3f} mm, "
        f"initial {displacement_increment_mm:.3f} mm)..."
    )
    print(
        f"🕐 Pushover analysis start: {time.strftime('%H:%M:%S', time.localtime(time.time()))}"
    )
    pushover_start = time.time()
    ok, completed_steps, completed_displacement_mm = _run_adaptive_pushover(
        load_node,
        ctrl_dof,
        target_displacement_mm,
        displacement_increment_mm,
        displacement_increment_min_mm,
        displacement_increment_max_mm,
        displacement_control_num_iter,
    )
    pushover_time = time.time() - pushover_start
    print(
        f"🕐 Pushover analysis end: {time.strftime('%H:%M:%S', time.localtime(time.time()))} - Duration: {pushover_time:.2f}s"
    )

    if ok != 0:
        print(
            "⚠️ Pushover analysis stopped before target "
            f"({completed_steps} converged steps, "
            f"{completed_displacement_mm:.3f}/{target_displacement_mm:.1f} mm); "
            "using available data"
        )
    else:
        print(
            "✅ Pushover target reached "
            f"({completed_steps} converged steps, "
            f"{completed_displacement_mm:.3f}/{target_displacement_mm:.1f} mm)"
        )

    _close_recorders()

    # === 6. Extract data ===
    disp_file = output_dir / "Displacement.5201.out"
    rotation_file = output_dir / "ColDisplacement.3201.out"
    force_files = [
        output_dir / "ColLocForce.3101.out",
        output_dir / "ColLocForce.3201.out",
        output_dir / "ColLocForce.3301.out",
    ]

    # Read displacement data
    try:
        disp_data = pd.read_csv(
            disp_file,
            sep=r"\s+",
            header=None,
            names=["time", "ux", "uy", "uz", "rx", "ry", "rz"],
        )
        # Recorder displacements are already in mm because the model uses N-mm-MPa units.
        displacement_mm = np.abs(disp_data["uy"].values)
    except Exception as e:
        print(f"❌ Failed to read displacement data: {e}")
        return None

    try:
        rotation_rad = _load_column_rotation(rotation_file)
    except Exception as e:
        print(f"❌ Failed to read column rotation data: {e}")
        return None

    # Read base shear and moment data (sum of column forces)
    try:
        force_responses = [
            _load_force_response(force_file) for force_file in force_files
        ]
    except Exception as e:
        print(f"❌ Failed to read force data: {e}")
        return None

    if not force_responses:
        print("❌ No force data available")
        return None

    force_lengths = [len(shear) for shear, _ in force_responses]
    common_force_len = min(force_lengths)
    if common_force_len == 0:
        print("❌ Force recorder files are empty")
        return None

    base_shear_kN = np.sum(
        [shear[:common_force_len] for shear, _ in force_responses], axis=0
    )
    base_moment_kNm = np.sum(
        [moment[:common_force_len] for _, moment in force_responses], axis=0
    )

    if base_shear_kN is None or base_moment_kNm is None:
        print("❌ No force data available")
        return None

    # Ensure arrays are same length
    min_len = min(
        len(displacement_mm),
        len(base_shear_kN),
        len(rotation_rad),
        len(base_moment_kNm),
    )
    displacement_mm = displacement_mm[:min_len]
    base_shear_kN = base_shear_kN[:min_len]
    rotation_rad = rotation_rad[:min_len]
    base_moment_kNm = base_moment_kNm[:min_len]

    # Remove duplicate displacements (if any)
    unique_indices = np.unique(displacement_mm, return_index=True)[1]
    displacement_mm = displacement_mm[unique_indices]
    base_shear_kN = base_shear_kN[unique_indices]
    rotation_rad = rotation_rad[unique_indices]
    base_moment_kNm = base_moment_kNm[unique_indices]

    # Filter out zero or negative displacements
    valid_idx = displacement_mm > 0
    displacement_mm = displacement_mm[valid_idx]
    base_shear_kN = base_shear_kN[valid_idx]
    rotation_rad = rotation_rad[valid_idx]
    base_moment_kNm = base_moment_kNm[valid_idx]

    if len(displacement_mm) < 10:
        print("❌ Insufficient data points for regression")
        return None

    # === 6. Post-processing ===
    print("📊 Processing results and fitting bilinear model...")
    post_start = time.time()

    # === 7. Bilinear regression ===
    try:
        k1, k2, dy_mm, vy_kN = fit_bilinear_profile(displacement_mm, base_shear_kN)

        if dy_mm is None or vy_kN is None:
            print("❌ Bilinear regression returned invalid yield parameters")
            return None

        i_yield = int(np.argmin(np.abs(displacement_mm - dy_mm)))
        my_kNm = float(base_moment_kNm[i_yield])
        thy_rad = float(rotation_rad[i_yield])

        capacity_point = (vy_kN, dy_mm, my_kNm, thy_rad)

        post_time = time.time() - post_start
        total_time = time.time() - start_time

        print(
            f"✅ Capacity point: Vy={vy_kN:.1f}kN, Dy={dy_mm:.1f}mm, My={my_kNm:.1f}kNm, Thy={thy_rad:.4f}rad"
        )
        print(f"📁 Recorder files saved to: {output_dir}")

        return (
            capacity_point,
            displacement_mm,
            base_shear_kN,
            rotation_rad,
            base_moment_kNm,
        )

    except Exception as e:
        print(f"❌ Bilinear regression failed: {e}")
        return None


# In[ ]:
