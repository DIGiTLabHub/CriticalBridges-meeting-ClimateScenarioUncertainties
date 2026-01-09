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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from .RecorderColFiber import *
from .model_setup import build_model
from config.paths import get_latest_excel_file, get_simulation_output_folder
from config.parameters import ANALYSIS, MATERIALS, SCOUR
from src.postprocessing.bilinear_fit import fit_bilinear_profile


def run_single_pushover_simulation(scenario='missouri', random_seed=None):
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
    if random_seed is not None:
        np.random.seed(random_seed)

    # === 1. Sample parameters ===
    # Scour depth sampling (lognormal distribution based on scenario)
    scenario_params = SCOUR['scenarios'][scenario]
    vel = scenario_params['velocity_m_s']
    erosion_rate = scenario_params['erosion_rate_mm_hr']

    # Use simplified scour model - sample from lognormal with mean based on velocity
    # Higher velocity -> higher scour depth
    scour_mean_m = 0.5 + 0.1 * vel  # Rough approximation: Missouri~0.6m, Colorado~1.1m, Extreme~2.1m
    scour_std_m = 0.2
    scour_depth_m = np.random.lognormal(np.log(scour_mean_m), scour_std_m/scour_mean_m)

    # Material property sampling
    fc_mean = MATERIALS['concrete']['mean_MPa']
    fc_std = MATERIALS['concrete']['std_MPa']
    fc_MPa = np.random.normal(fc_mean, fc_std)

    fy_mean = MATERIALS['steel']['mean_MPa']
    fy_std = MATERIALS['steel']['std_MPa']
    fy_MPa = np.random.lognormal(np.log(fy_mean), fy_std/fy_mean)

    print(f"🔄 Sampled parameters: Scour={scour_depth_m:.3f}m, fc={fc_MPa:.1f}MPa, fy={fy_MPa:.1f}MPa")

    # === 2. Build model ===
    scour_depth_mm = scour_depth_m * 1000.0
    op.wipe()
    build_model(fc_MPa, fy_MPa, scour_depth_mm)

    # === 3. Gravity analysis ===
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

    # === 4. Pushover setup ===
    # Define pushover parameters (from config)
    pushover_config = ANALYSIS['pushover']
    max_drift = pushover_config['max_drift_ratio']  # 0.05
    ctrl_dof = pushover_config['control_dof']  # 2 (Y-direction)

    # Pushover load pattern
    load_node = 5201  # Top node
    pattern_tag = 200
    weight_per_node = 976786.705  # From GravityLoad.py
    total_weight = 3 * weight_per_node  # 3 nodes
    load_factor = 0.1  # 10% of weight as lateral load

    op.timeSeries('Linear', 2)
    op.pattern('Plain', pattern_tag, 2)
    op.load(load_node, 0.0, load_factor, 0.0, 0.0, 0.0, 0.0)

    # === 5. Analysis setup ===
    # Create temporary directory for recorders
    with tempfile.TemporaryDirectory() as temp_dir:
        define_recorders(folder=temp_dir)
        define_displacement_recorders(folder=temp_dir)

        op.wipeAnalysis()
        op.constraints('Transformation')
        op.numberer('RCM')
        op.system('BandGeneral')
        op.test('EnergyIncr', 1.0e-6, 100)
        op.algorithm('Newton')
        op.integrator('DisplacementControl', load_node, ctrl_dof, 1.0e-4)  # 0.1mm increments
        op.analysis('Static')

        # Run pushover analysis
        max_steps = int(max_drift * 1000 / 0.1)  # Steps for 5% drift with 0.1mm increments
        ok = op.analyze(max_steps)

        if ok != 0:
            print(f"⚠️ Pushover analysis stopped at step {ok}, using available data")

        # === 6. Extract data ===
        disp_file = os.path.join(temp_dir, 'Displacement.5201.out')
        force_files = [
            os.path.join(temp_dir, 'ColLocForce.3101.out'),
            os.path.join(temp_dir, 'ColLocForce.3201.out'),
            os.path.join(temp_dir, 'ColLocForce.3301.out')
        ]

        # Read displacement data
        try:
            disp_data = pd.read_csv(disp_file, sep=r'\s+', header=None,
                                  names=['time', 'ux', 'uy', 'uz', 'rx', 'ry', 'rz'])
            displacement_mm = np.abs(disp_data['uy'].values) * 1000.0  # Convert to mm
        except Exception as e:
            print(f"❌ Failed to read displacement data: {e}")
            return None

        # Read base shear data (sum of column forces)
        base_shear_kN = None
        for force_file in force_files:
            try:
                force_data = pd.read_csv(force_file, sep=r'\s+', header=None,
                                       names=['time', 'P', 'V2', 'V3', 'T', 'M2', 'M3'])
                Vi = np.abs(force_data['V2'].values) / 1000.0  # Convert to kN
                base_shear_kN = Vi if base_shear_kN is None else base_shear_kN + Vi
            except Exception as e:
                print(f"❌ Failed to read force data: {e}")
                return None

        if base_shear_kN is None:
            print("❌ No force data available")
            return None

        # Ensure arrays are same length
        min_len = min(len(displacement_mm), len(base_shear_kN))
        displacement_mm = displacement_mm[:min_len]
        base_shear_kN = base_shear_kN[:min_len]

        # Remove duplicate displacements (if any)
        unique_indices = np.unique(displacement_mm, return_index=True)[1]
        displacement_mm = displacement_mm[unique_indices]
        base_shear_kN = base_shear_kN[unique_indices]

        # Filter out zero or negative displacements
        valid_idx = displacement_mm > 0
        displacement_mm = displacement_mm[valid_idx]
        base_shear_kN = base_shear_kN[valid_idx]

        if len(displacement_mm) < 10:
            print("❌ Insufficient data points for regression")
            return None

        # === 7. Bilinear regression ===
        try:
            k1, k2, dy_mm, vy_kN = fit_bilinear_profile(displacement_mm, base_shear_kN)

            # Calculate additional capacity parameters
            # My = moment at yield (approximated from shear and lever arm)
            lever_arm_m = 13.0  # Approximate distance from base to center of mass
            my_kNm = vy_kN * lever_arm_m

            # Thy = rotation at yield (dy / lever_arm)
            thy_rad = dy_mm / 1000.0 / lever_arm_m  # Convert mm to m first

            capacity_point = (vy_kN, dy_mm, my_kNm, thy_rad)

            print(f"✅ Capacity point: Vy={vy_kN:.1f}kN, Dy={dy_mm:.1f}mm, My={my_kNm:.1f}kNm, Thy={thy_rad:.4f}rad")

            return capacity_point

        except Exception as e:
            print(f"❌ Bilinear regression failed: {e}")
            return None


# === Loop through each scenario ===
for label, sheet_name in scenario_sheets.items():
    print(f"\n🟦 Processing scenario: {label}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    for i, row in df.iterrows():
        scourDepth = row["Scour_Depth_mm"] / 1000.0
        scourDepthmm = round(row["Scour_Depth_mm"] + LCol, 1)
        fc = row["fc'_MPa"]
        fy = row["fy_MPa"]

        print(f"\n🔄 {label} | Sample {i+1}: Scour = {scourDepth:.3f} m | fc' = {fc:.2f} MPa | fy = {fy:.2f} MPa")

        # === 1. Build model ===
        op.wipe()
        build_model(fc, fy, scourDepthmm)

        # === 2. Gravity analysis ===
        op.constraints("Transformation")
        op.numberer("RCM")
        op.system("BandGeneral")
        op.algorithm("Newton")
        op.test("NormDispIncr", 1.0e-6, 1000)
        op.integrator("LoadControl", 1.0)
        op.analysis("Static")

        result = op.analyze(1)
        if result != 0:
            print(f"❌ Gravity failed for {label} sample {i+1}")
            continue
        op.reactions()
        op.loadConst("-time", 0.0)

        # === 3. Lateral load ===
        op.timeSeries('Linear', 2)
        op.pattern('Plain', patternTag, 2)
        op.load(loadNodeTag, *load_vector)

        # === 4. Recorders ===
        depth = round(row["Scour_Depth_mm"], 1)
        folder = f"RecorderData/{label}/scour_{depth:.1f}"
        define_recorders(folder=folder)
        define_displacement_recorders(folder=folder)

        # === 5. Analysis setup ===
        op.wipeAnalysis()
        op.constraints('Transformation')
        op.numberer('RCM')
        op.system('BandGeneral')
        op.test('EnergyIncr', tol, maxNumIter)
        op.algorithm('Newton')
        op.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)
        op.analysis('Static')

        ok = op.analyze(Nsteps)
        print(f"Initial result: {ok}")

        # === 6. Fallback if failed ===
        if ok != 0:
            test_dict = {
                1: 'NormDispIncr',
                2: 'RelativeEnergyIncr',
                4: 'RelativeNormUnbalance',
                5: 'RelativeNormDispIncr',
                6: 'NormUnbalance'
            }
            algo_dict = {
                1: 'KrylovNewton',
                2: 'SecantNewton',
                4: 'RaphsonNewton',
                5: 'PeriodicNewton',
                6: 'BFGS',
                7: 'Broyden',
                8: 'NewtonLineSearch'
            }

            for test_type in test_dict.values():
                for algo_type in algo_dict.values():
                    if ok != 0:
                        if algo_type in ['KrylovNewton', 'SecantNewton']:
                            op.algorithm(algo_type, '-initial')
                        else:
                            op.algorithm(algo_type)
                        op.test(test_type, tol, 1000)
                        ok = op.analyze(Nsteps)
                        print(f"Trying {test_type} + {algo_type} → Result: {ok}")
                        if ok == 0:
                            break
                if ok == 0:
                    break

        # === 7. Final results ===
        try:
            u = op.nodeDisp(IDctrlNode, IDctrlDOF)
            print(f"✅ Final uy @ Node {IDctrlNode}: u = {u:.6f} m")
        except:
            print("❌ Displacement retrieval failed.")

        print("--------------------------------------------------")

print("✅ All scenarios processed.")


# In[ ]:




