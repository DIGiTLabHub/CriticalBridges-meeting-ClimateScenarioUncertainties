"""
Central Parameter File: params.py
For Surrogate Learning of Scoured Bridges
"""

# ============================================
# 1. GEOMETRY & STRUCTURAL DIMENSIONS
# ============================================
GEOMETRY = {
    "span_length_m": 35,
    "deck_width_m": 12,
    "num_spans": 4,
    # Backward-compatible name retained for existing scripts. This value is
    # the effective bridge height from the unscoured riverbed to deck level,
    # not the clear height of the column between cap and deck.
    "column_height_m": 13.05,
    "pier_diameter_m": 1.5,
}

# ============================================
# 2. MATERIAL PROPERTIES
# ============================================
MATERIALS = {
    "concrete": {"mean_MPa": 27.0, "std_MPa": 1.0, "distribution": "normal"},
    "steel": {"mean_MPa": 420.0, "std_MPa": 4.2, "distribution": "lognormal"},
}

# ============================================
# 3. SCOUR HAZARD MODELING
# ============================================
SCOUR = {
    "kinematic_viscosity_m2_s": 1.0e-6,
    "density_water_kg_m3": 1000,
    "years_of_exposure": 50,
    "num_lhs_samples_per_scenario": 1000,
    # Discrete realized scour depths s_z (m) at which the FE soil-spring
    # removal state changes. The grid follows the modeled p-y/t-z spring
    # elevations: 0.5 m spacing to 10 m, then 1.0 m spacing to the deepest
    # spring row at 19 m. The 20 m endpoint represents physical scour to the
    # pile tip; it has the same modeled spring-removal state as 19 m.
    "spring_removal_depths_m": [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
    ],
    "scenarios": {
        "missouri": {
            "velocity_m_s": 2.9,
            "erosion_rate_mm_hr": 100,
            "description": "Missouri River",
        },
        "colorado": {
            "velocity_m_s": 6.5,
            "erosion_rate_mm_hr": 500,
            "description": "Colorado River",
        },
        "extreme": {
            "velocity_m_s": 10.0,
            "erosion_rate_mm_hr": 1000,
            "description": "Extreme Case",
        },
    },
}

# ============================================
# 4. OPENSEES ANALYSIS PARAMETERS
# ============================================
ANALYSIS = {
    "gravity": {
        "tolerance": 1.0e-8,
        "max_iterations": 3000,
        "load_increment": 0.1,
        "num_increments": 10,
    },
    "pushover": {
        "max_drift_ratio": 0.05,
        # Effective bridge height from the unscoured riverbed (z = -13.05 m)
        # to the deck level (z = 0 m), used only to set the pushover drift
        # control target.
        "effective_bridge_height_m": 13.05,
        # Adaptive DisplacementControl settings in model units (mm). The
        # previous 6.525 mm nominal step was backsolved for 100 steps at 5%
        # drift, but that was too coarse for forceBeamColumn convergence.
        "displacement_increment_mm": 1.0,
        "displacement_increment_min_mm": 0.02,
        "displacement_increment_max_mm": 1.0,
        "displacement_control_num_iter": 20,
        "control_dof": 2,
        "tolerance": 1.0e-6,
        "max_iterations": 100,
    },
    "dynamic": {"tolerance": 1.0e-8, "max_iterations": 10},
}

# ============================================
# 5. SURROGATE MODELING
# ============================================
SURROGATE = {
    "features": ["Sz", "Sz_sq", "Sz_cu", "log_Sz", "inv_Sz", "sqrt_Sz"],
    "targets": ["Vy", "Dy", "My", "Thy"],
    "gbr": {
        "n_estimators": 700,
        "max_depth": 3,
        "learning_rate": 0.015,
        "subsample": 0.85,
    },
    "svr": {"kernel": "rbf", "C": 100, "epsilon": 0.01, "gamma": 0.1},
    "bootstrap_ensemble_size": 30,
}

# ============================================
# 6. FILE PATHS
# ============================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GEOMETRY_DIR = DATA_DIR / "geometry"
INPUT_DATA_DIR = DATA_DIR / "input"
OUTPUT_DATA_DIR = DATA_DIR / "output"
RECORDER_DATA_DIR = PROJECT_ROOT / "RecorderData"

PATHS = {
    "project_root": PROJECT_ROOT,
    "data_root": DATA_DIR,
    "geometry_data": GEOMETRY_DIR,
    "input_data": INPUT_DATA_DIR,
    "output_data": OUTPUT_DATA_DIR,
    "recorder_data": RECORDER_DATA_DIR,
    "nodes_file": GEOMETRY_DIR / "nodes.json",
    "elements_file": GEOMETRY_DIR / "elements.json",
    "restraints_file": GEOMETRY_DIR / "restraints.json",
    "constraints_file": GEOMETRY_DIR / "constraints.json",
    "masses_file": GEOMETRY_DIR / "masses.json",
}

# ============================================
# 7. OUTPUT VARIABLES
# ============================================
OUTPUT_VARIABLES = {
    "Vy": "yield_base_shear_kN",
    "Dy": "yield_displacement_mm",
    "My": "yield_moment_kNm",
    "Thy": "yield_rotation_rad",
}
