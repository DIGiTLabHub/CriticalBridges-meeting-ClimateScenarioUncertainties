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
    "column_height_m": 13.05,
    "pier_diameter_m": 1.5,
}

# ============================================
# 2. MATERIAL PROPERTIES
# ============================================
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

# ============================================
# 3. SCOUR HAZARD MODELING
# ============================================
SCOUR = {
    "kinematic_viscosity_m2_s": 1.0e-6,
    "density_water_kg_m3": 1000,
    "years_of_exposure": 50,
    "num_lhs_samples_per_scenario": 1000,
    "scenarios": {
        "missouri": {
            "velocity_m_s": 2.9,
            "erosion_rate_mm_hr": 100,
            "description": "Missouri River"
        },
        "colorado": {
            "velocity_m_s": 6.5,
            "erosion_rate_mm_hr": 500,
            "description": "Colorado River"
        },
        "extreme": {
            "velocity_m_s": 10.0,
            "erosion_rate_mm_hr": 1000,
            "description": "Extreme Case"
        }
    }
}

# ============================================
# 4. OPENSEES ANALYSIS PARAMETERS
# ============================================
ANALYSIS = {
    "gravity": {
        "tolerance": 1.0e-8,
        "max_iterations": 3000,
        "load_increment": 0.1,
        "num_increments": 10
    },
    "pushover": {
        "max_drift_ratio": 0.05,
        "control_dof": 2,
        "tolerance": 1.0e-6,
        "max_iterations": 100
    },
    "dynamic": {
        "tolerance": 1.0e-8,
        "max_iterations": 10
    }
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
        "subsample": 0.85
    },
    "svr": {
        "kernel": "rbf",
        "C": 100,
        "epsilon": 0.01,
        "gamma": 0.1
    },
    "bootstrap_ensemble_size": 30
}

# ============================================
# 6. FILE PATHS
# ============================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
GEOMETRY_DIR = DATA_DIR / "geometry"
INPUT_DATA_DIR = DATA_DIR / "input"
OUTPUT_DATA_DIR = DATA_DIR / "output"
RECORDER_DATA_DIR = Path("RecorderData")  # Keep for backward compatibility

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
    "masses_file": GEOMETRY_DIR / "masses.json"
}

# ============================================
# 7. OUTPUT VARIABLES
# ============================================
OUTPUT_VARIABLES = {
    "Vy": "yield_base_shear_kN",
    "Dy": "yield_displacement_mm",
    "My": "yield_moment_kNm",
    "Thy": "yield_rotation_rad"
}
