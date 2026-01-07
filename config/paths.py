"""
File path configuration for the project.
Centralizes all file and directory paths.
"""
from pathlib import Path
from .parameters import PATHS as PARAM_PATHS


def get_simulation_output_folder(scenario: str, scour_depth_mm: float) -> Path:
    """
    Get output folder for a specific simulation.

    Parameters
    ----------
    scenario : str
        Scenario name (e.g., 'missouri', 'colorado', 'extreme')
    scour_depth_mm : float
        Scour depth in millimeters

    Returns
    -------
    Path
        Path to simulation output folder
    """
    return PARAM_PATHS['recorder_data'] / scenario / f"scour_{scour_depth_mm:.1f}"


def get_latest_excel_file(pattern: str = "Scour_Materials_*.xlsx") -> Path:
    """
    Find the most recent Excel file matching a pattern.

    Parameters
    ----------
    pattern : str
        File pattern to match

    Returns
    -------
    Path
        Path to the most recent file
    """
    import glob
    files = list(PARAM_PATHS['recorder_data'].glob(pattern))
    if files:
        return max(files, key=lambda f: f.stat().st_mtime)
    return None


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        PARAM_PATHS['data_root'],
        PARAM_PATHS['geometry_data'],
        PARAM_PATHS['input_data'],
        PARAM_PATHS['output_data'],
        PARAM_PATHS['recorder_data'],
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
