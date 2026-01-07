"""
Scour-Critical Bridge Simulator
"""

__version__ = "0.2.0"
__author__ = "Project Team"

# Main package imports
from .bridge_modeling import build_model
from .scour import LHS_scour_hazard

__all__ = [
    '__version__',
    '__author__',
    'build_model',
    'LHS_scour_hazard'
]
