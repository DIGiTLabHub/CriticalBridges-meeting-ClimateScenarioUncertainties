#!/usr/bin/env python
# coding: utf-8

"""
Model setup - Build complete OpenSeesPy bridge model.
Refactored to use JSON data files instead of hardcoded Python lists.
"""

import openseespy.opensees as ops
from .os_model_functions import *
from .geometry.geometry_loader import GeometryLoader
from .GeoTrans import defineGeoTrans
from .SectionMat import defineSectionMaterials
from .ColSection import defineColSection
from .Element import define_elements
from .ZeroLengthElement import defineZeroLengthElement
from .GravityLoad import defineLoads


def build_model(fc: float, fy: float, scourDepth: float = 0):
    """
    Build complete OpenSeesPy bridge model with specified materials.

    Parameters
    ----------
    fc : float
        Concrete compressive strength (MPa)
    fy : float
        Steel yield strength (MPa)
    scourDepth : float, optional
        Scour depth (mm), default=0

    Returns
    -------
    None
        Creates OpenSees model in memory
    """
    # Initialize geometry loader
    geo_loader = GeometryLoader()

    # Load geometry data from JSON files
    nodes = geo_loader.load_nodes()
    restraints = geo_loader.load_restraints()
    constraints = geo_loader.load_constraints()
    masses = geo_loader.load_masses()

    # Build model
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    define_nodes(nodes)
    apply_restraints(restraints)
    apply_constraints(constraints)
    apply_masses(masses)
    defineGeoTrans()
    defineSectionMaterials(fc, fy)
    defineColSection()
    define_elements()
    defineZeroLengthElement(scourDepth)
    defineLoads()

    print(f"✅ Model built: fc={fc:.2f} MPa, fy={fy:.2f} MPa, scourDepth={scourDepth:.1f} mm")



