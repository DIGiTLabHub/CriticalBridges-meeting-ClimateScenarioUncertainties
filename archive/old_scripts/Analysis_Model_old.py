#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openseespy.opensees import *
from Node import define_nodes  # Import node definition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.spatial import KDTree

model('basic', '-ndm', 3, '-ndf', 6)

# Define nodes
define_nodes()

# Function to extract node coordinates
def get_node_coordinates():
    node_tags = getNodeTags()  # Get all node IDs
    coordinates = {tag: tuple(nodeCoord(tag)) for tag in node_tags}  # Store as dictionary
    return coordinates

# Get node coordinates
coordinates = get_node_coordinates()

# Convert to array for easy processing
node_list = np.array(list(coordinates.values()))
node_tags = list(coordinates.keys())

# Build KDTree for quick nearest neighbor searches
tree = KDTree(node_list)

# Function to find connections automatically
def find_connections():
    connections = set()
    tolerance = 50  # Adjust this threshold based on expected node spacing

    for i, node1 in enumerate(node_tags):
        coord1 = coordinates[node1]
        # Find nearest nodes within tolerance
        nearby_indices = tree.query_ball_point(coord1, r=tolerance)

        for j in nearby_indices:
            node2 = node_tags[j]
            coord2 = coordinates[node2]

            # Check if it's a valid connection (avoid self-connections)
            if node1 != node2:
                # Ensure unique connections (sorted to prevent duplicates)
                connections.add(tuple(sorted((node1, node2))))
    
    return list(connections)

# Get automatically detected connections
connections = find_connections()

# Extract bridge dimensions
x_vals, y_vals, z_vals = zip(*coordinates.values())
bridge_length = max(x_vals) - min(x_vals)
bridge_width = max(y_vals) - min(y_vals)
bridge_height = max(z_vals) - min(z_vals)

# Function to plot the truss bridge
def plot_truss():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes as points
    ax.scatter(x_vals, y_vals, z_vals, c='blue', marker='o', s=10)

    # Plot members as lines
    segments = [(coordinates[n1], coordinates[n2]) for n1, n2 in connections]
    line_collection = Line3DCollection(segments, colors='black', linewidths=1)
    ax.add_collection3d(line_collection)

    # Labels
    ax.set_xlabel('X (Length)')
    ax.set_ylabel('Y (Width)')
    ax.set_zlabel('Z (Height)')
    ax.set_title('3D Truss Bridge Structure')

    plt.show()

# Plot the truss bridge
plot_truss()

# Text description of the bridge
bridge_description = f"""
The truss bridge consists of a series of connected nodes forming a structural system.
- **Length:** {bridge_length:.2f} units
- **Width:** {bridge_width:.2f} units
- **Height:** {bridge_height:.2f} units
- **Node count:** {len(coordinates)}
- **Element count:** {len(connections)}

The structure primarily includes:
- Automatically detected **top and bottom chords**.
- **Vertical members** connecting different levels.
- **Diagonal members** for load distribution.

This truss bridge design ensures **load distribution efficiency** and **structural stability** for spanning a given distance.
"""

# Print description
print(bridge_description)


# In[2]:


import Parameters  # Define parameters


# In[3]:


import openseespy.opensees as ops
from Node import define_nodes
from Restraint import restraints  # Import the restraints list from Restraint.py

ops.model('basic', '-ndm', 3, '-ndf', 6)

# Apply restraints inside the main script
for node_id, dx, dy, dz, rx, ry, rz in restraints:
    ops.fix(node_id, dx, dy, dz, rx, ry, rz)  # Apply fixity
    print(f"Applied fixity to node {node_id}: [{dx}, {dy}, {dz}, {rx}, {ry}, {rz}]")


# In[4]:


import openseespy.opensees as ops
from Constraint import apply_constraints, constraints  # Import constraints

# Apply constraints
apply_constraints()

# Verify constraints
print("\n🛠 **Constraints Verification**")
for retained, constrained, dof in constraints:
    print(f"✔ Node {constrained} constrained to Node {retained} on DOFs: {dof}")


# In[5]:


import Mass


# In[6]:


import openseespy.opensees as ops
from Node import define_nodes
from Restraint import restraints
from Constraint import apply_constraints, constraints
from Mass import apply_masses, masses  # ✅ Import mass function and list

ops.model('basic', '-ndm', 3, '-ndf', 6)

# Apply masses
print("\n🛠 Applying Masses...")
applied_masses = apply_masses()  # ✅ Store returned masses for verification
print("✅ Masses successfully applied!\n")

# Verify applied masses in OpenSees
print("\n🛠 **Mass Verification**")
for nodeTag, _, _, _, _, _, _ in applied_masses:
    applied_mass = ops.nodeMass(nodeTag)  # Get mass from OpenSees
    print(f"✔ Node {nodeTag} - OpenSees Mass: {applied_mass}")


# In[7]:


import GeoTrans  # Define geometric transformations
# Now call the function that sets up your uniaxial materials
GeoTrans.defineGeoTrans()


# In[8]:


import openseespy.opensees as ops
import SectionMat  # Import the Python file containing defineSectionMaterials()

# Start defining your model
ops.model('basic', '-ndm', 3, '-ndf', 6)

# Now call the function that sets up your uniaxial materials
SectionMat.defineSectionMaterials()

# ------------
# Continue with the rest of your model-building commands, for example:
# ops.node(1, 0.0, 0.0, 0.0)
# ops.node(2, 5.0, 0.0, 0.0)
# ops.element('truss', ...)
# ...
# ------------


# In[9]:


import openseespy.opensees as ops


# Now import ColSection
import ColSection

# Start a new model
ops.model('basic', '-ndm', 3, '-ndf', 6)

# Define the column section (fibers, etc.)
ColSection.defineColSection()

# After this, your section with ID=1 is ready to be used.
# Next, continue building your nodes, elements, constraints, loads, analysis, etc.

# Example usage:
# ops.node(1, 0.0, 0.0, 0.0)
# ops.node(2, 0.0, 0.0, 3.0)
# ...
# ops.element('forceBeamColumn', 1, 1, 2, numIntgrPts, 1, ...)
# ...


# In[10]:


import openseespy.opensees as ops
import Element  # This just *loads* the module, doesn't run define_elements()

ops.model('Basic', '-ndm', 3, '-ndf', 6)

# Now call the function to create the elements:
Element.define_elements()

# Because define_elements() is now called, 
# you should see the print statements, e.g.:
#
#   ✅ <number> elements have been successfully defined in OpenSees.
#   Element defined.

# Continue your analysis or other definitions...


# In[11]:


import ZeroLengthElement
ZeroLengthElement.defineZeroLengthElement()  # Same exact capitalization


# In[12]:


import Load

# now define loads
Load.defineLoads()


# In[13]:


import openseespy.opensees as ops

# -- Step 1: Set up the analysis --
ops.constraints("Transformation")       # how boundary conditions are enforced
ops.numberer("RCM")                    # renumber dof's to minimize bandwidth
ops.system("BandGeneral")              # solver for the system of equations
ops.algorithm("Newton")                # use Newton's method
ops.test("NormDispIncr", 1.0e-6, 1000) # convergence test (tolerance, maxIter)
ops.integrator("LoadControl", 1.0)     # apply the loads in a single step
ops.analysis("Static")                 # define the analysis type

# -- Step 2: Run the gravity analysis --
result = ops.analyze(1)
if result != 0:
    print("Gravity analysis failed to converge.")
else:
    print("Gravity analysis completed successfully.")

# -- Step 3: Lock in the deformed shape under gravity --
ops.loadConst("-time", 0.0)

all_nodes = ops.getNodeTags()
for n in all_nodes:
    disp = ops.nodeDisp(n)
    print(f"Node {n} disp = {disp}")


# In[14]:


n=101

disp = ops.nodeDisp(n)  # [ux, uy, uz, rx, ry, rz] in 3D
print(f"Node {n} displacements: {disp}")

forces_local = ops.eleResponse(n, "force")  
print(f"Element {n} local-end-forces: {forces_local}")

rxn = ops.nodeReaction(n)  # [Rx, Ry, Rz, Mx, My, Mz]
print(f"Reaction at node {n}: {rxn}")


# In[15]:


ops.printModel()  


# In[16]:


ops.printModel("-JSON", "modelOutput.json")

