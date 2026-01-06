#!/usr/bin/env python
# coding: utf-8

# # Opensees-Py based bridge simulation - 

# ## Building the model including nodes, restraints/constraints, materials, sections, and elements.

# ### Import libraries

# In[1]:


# from openseespy.opensees import *
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np
from   os_model_functions import *

import opsvis as opsv # for visualization; need to install it first: https://opsvis.readthedocs.io/en/latest/index.html


# ### building the model now

# In[2]:


ops.wipe() # usually needed to wipe out any existing models when runing the notebook again.
ops.model('basic', '-ndm', 3, '-ndf', 6) # do it only one time at the begining for the modeling


# In[3]:


import Parameters  # Define constant parameters for the simulation


# In[4]:


# Import nodes and apply restraints
from Node import nodes
# len(nodes)
define_nodes(nodes)


# In[5]:


from Restraint import restraints  # Import the restraints list from Restraint.py
apply_restraints(restraints)


# In[6]:


# import openseespy.opensees as ops
from Constraint import constraints  # Import constraints
# Apply constraints
apply_constraints(constraints)


# In[7]:


from Mass import masses  # ✅ Import mass function and list
apply_masses(masses)


# In[8]:


from GeoTrans import defineGeoTrans # Define geometric transformations
# Now call the function that sets up your uniaxial materials
defineGeoTrans()


# In[9]:


from SectionMat import defineSectionMaterials # Import the Python file containing defineSectionMaterials()
# Now call the function that sets up your uniaxial materials
defineSectionMaterials()


# In[10]:


# Now import ColSection
from ColSection import defineColSection
# Define the column section (fibers, etc.)
defineColSection()


# In[11]:


from Element import  define_elements # This just *loads* the module, doesn't run define_elements()
# Now call the function to create the elements:
define_elements()


# In[12]:


from ZeroLengthElement import defineZeroLengthElement
defineZeroLengthElement()  # Same exact capitalization


# ## Perform eigenvalue analysis

# In[13]:


n_modes = 9
frequencies = eigen_analysis(n_modes)
# Output the first n_modes frequencies
for i, freq in enumerate(frequencies):
    print(f"Mode {i+1}: {freq:.4f} Hz")


# ## Perform gravity analysis

# In[14]:


from GravityLoad import  defineLoads # gravity loads
defineLoads()


# In[15]:


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


# In[16]:


n = 101

disp = ops.nodeDisp(n)  # [ux, uy, uz, rx, ry, rz] in 3D
print(f"Node {n} displacements: {disp}")

forces_local = ops.eleResponse(n, "force")  
print(f"Element {n} local-end-forces: {forces_local}")

rxn = ops.nodeReaction(n)  # [Rx, Ry, Rz, Mx, My, Mz]
print(f"Reaction at node {n}: {rxn}")


# In[23]:


# ops.printModel()  


# In[25]:


ops.printModel("-JSON", "modelOutput.json")


# ## Model Visualization

# In[19]:


opsv.plot_model()


# In[27]:


sfac = 2.0e1
fig_wi_he = 30., 20
model_no = 1
opsv.plot_mode_shape(model_no, sfac, 19, az_el=(106., 46.),
                     fig_wi_he=fig_wi_he)
plt.title(f'Mode {model_no}')


# In[1]:




