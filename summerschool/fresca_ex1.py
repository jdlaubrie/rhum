# SETTING UP FEniCS AND gmsh
import sys
sys.path.append("/usr/lib/python3/dist-packages/")
import gmsh
sys.path.append("/usr/lib/petsc/lib/python3/dist-packages/")
import dolfinx

# IMPORTING DL-ROM LIBRARIES
import torch # Pytorch tensors, autodiff...
from cores import CPU, GPU # For swapping cores
import fespaces as fe # Handles finite element spaces
import geometry as geo # Mesh creation
import matplotlib.pyplot as plt # Plots

# Mesh generation
domain = geo.Circle((0,0), 1) - geo.Circle((0,0), 0.5)
mesh = geo.mesh(domain, stepsize = 0.1)
fe.plot(mesh)
plt.title("High fidelity mesh ($N_{h}$=%d)" % mesh.num_vertices())
plt.show()