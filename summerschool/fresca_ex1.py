# SETTING UP FEniCS AND gmsh
import gmsh
import dolfin

import sys
sys.path.append("/home/joan.laubriesoto/Coding/fresca_libs/")

# IMPORTING DL-ROM LIBRARIES
import torch # Pytorch tensors, autodiff...
from cores import CPU, GPU # For swapping cores
import fespaces as fe # Handles finite element spaces
import geometry as geo # Mesh creation
import matplotlib.pyplot as plt # Plots

#========================================================================================#
# Full order model (FOM solver)
# Mesh generation
domain = geo.Circle((0,0), 1) - geo.Circle((0,0), 0.5)
mesh = geo.mesh(domain, stepsize = 0.1)
fe.plot(mesh)
plt.title("High fidelity mesh ($N_{h}$=%d)" % mesh.num_vertices())
plt.show()

# Ground truth model
space = fe.space(mesh, 'CG', 1) # piecewise linear polynomials
x, y = fe.coordinates(space).T

x, y = CPU.tensor(x), CPU.tensor(y)
r, theta = (x**2 + y**2).sqrt(), torch.atan2(x, y)

# eps -> diffusion, omega -> advection
def solution(t, eps, omega):
  T = CPU.tensor([t])
  d2 = (torch.cos(theta) - torch.cos(omega*T))**2 + (torch.sin(theta) - torch.sin(omega*T))**2
  return (r-0.5)*(1.0-r)*torch.exp(- d2 /(2*eps*T) )

u = solution(t = 0.5, eps = 1, omega = 1)
fe.plot(u, space, colorbar = True)
plt.title("Solution field")
plt.show()

# Snapshots generation
eps = [1, 5] #[1, 5] #[0.1, 0.2]
omega = [1, 2] #[1, 2] #[2, 5]
time = [0.05, 2]
ntraject = 40
timesteps = 50
nh = space.dim()
U = CPU.zeros(ntraject, timesteps, nh)
mu = CPU.zeros(ntraject, timesteps, 3)

dt = (time[1]-time[0])/timesteps

for i in range(ntraject):
  e0, o0 = CPU.rand(), CPU.rand()
  e = (eps[1]-eps[0])*e0+eps[0]
  o = (omega[1]-omega[0])*o0+omega[0]
  for j in range(timesteps):
    t = time[0]+j*dt
    U[i,j] = solution(t, e, o)

    # normalized parameters
    mu[i,j,0] = j/timesteps
    mu[i,j,1] = e0
    mu[i,j,2] = o0

fe.gif("example", U[0], dt=dt, T=time[1] - time[0], space=space)

#========================================================================================#
# Reduced order model - the POD-NN approach
# a) Proper Orthogonal Decomposition

# Splitting of the data: we use 50% of the data to design the ROM, and 50% to test its performances
ntrain = 20
ntrain *= timesteps
u = U.reshape(-1, nh) # reshaping so that we list all snapshots one by one

from roms import POD

pod, eigs = POD(u[:ntrain], k = 10) # POD basis vectors and corresponding singular values
# NB: "pod" is a nmodes x nh tensor, that is, basis[k] contains the kth POD mode

plt.figure(figsize = (4,3))
plt.plot(eigs, '-.', color = 'red')
plt.title("Singular values decay", fontsize = 10)
plt.xlabel("# modes")
plt.show()

# Let's visualize some of the basis functions
plt.figure(figsize = (10, 2.5))
for j in range(2):
  plt.subplot(1,2,j+1)
  fe.plot(pod[j], space)
  plt.title("Mode n.%d" % (j+1), fontsize = 10)
plt.tight_layout()
plt.show()

# How well does the POD basis represent solutions?
from minns import L2
from roms import num2p
from roms import project

l2 = L2(space)
def mre(utrue, upred):
  return ( l2(utrue-upred)/l2(utrue) ).mean()

urecon = project(pod, u)
print("Mean relative error (training): %s." % num2p(mre(u[ntrain:], urecon[ntrain:])))

which = -1
Urecon = urecon.reshape(ntraject, timesteps, nh)
fe.gifz("dimred", (U[which], Urecon[which]), dt = dt, T = time[1]-time[0], space = space, titles = ["Ground truth", "POD approximation"])

#-------------------------------------------------------#
# b) Learning the POD coordinates
# Projection onto POD coordinates
from roms import projectdown, projectup

c = projectdown(pod, u).squeeze(-1) # torch.Size([2000, 10, 1])
nbasis = c.shape[-1]

# Data reshaping
nparams = 3
mu = mu.reshape(-1, nparams)

from dnns import Dense, train

# Neural network design and initialization
phi = Dense(nparams, 50) + Dense(50, 50) + Dense(50, nbasis, activation = None)
phi.He()

def mse(ctrue, cpred): # mean of the squared Euclidean norm
  return (ctrue-cpred).pow(2).sum(axis = -1).mean()

def error(ctrue, cpred):
  return (ctrue-cpred).pow(2).sum(axis = -1).sqrt().mean()

# Training phase (try with epochs = 200)
train(phi, mu, c, ntrain = ntrain, epochs = 50, optim = torch.optim.LBFGS, lossf = mse, error = error)

def count_trainable_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Trainable parameters:", count_trainable_params(phi))

#-------------------------------------------------------#
# c) Assembling the ROM

phi.freeze()
urom = projectup(pod, phi(mu))

# How good is it?
print("ROM error: %s." % num2p(mre(u[ntrain:], urom[ntrain:])))

# Visual comparison
Urom = urom.reshape(ntraject, timesteps, nh)
which = -1
fe.gifz("podnn", (U[which], Urom[which]), dt = dt, T = time[1]-time[0], space = space, titles = ["Ground truth", "POD-NN approximation"])

#========================================================================================#
# Reduced order model - the DL-ROM approach
# a) Autoencoder

encoder = Dense(nh, 2*nparams)
decoder = Dense(2*nparams, 100) + Dense(100, nh)

ae = encoder+decoder
ae.He()

def error(utrue, upred):
  return mre(utrue, upred)

# Training phase (try with epochs = 300)
train(ae, u, u, ntrain = ntrain, epochs = 50, optim = torch.optim.LBFGS, lossf = mre, error = error, conv = num2p)

#-------------------------------------------------------#
# b) Change of coordinates
from dnns import Weightless
ae.freeze()
z = encoder(u) # AE latent variables
nparams = 3
nlatent = z.shape[-1]
mu = mu.reshape(-1, nparams)

class Embedd(Weightless):
  def forward(self, x):
    return torch.cat([x, x.sin(), x.cos(), x[:,[0]]*x[:,[1]], x[:,[1]]*x[:,[2]], x[:,[2]]*x[:,[0]]], axis = 1)

phi = Embedd() + Dense(12, 50) + Dense(50, 50) + Dense(50, 50) + Dense(50, nlatent)
phi.He()

def mse(ztrue, zpred):
  return (ztrue-zpred).pow(2).sum(axis = -1).mean()

def error(ztrue, zpred):
  return (ztrue-zpred).pow(2).sum(axis = -1).sqrt().mean()

# Training phase (try with epochs = 500)
train(phi, mu, z, ntrain = ntrain, epochs = 500, optim = torch.optim.LBFGS, lossf = mse, error = error, conv = num2p)

#-------------------------------------------------------#
# c) Assembling the ROM
phi.freeze()
udlrom = decoder(phi(mu))
print("ROM error: %s." % num2p(mre(u[ntrain:], udlrom[ntrain:])))

Udlrom = udlrom.reshape(ntraject, timesteps, nh)
fe.gifz("dlrom", (U[which], Udlrom[which]), dt = dt, T = time[1]-time[0], space = space, titles = ["Ground truth", "DL-ROM approximation"])

#========================================================================================#
# Application to digital twins
from IPython.display import Image
Image("DT.png")

locs = [25*j for j in range(14)]
fe.plot(mesh)
plt.plot(*fe.coordinates(space)[locs].T,'.r', markersize = 10)
plt.title("Sensors location")
plt.show()

def ROM(*args):
  lengths = []
  for arg in args:
    try:
      lengths.append(len(arg))
    except:
      lengths.append(1)
  n = max(lengths)
  params = CPU.zeros(n, 3)
  for i in range(len(args)):
    params[:,i] = params[:,i] + args[i]
  return decoder(phi(params))

from numpy import linspace, meshgrid
epsilons, omegas = meshgrid(linspace(0.0, 1.0, 30), linspace(0.0, 1.0, 30))
epsilons = CPU.tensor(epsilons.reshape(-1))
omegas = CPU.tensor(omegas.reshape(-1))
dose = 0.5

def variance(v):
  u = v.unsqueeze(0) if len(v.shape)==1 else v
  return l2(u-u.mean(axis = -1).unsqueeze(-1)).pow(2)

def feedback(sensorlocs, sensorvals, t):
  solutions = ROM(t, epsilons, omegas)
  diff = (solutions[:, sensorlocs] - sensorvals).pow(2).sum(axis = -1)
  candidate = diff.argmin()
  twin = solutions[candidate]
  return variance(twin) < 0.0003 or epsilons[candidate] < dose

t, e, o = 0.0, 0.5, 0.5

def state(t, e, o):
  return solution((time[1]-time[0])*t+time[0], (eps[1]-eps[0])*e+eps[0], (omega[1]-omega[0])*o+omega[0])

steps = 70
dt = 1.0/200.0
objective = []
for k in range(steps):
  s = state(t,e,o)
  objective.append(variance(s).item())
  vals = s[locs]
  if(not feedback(locs, vals, t)):
    e = e - dose
  t += dt

plt.plot(objective, '-b')
plt.plot([0, steps], [0.0003, 0.0003], '--r')
vv = variance(u)
plt.axis([0, steps, vv.min().item(), vv.max().item()])
plt.title("Objective function trajectory")
plt.show()