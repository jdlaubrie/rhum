import os
from pathlib import Path

# SETTING UP FEniCS
import dolfin

# IMPORTING DL-ROM LIBRARIES
import numpy as np
import torch # Pytorch tensors, autodiff...
import matplotlib.pyplot as plt # Plots

#========================================================================================#
#=====================    GEOMETRY     ==============================#
#========================================================================================#
from IPython.display import clear_output

class Domain(object):
    def __init__(self, main, other, operation=None):
        """Combines two domains via the specified operation."""
        self.a, self.b, self.op = main, other, operation
        self.index = 0
        self.dim = max(main.dim, other.dim)

    def script(self, index=1):
        """Writes a gmsh script describing the domain."""
        res, j = self.a.script(index)
        res0, j0 = self.b.script(j)
        self.index = j0
        if (self.op == "u"):
            res0 += "BooleanUnion{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index,
                                                        self.b.entity(), self.b.index)
        elif (self.op == "i"):
            res0 += "BooleanIntersection{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index,
                                                               self.b.entity(), self.b.index)
        elif (self.op == "d"):
            res0 += "BooleanDifference{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index,
                                                             self.b.entity(), self.b.index)
        return res + res0, j0 + 1

    def __add__(self, other):
        return Domain(self, other, "u")

    def __sub__(self, other):
        return Domain(self, other, "d")

    def __mul__(self, other):
        return Domain(self, other, "i")

    def entity(self):
        if self.dim == 2:
            return "Surface"
        elif self.dim == 3:
            return "Volume"

class Circle(Domain):
    def __init__(self, p, r=1):
        self.p = p
        self.r = r
        self.index = 0
        self.dim = 2

    def script(self, index=1):
        self.index = index
        return 'Disk(%d) = {%f, %f, 0.0, %f};\n' % (index, self.p[0], self.p[1],
                                                self.r), index + 1

def geo_mesh(domain, stepsize, structured=False):
    """
    Input
        domain   (Domain)  object with the geometry\n
        stepsize (float)   Element size\n

    Output
        (dolfin.cpp.mesh.Mesh)\n
    """

    if (structured and domain.dim != 2):
        raise RuntimeError("Structured meshes are only available for 2D geometries.")
    code = 'SetFactory("OpenCASCADE");\nMesh.CharacteristicLengthMin = %f;\nMesh.CharacteristicLengthMax = %f;\n' % (
        stepsize, stepsize)
    code += domain.script()[0]
    extra = "\nTransfinite %s {%d};" % (domain.entity(), domain.index) if structured else ""
    code += '\nPhysical %s(%d) = {%d};%s\nMesh.MshFileVersion = 2.0;' % (domain.entity(),
                                                                           domain.index + 1,
                                                                           domain.index,
                                                                           extra)

    idf = np.random.randint(100000)
    print(code, file=open('%d.geo' % idf, 'w'))
    os.system("gmsh -%d %d.geo" % (domain.dim, idf))
    clear_output(wait=True)
    os.system("dolfin-convert %d.msh %d.xml" % (idf, idf))
    clear_output(wait=True)
    mesh = dolfin.cpp.mesh.Mesh("%d.xml" % idf)
    os.remove("%d.msh" % idf)
    os.remove("%d.xml" % idf)
    try:
        os.remove("%d_physical_region.xml" % idf)
    except:
        None
    os.remove("%d.geo" % idf)
    return mesh

#========================================================================================#
#=====================    GIFS     ==============================#
#========================================================================================#
import imageio.v2 as imageio

def gif_save(drawframe, frames, name, remove=True):
    """Constructs a GIF given a way to plot each frame.

    Input
        drawframe       (function)      Function that specifies how to plot each frame. It should have a single argument,
                                        that being the number of the current frame.
        frames          (int)           Total number of frames.
        name            (str)           Path where to save the GIF file.
        transparency    (bool)          Whether to impose or not a transparent background. Defaults to False.
        remove          (bool)          Whether to leave on the disk or not the files corresponding to each frame.
                                        Defaults to True.
    """
    filenames = []
    for i in range(frames):
        # plot frame
        drawframe(i)

        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)

        # save frame
        plt.savefig(filename)
        plt.close()
    # build gif
    with imageio.get_writer(name + '.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    if (remove):
        for filename in set(filenames):
            os.remove(filename)

#========================================================================================#
#=====================    FINITE ELEMENT     ==============================#
#========================================================================================#
from ufl_legacy.finiteelement.mixedelement import VectorElement, FiniteElement
from ufl_legacy.finiteelement.enrichedelement import NodalEnrichedElement
from fenics import FunctionSpace
from fenics import Function

def fe_space(mesh, obj, deg, scalar=True, bubble=False):
    """Returns the Finite Element (FE) space of specified type (e.g. continuous/discontinuous galerkin) and degree.
    Note: only constructs FE spaces of scalar-valued functions.

    Input
        mesh    (dolfin.cpp.mesh.Mesh)  Underlying mesh of reference\n
        obj     (str)                   Type of space. 'CG' = Continuous Galerkin, 'DG' = Discontinuous Galerkin\n
        deg     (int)                   Polynomial degree at each element\n
        scalar  (bool)                  Whether the space consists of scalar or vector-valued functions (in which
                                        case scalar == True and scalar == False respectively). Defaults to True\n
        bubble  (bool)                  If True, enriches each element with bubble polynomials. Defaults to False\n

    Output
        (dolfin.function.functionspace.FunctionSpace)\n
    """
    if (scalar):
        if (bubble):
            element = FiniteElement(obj, mesh.ufl_cell(), deg) + FiniteElement("Bubble",
                                                                               mesh.ufl_cell(),
                                                                               mesh.topology().dim() + 1)
        else:
            element = FiniteElement(obj, mesh.ufl_cell(), deg)
    else:
        if (bubble):
            element = VectorElement(
                NodalEnrichedElement(FiniteElement(obj, mesh.ufl_cell(), deg),
                                     FiniteElement("Bubble", mesh.ufl_cell(),
                                                   mesh.topology().dim() + 1)))
        else:
            element = VectorElement(obj, mesh.ufl_cell(), deg)

    return FunctionSpace(mesh, element)


def fe_coordinates(space):
    """Returns the coordinates of the degrees of freedom for the given functional space.

    Input
        space   (dolfin.function.functionspace.FunctionSpace).      Functional space for which the dofs have to be located.

    Output
        (numpy.ndarray).
    """
    return space.tabulate_dof_coordinates().astype("float32")

def fe_asvector(u, space):
    """Given a vector of dof values, returns the corresponding object in the functional space.

    Input
        u       (numpy.ndarray or torch.Tensor)                     Vector collecting the values of the function at the
                                                                    degrees of freedom. If u has shape (,n), then
                                                                    the functional space of interest should have n dof.

        space   (dolfin.function.functionspace.FunctionSpace).      Functional space where u belongs.

    Output
        (dolfin.function.function.Function).

    """
    uv = Function(space)
    udata = u if (not isinstance(u, torch.Tensor)) else u.detach().cpu().numpy()
    uv.vector()[:] = udata
    return uv

def fe_plot(obj, space=None, vmin=None, vmax=None, colorbar=False, axis="off", shrink=0.8,
         levels=200, cmap=None):
    """Plots mesh and functional objects.

    Input
        obj         (dolfin.cpp.mesh.Mesh, numpy.ndarray or torch.Tensor)   Object to be plotted. It should be either a mesh
                                                                            or an array containing the values of some function
                                                                            at the degrees of freedom.
        space       (dolfin.function.functionspace.FunctionSpace)           Functional space where 'obj' belongs (assuming 'obj' is not a mesh).
                                                                            Defaults to None, in which case 'obj' is assumed to be a mesh.
        vmin        (float)                                                 If a colorbar is added, then the color legend is calibrated in such a way that vmin
                                                                            is considered the smallest value. Ignored if space = None.
        vmax        (float)                                                 Analogous to vmin.
        colorbar    (bool)                                                  Whether to add or not a colorbar. Ignored if len(*args)=1.
        axis        (obj)                                                   Axis specifics (cf. matplotlib.pyplot.axis). Defaults to "off", thus hiding the axis.
        shrink      (float)                                                 Shrinks the colorbar by the specified factor (defaults to 0.8). Ignored if colorbar = False.
    """
    try:
        if (space == None):
            dolfin.common.plotting.plot(obj)
        else:
            uv = fe_asvector(obj, space)
            if (space.element().value_dimension(0) == 1):
                c = dolfin.common.plotting.plot(uv, vmin=vmin, vmax=vmax,
                                                levels=np.linspace(float(obj.min()),
                                                                      float(obj.max()),
                                                                      levels), cmap=cmap)
            else:
                c = dolfin.common.plotting.plot(uv, cmap=cmap)
            if (colorbar):
                plt.colorbar(c, shrink=shrink)
    except:
        raise RuntimeError(
            "First argument should be either a dolfin.cpp.mesh.Mesh or a structure containing the dof values of some function (in which case 'space' must be != None).")
    plt.axis(axis)


def fe_gif(name, U, dt, T, space, axis="off", figsize=(4, 4), colorbar=False):
    """Builds a GIF animation given the values of a functional object at multiple time steps.

    Input
        name    (str)                                               Path where to save the GIF file.
        U       (numpy.ndarray or torch.Tensor)                     Array of shape (N,n). Each U[j] should contain the
                                                                    values of a functional object at its degrees of freedom.
        dt      (float)                                             Time step with each frame.
        T       (float)                                             Final time. The GIF will have int(T/dt) frames
        space   (dolfin.function.functionspace.FunctionSpace).      Functional space where the U[i]'s belong.
        axis    (obj)                                               Axis specifics (cf. matplotlib.pyplot.axis). Defaults to "off", thus hiding the axis.
        figsize (tuple)                                             Sizes of the window where to plot, width = figsize[0], height = figsize[1].
                                                                    See matplotlib.pyplot.plot.
    """
    frames = int(T / dt)
    N = len(U)
    step = N // frames
    vmin = U.min()
    vmax = U.max()

    def drawframe(i):
        plt.figure(figsize=figsize)
        fe_plot(U[i * step], space, axis=axis, vmin=vmin, vmax=vmax, colorbar=colorbar)

    gif_save(drawframe, frames, name)


def fe_gifz(name, Us, dt, T, space, axis="off", figsize=(4, 4), colorbar=False, titles=None):
    """Builds a GIF animation given the values of a functional object at multiple time steps.

    Input
        name    (str)                                               Path where to save the GIF file.
        U       (numpy.ndarray or torch.Tensor)                     Array of shape (N,n). Each U[j] should contain the
                                                                    values of a functional object at its degrees of freedom.
        dt      (float)                                             Time step with each frame.
        T       (float)                                             Final time. The GIF will have int(T/dt) frames
        space   (dolfin.function.functionspace.FunctionSpace).      Functional space where the U[i]'s belong.
        axis    (obj)                                               Axis specifics (cf. matplotlib.pyplot.axis). Defaults to "off", thus hiding the axis.
        figsize (tuple)                                             Sizes of the window where to plot, width = figsize[0], height = figsize[1].
                                                                    See matplotlib.pyplot.plot.
    """
    U = Us[0]
    frames = int(T / dt)
    N = len(U)
    step = N // frames
    vmin = U.min()
    vmax = U.max()
    fsize = len(Us) * figsize[0], figsize[1]

    def drawframe(i):
        plt.figure(figsize=fsize)
        for j in range(len(Us)):
            U = Us[j]
            plt.subplot(1, len(Us), j + 1)
            fe_plot(U[i * step], space, axis=axis, vmin=vmin, vmax=vmax, colorbar=colorbar)
            if (titles != None):
                plt.title(titles[j])

    gif_save(drawframe, frames, name)

#========================================================================================#
#=====================    CORES     ==============================#
#========================================================================================#
class Core(object):
    """Class for managing CPU and GPU tensors. Objects of this class have the following attributes.

    Attributes
       device   (torch.device)  Underlying core. Equals either torch.device('cpu') or torch.device('cuda:0').

    """

    def __init__(self, device):
        """Creates a reference to the specified core.

        Input
            device  (str)   Device to be used (not case-sensitive). Accepted strings are 'CPU' and 'GPU'."""

        self.dtype = torch.float
        if (device.lower() == "cpu"):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0")

    def tensor(self, array):
        """Converts a numpy array into a torch (float) tensor to be stored on the corresponding core.

        Input
            array   (numpy.ndarray)     Array to be converted.

        Output
            (torch.Tensor).
        """
        return torch.tensor(array, dtype=self.dtype, device=self.device)

    def zeros(self, *shape):
        """Returns a tensor full of zeros.

        Input
            *shape  (tuple of int)     Shape of the tensor. E.g., self.zeros(2,3) creates a 2x3 tensor full of zeros.

        Output
            (torch.Tensor).
        """
        return torch.zeros(*shape, dtype=self.dtype, device=self.device)

    def load(self, *paths):
        """Loads a list of arrays into a single tensor.

        Input
            paths (tuple of str)    Paths where each array is stored. These are assumed to be in .npy format.
                                    All the arrays must have the same shape (except for the first, batch, dimension).

        Output
            (torch.Tensor)."""
        res = []
        for path in paths:
            res.append(self.tensor(np.load(path)))
        return torch.cat(tuple(res))

    def rand(self, *dims):
        """Returns a tensor with randomly filled values. The latter are sampled uniformely from [0,1].

        Input
            *dims   (tuple of int)  Shape of the tensor.

        Output
            (torch.Tensor)."""
        return self.tensor(np.random.rand(*dims))

    def randn(self, *dims):
        """Returns a tensor with randomly filled values. The latter are sampled independently from the normal distribution N(0,1).

        Input
            *dims   (tuple of int)  Shape of the tensor.

        Output
            (torch.Tensor)."""
        return self.tensor(np.random.randn(*dims))

    def __eq__(self, other):
        """Compares two cores.

        Input
            other   (dlroms.cores.Core)     Core to be compared with the current one.

        Output
            (bool) returns True if the two cores both refer to the CPU or GPU respectively."""
        return self.device == other.device

CPU = Core("CPU")
GPU = Core("GPU")

def coreof(u):
    """Returns the core over with a tensor is stored.

    Input
        u   (torch.Tensor)

    Output
        (dlroms.cores.Core)."""

    if (isinstance(u, torch.Tensor)):
        if (u.device == CPU.device):
            return CPU
        elif (u.device == GPU.device):
            return GPU
        else:
            raise RuntimeError("Tensor is stored on an unknown core.")
    else:
        raise RuntimeError("Can only retrieve the core of a torch tensor.")

#========================================================================================#
#=====================    ROMs     ==============================#
#========================================================================================#
from scipy.linalg import eigh

def POD(U, k):
    """Principal Orthogonal Decomposition of the snapshots matrix U into k modes."""
    if (isinstance(U, torch.Tensor)):
        U0 = U.cpu().numpy()
    else:
        U0 = U
    M = np.dot(U0, U0.T)                    # (ntrain,nnodes)*(ntrain,nnodes).T -> (ntrain,ntrain)
    N = U.shape[0]                          # matrix size, number of eigenvalues
    # compute egienvalues (w) and eigenvectors (v)
    w, v = eigh(M, subset_by_index=(N-k, N-1))              # pick the k-ths biggest values
    basis, eigenvalues = np.dot((v/np.sqrt(w)).T, U0), w
    basis, eigenvalues = np.flip(basis, axis = 0)+0, np.flip(eigenvalues)+0
    if (isinstance(U, torch.Tensor)):
        core = coreof(U)
        return core.tensor(basis), core.tensor(eigenvalues)
    else:
        return basis, eigenvalues

def num2p(prob):
    """Converts a number to percentage format."""
    return ("%.2f" % (100*prob)) + "%"

def gramschmidt(V):
    """Orthonormalizes a collection of matrices. V should be a torch tensor in the format batch dimension x space dimension x number of basis."""
    return torch.linalg.qr(V, mode = 'reduced')[0]

def project(vbasis, u, orth = True):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of vectors u = [u1,...,uk], where uj has length Nh, yields the batched
    matrix vector multiplication [(Vj'*Vj)*uj], i.e. the sequence of reconstructed vectors."""
    if(len(vbasis.shape)<3):
        return project(vbasis.unsqueeze(0), u, orth)
    else:
        if(orth):
            return project(gramschmidt(vbasis.transpose(1,2)).transpose(2,1), u, orth = False)
        else:
            return projectup(vbasis, projectdown(vbasis, u))

def projectdown(vbasis, u):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of vectors u = [u1,...,uk], where uj has length Nh, yields the batched
    matrix vector multiplication [Vj*uj], i.e. the sequence of basis coefficients."""
    if(len(vbasis.shape)<3):
        return projectdown(vbasis.unsqueeze(0), u)
    else:
        nh = np.prod(u[0].shape)
        n, nb = vbasis.shape[:2]
        return vbasis.reshape(n, nb, -1).matmul(u.reshape(-1,nh,1))

def projectup(vbasis, c):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of coefficients c = [c1,...,ck], where cj has length b, yields the batched
    matrix vector multiplication [Vj.T*cj], i.e. the sequence of expanded vectors."""
    if(len(vbasis.shape)<3):
      return projectup(vbasis.unsqueeze(0), c)
    else:
      b = c.shape[1]
      n, nb = vbasis.shape[:2]
      return vbasis.reshape(n, nb, -1).transpose(dim0 = 1, dim1 = 2).matmul(c.reshape(-1,b,1)).reshape(-1, vbasis.shape[-1])

#========================================================================================#
#=====================    DNNS     ==============================#
#========================================================================================#
from time import perf_counter

leakyReLU = torch.nn.LeakyReLU(0.1)

class Layer(torch.nn.Module):
    """Layer of a neural network. It is implemented as a subclass of 'torch.nn.Module'. Acts as an abstract class.
    Objects of class layer have the following attributes:

    core (Core): specifies wheather the layer is stored on CPU or GPU.
    rho (function): the activation function of the layer.

    All objects of class layer should implement an abstract method '.module()' that returns the underlying torch.nn.Module.
    Layers can be applied to tensors using .forward() or simply .(), i.e. following the syntax of function calls.
    """

    def __init__(self, activation):
        """Creates a new layer with a given activation function (activation). By default, the layer is initiated on CPU."""
        super(Layer, self).__init__()
        self.rho = activation
        if (activation == None):
            self.rho = torch.nn.Identity()
        self.core = CPU

    def w(self):
        """Returns the weights of the layer."""
        return self.module().weight

    def b(self):
        """Returns the bias vector of the layer."""
        return self.module().bias

    def scale(self, factor):
        """Sets to zero the bias and scales the weight matrix by 'factor'."""
        self.load(factor * self.w().detach().cpu().numpy(),
                  0.0 * self.b().detach().cpu().numpy())

    def zeros(self):
        """Sets to zero all weights and biases."""
        self.module().weight = torch.nn.Parameter(0.0 * self.module().weight)
        self.module().bias = torch.nn.Parameter(0.0 * self.module().bias)

    def moveOn(self, core):
        """Transfers the layer structure on the specified core."""
        self.core = core
        if (core != CPU):
            self.cuda()
        else:
            self.cpu()

    def cuda(self):
        """Transfers the layer to the GPU."""
        self.module().cuda()

    def cpu(self):
        """Transfers the layer to the CPU."""
        self.module().cpu()

    def l2(self):
        """Returns the square sum of all weights within the layer."""
        return (self.module().weight ** 2).sum()

    def l1(self):
        """Returns the absolute sum of all weights within the layer."""
        return self.module().weight.abs().sum()

    def outdim(self, inputdim):
        """Given a tuple for the input dimension, returns the corresponding output dimension."""
        return tuple(self.forward(self.core.zeros(*inputdim)).size())

    def load(self, w, b=None):
        """Given a pair of weights and biases, it loads them as parameters for the Layer.

        Input:
        w (numpy array): weights
        b (numpy array): bias vector. Defaults to None (i.e., only loads w).
        """
        self.module().weight = torch.nn.Parameter(self.core.tensor(w))
        try:
            self.module().bias = torch.nn.Parameter(self.core.tensor(b))
        except:
            None

    def inherit(self, other):
        """Inherits the weight and bias from another network. Additional entries are left to zero.
        It can be seen as a naive form of transfer learning.

        Input:
        other (Layer): the layer from which the parameters are learned.

        Output:
        None, but the current network has now updated parameters.
        """
        self.zeros()
        with torch.no_grad():
            where = tuple([slice(0, s) for s in other.w().size()])
            self.module().weight[where] = torch.nn.Parameter(
                self.core.tensor(other.w().detach().cpu().numpy()))
            where = tuple([slice(0, s) for s in other.b().size()])
            self.module().bias[where] = torch.nn.Parameter(
                self.core.tensor(other.b().detach().cpu().numpy()))

    def __add__(self, other):
        """Connects the current layer to another layer (or a sequence of layers), and returns
        the corresponding nested architecture.

        Input:
        self (Layer): the current layer
        other (Layer / Consecutive): the architecture to be connected on top.

        Output:
        The full architecture, stored as 'Consecutive' object.
        """
        if (isinstance(other, Consecutive) and (not isinstance(other, Parallel))):
            n = len(other)
            layers = [self] + [other[i] for i in range(n)]
            return Consecutive(*tuple(layers))
        else:
            if (other == 0.0):
                return self
            else:
                return Consecutive(self, other)

    def __pow__(self, number):
        """Creates a Parallel architecture by pasting 'number' copies of the same layer next to each other."""
        if (number > 1):
            l = [self]
            for i in range(number - 1):
                l.append(self.copy())
            return Parallel(*tuple(l))
        elif (number == 1):
            return self
        else:
            return 0.0

    def __mul__(self, number):
        """Creates a deep neural network by pasting 'number' copies of the same layer."""
        if (number > 1):
            x = self + self
            for i in range(number - 2):
                x = x + self
            return x
        elif (number == 1):
            return self
        else:
            return 0.0

    def __rmul__(self, number):
        """See self.__mul__."""
        return self * number

    def dof(self):
        """Degrees of freedom in the layer, defined as the number of active weights and biases."""
        return np.prod(tuple(self.module().weight.size())) + len(self.module().bias)

    def He(self, seed=None):
        """He initialization of the weights."""
        if (seed != None):
            torch.manual_seed(seed)
        torch.nn.init.kaiming_normal_(self.module().weight, mode='fan_out',
                                      nonlinearity='leaky_relu', a=0.1)

    def inputdim(self):
        """Returns the expected input dimension for the layer."""
        return self.module().in_features

    def freeze(self, w=True, b=True):
        """Freezes the layer so that its parameters to not require gradients.
        Input:
        w (boolean): wheather to fix or not the weights.
        b (boolean): wheather to fix or not the bias."""
        if (w and b):
            self.module().requires_grad_(False)
        elif (w):
            self.module().weight.requires_grad_(False)
        elif (b):
            self.module().bias.requires_grad_(False)

    def dictionary(self, label=""):
        """Returns a dictionary with the layer parameters. An additional label can be added."""
        return {('w' + label): self.w().detach().cpu().numpy(),
                ('b' + label): self.b().detach().cpu().numpy()}

    def parameters(self):
        ps = list(super(Layer, self).parameters())
        res = []
        for p in ps:
            if (p.requires_grad):
                res.append(p)

        return res


class Dense(Layer):
    """Fully connected Layer."""

    def __init__(self, input_dim, output_dim, activation=leakyReLU):
        """Creates a Dense Layer with given input dimension, output dimension and activation function."""
        super(Dense, self).__init__(activation)
        self.lin = torch.nn.Linear(input_dim, output_dim)

    def module(self):
        return self.lin

    def forward(self, x):
        return self.rho(self.lin(x))


class Sparse(Layer):
    """Layer with weights that have a (priorly) fixed sparsity."""

    def __init__(self, mask, activation=leakyReLU):
        """Creates a Sparse layer.

        Input:
        mask (numpy 2D array): a 2D array that works as sample for the weight matrix.
        It should have the same sparsity required to the weight matrix.
        activation (function): activation function of the layer.
        """
        super(Sparse, self).__init__(activation)
        self.loc = np.nonzero(mask)
        self.in_d, self.out_d = mask.shape
        self.weight = torch.nn.Parameter(CPU.zeros(len(self.loc[0])))
        self.bias = torch.nn.Parameter(CPU.zeros(self.out_d))

    def moveOn(self, core):
        self.core = core
        with torch.no_grad():
            if (core == GPU):
                self.weight = torch.nn.Parameter(self.weight.cuda())
                self.bias = torch.nn.Parameter(self.bias.cuda())
            else:
                self.weight = torch.nn.Parameter(self.weight.cpu())
                self.bias = torch.nn.Parameter(self.bias.cpu())

    def module(self):
        return self

    def forward(self, x):
        W = self.core.zeros(self.in_d, self.out_d)
        W[self.loc] = self.weight
        return self.rho(self.bias + x.mm(W))

    def inherit(self, other):
        W = self.core.zeros(self.in_d, self.out_d)
        W[other.loc] = other.weight
        with torch.no_grad():
            self.weight = torch.nn.Parameter(self.core.copy(W[self.loc]))
            self.bias = torch.nn.Parameter(self.core.copy(other.bias))

    def He(self, seed=None):
        nw = len(self.weight)
        with torch.no_grad():
            self.weight = torch.nn.Parameter(torch.rand(nw) / np.sqrt(nw))

    def W(self):
        W = self.core.zeros(self.in_d, self.out_d)
        W[self.loc] = self.weight
        return W

    def dictionary(self, label=""):
        return {('w' + label): self.w().detach().cpu().numpy(),
                ('b' + label): self.b().detach().cpu().numpy(),
                ('indexes' + label): self.loc}

    def load(self, w, b=None, indexes=None):
        super(Sparse, self).load(w, b)
        if (isinstance(indexes, np.ndarray)):
            self.loc = indexes

#    def cuda(self):
#        self.moveOn(GPU)


class Consecutive(torch.nn.Sequential):
    """Class that handles deep neural networks, obtained by connecting multiple layers.
    It is implemented as a subclass of torch.nn.Sequential.

    Objects of this class support indexing, so that self[k] returns the kth Layer in the architecture.
    """

    def scale(self, factor):
        """Scales all layers in the architecture (see Layer.scale)."""
        for nn in self:
            nn.scale(factor)

    def l2(self):
        """Returns the squared sum of all weights in the architecture."""
        m = 0.0
        N = len(self)
        for i in range(N):
            m += self[i].l2()
        return m

    def l1(self):
        """Returns the absolute sum of all weights in the architecture."""
        m = 0.0
        N = len(self)
        for i in range(N):
            m += self[i].l1norm()
        return m

    def zeros(self):
        """Sets to zero all weights and biases in the architecture."""
        N = len(self)
        for i in range(N):
            self[i].zeros()

    def outdim(self, input_dim=None):
        """Analogous to Layer.outdim."""
        if (input_dim == None):
            input_dim = self[0].inputdim()
        m = input_dim
        N = len(self)
        for i in range(N):
            m = self[i].outdim(m)
        return m

    def inputdim(self):
        """Analogous to Layer.inputdim."""
        return self[0].inputdim()

    def moveOn(self, core):
        """Transfers all layers to the specified core."""
        for layer in self:
            layer.moveOn(core)

    def cuda(self):
        """Transfers all layers to the GPU."""
        N = len(self)
        for i in range(N):
            m = self[i].cuda()

    def cpu(self):
        """Transfers all layers to the CPU."""
        N = len(self)
        for i in range(N):
            m = self[i].cpu()

    def stretch(self):
        res = []
        for nn in self:
            if (isinstance(nn, Consecutive)):
                res += nn.stretch()
            else:
                res += [nn]
        return res

    def dictionary(self, label=""):
        """Returns a dictionary with all the parameters in the network. An additional label can be passed."""
        params = dict()
        k = 0
        for nn in self:
            k += 1
            params.update(nn.dictionary(str(k) + label))
        return params

    def save(self, path, label=""):
        """Stores the whole architecture to the specified path. An additional label can be added (does not influence the name of
        the file, it is only used internally; defaults to '')."""
        if (len(self) == len(self.stretch())):
            params = self.dictionary(label)
            np.savez(path, **params)
        else:
            Consecutive(*self.stretch()).save(path, label)

    def load(self, path, label=""):
        """Loads the architecture parameters from stored data.

        Input:
        path (string): system path where the parameters are stored.
        label (string): additional label required if the stored data had one.
        """
        if (len(self) == len(self.stretch())):
            try:
                params = np.load(path)
            except:
                params = np.load(path + ".npz")
            k = 0
            for nn in self:
                k += 1
                try:
                    if (isinstance(nn, Sparse)):
                        nn.load(w=params['w' + str(k) + label],
                                b=params['b' + str(k) + label],
                                indexes=params['indexes' + str(k) + label])
                    else:
                        nn.load(params['w' + str(k) + label],
                                params['b' + str(k) + label])
                except:
                    None
        else:
            Consecutive(*self.stretch()).load(path, label)

    def __add__(self, other):
        """Augments the current architecture by connecting it with a second one.

        Input:
        self (Consecutive): current architecture.
        other (Layer / Consecutive): neural network to be added at the end.

        Output:
        A Consecutive object consisting of the nested neural network self+other.
        """
        if (isinstance(other, Consecutive) and (not isinstance(other, Parallel))):
            n1 = len(self)
            n2 = len(other)
            layers = [self[i] for i in range(n1)] + [other[i] for i in range(n2)]
            return Consecutive(*tuple(layers))
        else:
            if (other == 0.0):
                return self
            else:
                n1 = len(self)
                layers = [self[i] for i in range(n1)]
                layers.append(other)
                return Consecutive(*tuple(layers))

    def dof(self):
        """Total number of (learnable) weights and biases in the network."""
        res = 0
        for x in self:
            res += x.dof()
        return res

    def He(self, seed=None):
        """Applies the He initialization to all layers in the architecture."""
        for x in self:
            x.He(seed)

    def parameters(self):
        """Returns the list of all learnable parameters in the network. Used as argument for torch optimizers."""
        p = []
        for f in self:
            p += list(f.parameters())
        return p

    def dims(self, inputdim=None):
        """Returns the sequence of dimensions through which an input passes when transformed by the network."""
        if (inputdim == None):
            inp = (1, self[0].inputdim())
        else:
            inp = inputdim
        journey = ""
        ii = inp[1:]
        if (len(ii) == 1):
            journey += str(ii[0])
        else:
            journey += str(inp[1:])
        for f in self:
            journey += " -> "
            inp = f.outdim(inp)
            ii = inp[1:]
            if (len(ii) == 1):
                journey += str(ii[0])
            else:
                journey += str(inp[1:])
        return journey

    def freeze(self, w=True, b=True):
        """Freezes all layers in the network (see Layer.freeze)."""
        for f in self:
            f.freeze(w, b)

    def inherit(self, other):
        """Inherits the networks parameters from a given architecture (cf. Layer.inherit).
        The NN 'other' should have a depth less or equal to that of 'self'.

        Input:
        other (Consecutive): the architecture from which self shoud learn."""
        for i, nn in enumerate(other):
            if (type(self[i]) == type(nn)):
                self[i].inherit(nn)

    def files(self, string):
        return [string + ".npz"]

class Parallel(Consecutive):
    """Architecture with multiple layers that work in parallel but channel-wise. Implemented as a subclass of Consecutive.
    If f1,...,fk is the collection of layers, then Parallel(f1,..,fk)(x) = [f1(x1),..., fk(xk)], where x = [x1,...,xk] is
    structured in k channels."""

    def __init__(self, *args):
        super(Parallel, self).__init__(*args)

    def forward(self, x):
        res = [self[k](x[:, k]) for k in range(len(self))]
        return torch.stack(res, axis=1)

    def __add__(self, other):
        """Augments the current architecture by connecting it with a second one.

        Input:
        self (Parallel): current architecture.
        other (Layer / Consecutive): neural network to be added at the end.

        Output:
        A Consecutive object consisting of the nested neural network self+other.
        """
        if (isinstance(other, Consecutive) and (not isinstance(other, Parallel))):
            n1 = len(self)
            n2 = len(other)
            layers = [self] + [other[i] for i in range(n2)]
            return Consecutive(*tuple(layers))
        else:
            if (other == 0.0):
                return self
            else:
                return Consecutive(self, other)

class Clock(object):
    """Class for measuring (computational) time intervals. Objects of this class have the following attributes:

    tstart (float): time at which the clock was started (in seconds).
    tstop  (float): time at which the clock was stopped (in seconds).
    """

    def __init__(self):
        """Creates a new clock."""
        self.tstart = 0
        self.tstop = 0

    def start(self):
        """Starts the clock."""
        self.tstart = perf_counter()

    def stop(self):
        """Stops the clock."""
        self.tstop = perf_counter()

    def elapsed(self):
        """Returns the elapsed time between the calls .start() and .stop()."""
        dt = self.tstop - self.tstart

        if (dt < 0):
            raise RuntimeError("Clock still running.")
        else:
            return dt

    def elapsedTime(self):
        """Analogous to .elapsed() but returns the output in string format."""
        return Clock.parse(self.elapsed())

    @classmethod
    def parse(cls, time):
        """Converts an amount of seconds in a string of the form '# hours #minutes #seconds'."""
        h = time // 3600
        m = (time - 3600 * h) // 60
        s = time - 3600 * h - 60 * m

        if (h > 0):
            return ("%d hours %d minutes %.2f seconds" % (h, m, s))
        elif (m > 0):
            return ("%d minutes %.2f seconds" % (m, s))
        else:
            return ("%.2f seconds" % s)

    @classmethod
    def shortparse(cls, time):
        """Analogous to Clock.parse but uses the format '#h #m #s'."""
        h = time // 3600
        m = (time - 3600 * h) // 60
        s = time - 3600 * h - 60 * m

        if (h > 0):
            return ("%d h %d m %.2f s" % (h, m, s))
        elif (m > 0):
            return ("%d m %.2f s" % (m, s))
        else:
            return ("%.2f s" % s)


def train(dnn, mu, u, ntrain, epochs, optim=torch.optim.LBFGS, lr=1, lossf=None,
          error=None, verbose=True, until=None, early=False, conv=lambda x: "%.2e" % x,
          best=False, cleanup=True):
    optimizer = optim(dnn.parameters(), lr=lr)
    ntest = len(mu) - ntrain
    mutrain, utrain, mutest, utest = mu[:ntrain], u[:ntrain], mu[-ntest:], u[-ntest:]

    if (error == None):
        def error(a, b):
            return lossf(a, b).item()

    err = []
    clock = Clock()
    clock.start()
    bestv = np.inf
    #tempcode = int(np.random.rand(1) * 1000) # numpy deprecated
    tempcode = np.random.randint(1000)

    for e in range(epochs):

        def closure():
            optimizer.zero_grad()
            loss = lossf(utrain, dnn(mutrain))
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            if (dnn.l2().isnan().item()):
                break
            err.append([error(utrain, dnn(mutrain)),
                        error(utest, dnn(mutest))])
            if (verbose):
                if (cleanup):
                    clear_output(wait=True)
                print("\t\tTrain\tTest")
                print("Epoch " + str(e + 1) + ":\t" + conv(err[-1][0]) + "\t" + conv(
                    err[-1][1]) + ".")
            if (early and e > 0):
                if ((err[-1][1] > err[-2][1]) and (err[-1][0] < err[-2][0])):
                    break
            if (until != None):
                if (err[-1][0] < until):
                    break
            if (best and e > 0):
                if (err[-1][1] < bestv):
                    bestv = err[-1][1] + 0.0
                    dnn.save("temp%d" % tempcode)

    if (best):
        try:
            dnn.load("temp%d" % tempcode)
            for file in dnn.files("temp%d" % tempcode):
                os.remove(file)
        except:
            None
    clock.stop()
    if (verbose):
        print("\n Training complete. Elapsed time: " + clock.elapsedTime() + ".")
    # return err, clock.elapsed()

#========================================================================================#
#=====================    MINNS     ==============================#
#========================================================================================#
class Operator(Sparse):
    def __init__(self, matrix):
        matrix[np.abs(matrix) < 1e-10] = 0
        super(Operator, self).__init__(matrix, None)
        self.load(matrix[np.nonzero(matrix)])
        self.freeze()

    def moveOn(self, core):
        super(Operator, self).moveOn(core)
        self.freeze()

class Bilinear(Operator):
    def __init__(self, operator, space, vspace=None, bcs=[]):
        space1 = space
        space2 = space if (vspace == None) else vspace
        v1, v2 = dolfin.function.argument.TrialFunction(
            space1), dolfin.function.argument.TestFunction(space2)
        M = dolfin.fem.assembling.assemble(operator(v1, v2))
        for bc in bcs:
            bc.apply(M)
        super(Bilinear, self).__init__(M.array())

    def forward(self, x):
        return x[0].mm(self.W().mm(x[1].T))

class Norm(Bilinear):
    def forward(self, x):
        return (x.mm(self.W())*x).sum(axis = -1).sqrt()

class L2(Norm):
    def __init__(self, space):
        def operator(u,v):
            return dolfin.inner(u, v)*dolfin.dx
        super(L2, self).__init__(operator, space)

#========================================================================================#
#=====================    ACTUAL STARING OF THE SCRIPT     ==============================#
#========================================================================================#
scratch_folder_path: Path = Path('./scratch/test_pod/')
scratch_folder_path.mkdir(parents=True, exist_ok=True)

#========================================================================================#
# Full order model (FOM solver)
# Mesh generation
domain = Circle((0,0), 1.0) - Circle((0,0), 0.5)
mesh = geo_mesh(domain, stepsize = 0.3) # dolfin mesh with its objects
fe_plot(mesh)
plt.title("High fidelity mesh ($N_{h}$=%d)" % mesh.num_vertices())
plt.savefig(Path(scratch_folder_path, 'mesh_pic.pdf'))
plt.close()

# Ground-truth model, shape functions and node coordinates
space = fe_space(mesh, 'CG', 1)     # piecewise linear polynomials
x, y = fe_coordinates(space).T              # mesh node coordinates

# torch tensor and polar coordinates
x, y = CPU.tensor(x), CPU.tensor(y)                #make a CPU-torch tensor from a numpy array
r, theta = (x**2 + y**2).sqrt(), torch.atan2(x, y) #convert to polar coordinates (x,y)->(r,theta)

# solution function
# eps -> diffusion, omega -> advection
def solution(t, eps, omega):
    """
    Output
        sol  (torch tensor)  solution tensor at nodes (r,theta)
    """
    T = CPU.tensor([t])
    d2 = (torch.cos(theta) - torch.cos(omega*T))**2 + (torch.sin(theta) - torch.sin(omega*T))**2
    return (r-0.5)*(1.0-r)*torch.exp(- d2 /(2.0*eps*T) )

# a picture of the solution field at t=0.5
u = solution(t = 0.5, eps = 1, omega = 1)  #scalar solution at nodes
fe_plot(u, space, colorbar = True)
plt.title("Solution field")
plt.savefig(Path(scratch_folder_path, 'solution_pic.pdf'))
plt.close()

# Snapshots generation
eps = [1, 5] #[1, 5] #[0.1, 0.2]
omega = [1, 2] #[1, 2] #[2, 5]
time = [0.05, 2]
ntraject = 40                 # N experiments
timesteps = 50                # time steps
nh = space.dim()              # N nodes
U = CPU.zeros(ntraject, timesteps, nh)   # solution tensor (n_experiments, time_steps, n_nodes)
mu = CPU.zeros(ntraject, timesteps, 3)   # normalized tensor (n_experiments, time_steps, 3), elements in [0,1]

dt = (time[1]-time[0])/timesteps

for i in range(ntraject):                        # loop on n_experiemnts
    e0, o0 = CPU.rand(), CPU.rand()              # random number within [0,1)
    e = (eps[1]-eps[0])*e0+eps[0]                # random eps
    o = (omega[1]-omega[0])*o0+omega[0]          # random omega
    for j in range(timesteps):             # loop on time_steps
      t = time[0]+j*dt                     # time at each step
      U[i,j] = solution(t, e, o)           # solution by i_experiment and j_time_step
      # normalized parameters
      mu[i,j,0] = j/timesteps
      mu[i,j,1] = e0
      mu[i,j,2] = o0

# plot experiment=0
example_name = Path(scratch_folder_path, 'example')
fe_gif(str(example_name), U[0], dt=dt, T=time[1] - time[0], space=space)

#========================================================================================#
# Reduced order model - the POD-NN approach
# a) Proper Orthogonal Decomposition

# Splitting of the data: we use 50% of the data to design the ROM, and 50% to test its performances
ntrain = 20
ntrain *= timesteps
u = U.reshape(-1, nh) # reshaping so that we list all snapshots one by one (n_experiment*n_steps,n_nodes)

# POD: proper orthogonal decomposition
pod, eigs = POD(u[:ntrain], k = 10) # POD eigenvectors and corresponding eigenvalues
# NB: "pod" is a (kmodes,nnodes) tensor, that is, basis[k] contains the kth POD mode

plt.figure(figsize = (4,3))
plt.plot(eigs, '-.', color = 'red')
plt.title("Singular values decay", fontsize = 10)
plt.xlabel("# modes")
plt.savefig(Path(scratch_folder_path, 'pod_values_pic.pdf'))
plt.close()

# Let's visualize some of the basis functions
plt.figure(figsize = (10, 2.5))
for j in range(2):
  plt.subplot(1,2,j+1)
  fe_plot(pod[j], space)
  plt.title("Mode n.%d" % (j+1), fontsize = 10)
plt.tight_layout()
plt.savefig(Path(scratch_folder_path, 'modes_pic.pdf'))
plt.close()

# until here space have been used to assign the node coordinates and nodes for solution field

# How well does the POD basis represent solutions?
l2 = L2(space)
def mre(utrue, upred):
    print("\nMRE function")
    return ( l2(utrue-upred)/l2(utrue) ).mean()

urecon = project(pod, u)             # make predictions through the decomposition functions
print("Mean relative error (training): %s." % num2p(mre(u[ntrain:], urecon[ntrain:])))

# print an experiment; -1 the last one
which = -1
Urecon = urecon.reshape(ntraject, timesteps, nh)
dimred_name = Path(scratch_folder_path, 'dimred')
fe_gifz(str(dimred_name), (U[which], Urecon[which]), dt = dt, T = time[1]-time[0], space = space, titles = ["Ground truth", "POD approximation"])

#-------------------------------------------------------#
# b) Learning the POD coordinates
# Projection onto POD coordinates
#                                                [ntrain, nbasis]
c = projectdown(pod, u).squeeze(-1) # torch.Size([2000, 10, 1])
nbasis = c.shape[-1]

# Data reshaping
print(mu.shape)
nparams = 3
mu = mu.reshape(-1, nparams)

print(c.shape)
print(mu.shape)
sys.exit()

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
podnn_name = Path(scratch_folder_path, 'podnn')
fe_gifz(str(podnn_name), (U[which], Urom[which]), dt = dt, T = time[1]-time[0], space = space, titles = ["Ground truth", "POD-NN approximation"])
