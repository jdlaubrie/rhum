import numpy as np
from scipy.linalg import eigh
from scipy.linalg import solve as scisolve
import torch
from cores import coreof, CPU, GPU


def POD(U, k):
    """Principal Orthogonal Decomposition of the snapshots matrix U into k modes."""
    if(isinstance(U, torch.Tensor)):
        U0 = U.cpu().numpy()
    else:
        U0 = U
    M = np.dot(U0, U0.T)
    N = U.shape[0]
    w, v = eigh(M, subset_by_index = (N-k, N-1))
    basis, eigenvalues = np.dot((v/np.sqrt(w)).T, U0), w
    basis, eigenvalues = np.flip(basis, axis = 0)+0, np.flip(eigenvalues)+0
    if(isinstance(U, torch.Tensor)):
        core = coreof(U)
        return core.tensor(basis), core.tensor(eigenvalues)
    else:
        return basis, eigenvalues

def num2p(prob):
    """Converts a number to percentage format."""
    return ("%.2f" % (100*prob)) + "%"

def projectdown(vbasis, u):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of vectors u = [u1,...,uk], where uj has length Nh, yields the batched
    matrix vector multiplication [Vjuj], i.e. the sequence of basis coefficients."""
    if(len(vbasis.shape)<3):
      return projectdown(vbasis.unsqueeze(0), u)
    else:
      nh = np.prod(u[0].shape)
      n, nb = vbasis.shape[:2]
      return vbasis.reshape(n, nb, -1).matmul(u.reshape(-1,nh,1))

def projectup(vbasis, c):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of coefficients c = [c1,...,ck], where cj has length b, yields the batched
    matrix vector multiplication [Vj.Tcj], i.e. the sequence of expanded vectors."""
    if(len(vbasis.shape)<3):
      return projectup(vbasis.unsqueeze(0), c)
    else:
      b = c.shape[1]
      n, nb = vbasis.shape[:2]
      return vbasis.reshape(n, nb, -1).transpose(dim0 = 1, dim1 = 2).matmul(c.reshape(-1,b,1)).reshape(-1, vbasis.shape[-1])

def project(vbasis, u, orth = True):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of vectors u = [u1,...,uk], where uj has length Nh, yields the batched
    matrix vector multiplication [Vj'Vjuj], i.e. the sequence of reconstructed vectors."""
    if(len(vbasis.shape)<3):
        return project(vbasis.unsqueeze(0), u, orth)
    else:
        if(orth):
            return project(gramschmidt(vbasis.transpose(1,2)).transpose(2,1), u, orth = False)
        else:
            return projectup(vbasis, projectdown(vbasis, u))

def gramschmidt(V):
    """Orthonormalizes a collection of matrices. V should be a torch tensor in the format batch dimension x space dimension x number of basis."""
    return torch.linalg.qr(V, mode = 'reduced')[0]

def PAs(V1, V2, orth = True):
    """List of principal angles between the subspaces in V1 and V2. The Vj's should be in the format 
    batch dimension x space dimension x number of basis."""
    if(orth):
        A1, A2 = gramschmidt(V1), gramschmidt(V2)
    else:
        A1, A2 = V1, V2
    vals = torch.linalg.svdvals(A1.transpose(dim0 = 1, dim1 = 2).matmul(A2)).clamp(min=0,max=1)
    return vals.arccos()