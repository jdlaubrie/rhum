import numpy as np
from scipy.linalg import eigh
import sys

# Define the truss geometry
nodes = np.array([
    [0.0, 0.0],  # Node 0
    [0.3, 0.0],  # Node 1
    [0.7, 0.0],  # Node 2
    [1.0, 0.0]   # Node 3
])

elements = np.array([
    [0, 1],  # Element 0
    [1, 2],  # Element 1
    [2, 3]   # Element 2
])

# Material properties
E = 210e3  # Young's modulus (Pa)
A = 0.01   # Cross-sectional area (m^2)

load_node = 3
#========================================================================================#
# Function to compute the local stiffness matrix for a truss element
def local_stiffness_matrix(E, A, L):
    k = (E * A / L) * np.array([[1, -1], [-1, 1]])
    return k

# Function to assemble the global stiffness matrix
def assemble_global_stiffness(nodes, elements, E, A):
    num_dofs = nodes.shape[0]
    print("Number of degrees of freedom in the system: {0:d}".format(num_dofs))
    K_global = np.zeros((num_dofs, num_dofs))
    for element in elements:
        nodes_indices = element
        x1, y1 = nodes[nodes_indices[0]]
        x2, y2 = nodes[nodes_indices[1]]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        Ke = local_stiffness_matrix(E, A, L)

        # Map local DOFs to global DOFs
        dof_map = nodes_indices
        for i in range(2):
            for j in range(2):
                K_global[dof_map[i], dof_map[j]] += Ke[i, j]
    return K_global

# Apply boundary conditions
def apply_boundary_conditions(K_global, F_global, free_cols):
    K_bc = K_global[free_cols, :][:, free_cols]
    F_bc = F_global[free_cols]
    return K_bc, F_bc

# Generate snapshots by varying loads
def generate_snapshots(K_global, n_loads, free_cols):
    snapshots = np.zeros((len(n_loads), K_global.shape[0]), dtype=float)
    for i, load in enumerate(n_loads):
        u = np.zeros(K_global.shape[0], dtype=float)
        F_global = np.zeros(K_global.shape[0], dtype=float)
        F_global[load_node] = load  # Apply load at node 2
        K_global_bc, F_global_bc = apply_boundary_conditions(K_global.copy(), F_global, free_cols)
        u[free_cols] = np.linalg.solve(K_global_bc, F_global_bc)
        snapshots[i, :] = u
    return snapshots

# Perform POD
def perform_pod(snapshots, k):
    U = snapshots
    M = np.dot(U, U.T)
    N = U.shape[0]  # matrix size, number of eigenvalues
    w, v = eigh(M, subset_by_index=(N-k, N-1))
    #basis, eigenvalues = np.dot(v.T, U), w
    basis, eigenvalues = np.dot((v/np.sqrt(w)).T, U), w
    #basis, eigenvalues = np.flip(basis, axis=0)+0, np.flip(eigenvalues)+0
    return basis, eigenvalues

# Project the system onto the reduced basis
def project_solution(U, basis):
    U_project = U @ basis.T @ basis
    return U_project

# Compute the error between actual and reconstructed snapshots
def compute_error(actual_snapshots, projected_snapshots):
    return np.mean(np.linalg.norm(actual_snapshots - projected_snapshots, axis=1)/np.linalg.norm(actual_snapshots, axis=1))

#========================================================================================#
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Prepare training data
def prepare_training_data(snapshots, parameters):
    # Assuming parameters is a 2D array where each row corresponds to a set of parameters for a snapshot
    inputs = torch.tensor(parameters, dtype=torch.float32)
    targets = torch.tensor(snapshots, dtype=torch.float32)
    return inputs, targets

# Train the neural network
def train_neural_network(model, inputs, targets, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Predict POD coordinates using the neural network
def model_evaluation(model, parameters):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(parameters, dtype=torch.float32)
        predicted_coordinates = model(inputs)
    return predicted_coordinates.numpy()

#========================================================================================#
# Main ROM workflow
K_global = assemble_global_stiffness(nodes, elements, E, A)

# Define fixed degrees of freedom (fixing node 0 in x and y directions)
fixed_dofs = [0]
free_cols = np.ones(K_global.shape[0], dtype=bool)
free_cols[fixed_dofs] = 0

# Generate training snapshots for different load cases
#load_train = [1000, 1500, 2000, 2500]  # Example load cases
load_train = -500.0 + 3000.0*np.random.rand(5)  # Example load cases
snapshots_train = generate_snapshots(K_global, load_train, free_cols)
print("\nSnapshots (solution) for training:")
print("Load    | Displacements")
for i in range(load_train.shape[0]):
    print("{0:.2f} | ".format(load_train[i]) + np.array2string(snapshots_train[i,:]))

# Generate testing snapshots for different load cases
#load_test = [1200, 1600, 1900, 2300]  # Example load cases
load_test = -500.0 + 3000.0*np.random.rand(5)  # Example load cases
snapshots_test = generate_snapshots(K_global, load_test, free_cols)
print("\nSnapshots (solution) for testing:")
print("Load    | Displacements")
for i in range(load_test.shape[0]):
    print("{0:.2f} | ".format(load_test[i]) + np.array2string(snapshots_test[i,:]))

# Perform POD to extract the reduced basis
k = 2  # Number of POD modes to retain
basis, eigenvalues = perform_pod(snapshots_train, k)

# Project the system onto the reduced basis
snapshots_projected = project_solution(snapshots_test, basis)
print("\nSnapshots (solution) projected on the POD:")
print("Load    | Displacements")
for i in range(load_train.shape[0]):
    print("{0:.2f} | ".format(load_test[i]) + np.array2string(snapshots_projected[i,:]))

# Compute the error between actual and reconstructed snapshots
error_proj = compute_error(snapshots_test, snapshots_projected)
print("\nError between actual and reconstructed snapshots:", error_proj)

#-------------------------------------------------------#
# Prepare training data
parameters = np.array([[load] for load in load_train])  # Example parameters
snapshots_down = snapshots_train @ basis.T
inputs, targets = prepare_training_data(snapshots_down, parameters)

# Define the neural network
input_dim = inputs.shape[1]
hidden_dim = 10
output_dim = targets.shape[1]
model = SimpleNN(input_dim, hidden_dim, output_dim)

# Train the neural network
print("Training the neural network ...")
train_neural_network(model, inputs, targets)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Trainable parameters:", trainable_params)

new_parameters = np.array([[load] for load in load_test])  # Example parameters
trained_snapshots = model_evaluation(model, new_parameters)
snapshots_up = trained_snapshots @ basis
print("\nSnapshots (solution) trained from the neural network:")
print("Load    | Displacements")
for i in range(load_train.shape[0]):
    print("{0:.2f} | ".format(load_test[i]) + np.array2string(snapshots_up[i,:]))

# Compute the error between actual and trained snapshots
error_rom = compute_error(snapshots_test, snapshots_up)
print("\nError between actual and trained snapshots:", error_rom)

