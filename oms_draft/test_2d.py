import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from spectral_element_2d import DifferentiableSEM2D

print("=== 2D Poisson solver: -Delta u = f, u = sin(pi x) sin(pi y) ===")
print("f = 2*pi^2 * sin(pi x) * sin(pi y)")
print()

# 2x2 elements on [0,1]^2
elements = [
    (0.0, 0.5, 0.0, 0.5),
    (0.5, 1.0, 0.0, 0.5),
    (0.0, 0.5, 0.5, 1.0),
    (0.5, 1.0, 0.5, 1.0),
]

for p in [4, 8]:
    solver = DifferentiableSEM2D(elements, p)
    x_int = solver.node_coords[solver.interior, 0]
    y_int = solver.node_coords[solver.interior, 1]
    f_exact = 2 * np.pi**2 * np.sin(np.pi * x_int) * np.sin(np.pi * y_int)
    u_exact = np.sin(np.pi * x_int) * np.sin(np.pi * y_int)
    
    f_t = torch.tensor(f_exact, dtype=torch.float64)
    u_sol = solver.solve(f_t).detach().numpy()
    
    l2_err = np.linalg.norm(u_sol - u_exact) / np.linalg.norm(u_exact)
    max_err = np.max(np.abs(u_sol - u_exact))
    print(f"p={p}, n_interior={solver.n_interior}, n_nodes={solver.n_nodes}")
    print(f"  L2 error = {l2_err:.2e}, max error = {max_err:.2e}")
    print(f"  cond(K) = {np.linalg.cond(solver.K_int):.2e}")

print()
print("=== 2D Gradient check ===")
solver = DifferentiableSEM2D(elements, p=4)
n = solver.n_interior
x_int = solver.node_coords[solver.interior, 0]
y_int = solver.node_coords[solver.interior, 1]
u_d = torch.tensor(np.sin(np.pi * x_int) * np.sin(np.pi * y_int), dtype=torch.float64)

alpha = 0.001
def objective(f):
    u = solver.solve(f)
    return 0.5 * ((u - u_d)**2).sum() + alpha * 0.5 * (f**2).sum()

f0 = torch.zeros(n, dtype=torch.float64, requires_grad=True)
loss = objective(f0)
loss.backward()
grad_ad = f0.grad.detach().numpy()

eps = 1e-6
n_check = min(5, n)
grad_fd = np.zeros(n_check)
for i in range(n_check):
    fp = f0.detach().clone(); fp[i] += eps
    fm = f0.detach().clone(); fm[i] -= eps
    grad_fd[i] = (objective(fp).item() - objective(fm).item()) / (2 * eps)

rel_err = np.linalg.norm(grad_ad[:n_check] - grad_fd) / np.linalg.norm(grad_fd)
print(f"Checked {n_check}/{n} components, relative error = {rel_err:.2e}")

print()
print("=== 2D Optimization ===")
f = torch.zeros(n, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.LBFGS([f], lr=1.0, max_iter=100)
def closure():
    optimizer.zero_grad()
    loss = objective(f)
    loss.backward()
    return loss

for i in range(30):
    loss = optimizer.step(closure)

u_final = solver.solve(f).detach().numpy()
u_exact = np.sin(np.pi * x_int) * np.sin(np.pi * y_int)
f_exact = 2 * np.pi**2 * np.sin(np.pi * x_int) * np.sin(np.pi * y_int)
state_err = np.linalg.norm(u_final - u_exact) / np.linalg.norm(u_exact)
ctrl_err = np.linalg.norm(f.detach().numpy() - f_exact) / np.linalg.norm(f_exact)
print(f"State error = {state_err:.4%}")
print(f"Control error = {ctrl_err:.4%}")
print(f"Final loss = {objective(f).item():.6e}")
