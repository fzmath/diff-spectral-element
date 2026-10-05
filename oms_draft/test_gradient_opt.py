import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from spectral_element import DifferentiableSEM1D

torch.manual_seed(42)

print("=== Gradient check (AD vs finite difference) ===")
solver = DifferentiableSEM1D([(0.0, 0.5), (0.5, 1.0)], p=8)
n = solver.n_interior
x_int = solver.node_coords[solver.interior]
u_d = torch.tensor(np.sin(np.pi * x_int), dtype=torch.float64)

alpha = 0.001
def objective(f):
    u = solver.solve(f)
    return 0.5 * ((u - u_d)**2).sum() + alpha * 0.5 * (f**2).sum()

f0 = torch.zeros(n, dtype=torch.float64, requires_grad=True)
loss = objective(f0)
loss.backward()
grad_ad = f0.grad.detach().numpy()

# Finite difference on first 5 nodes only (for speed)
eps = 1e-6
n_check = min(5, n)
grad_fd = np.zeros(n_check)
for i in range(n_check):
    f_plus = f0.detach().clone()
    f_plus[i] += eps
    f_minus = f0.detach().clone()
    f_minus[i] -= eps
    grad_fd[i] = (objective(f_plus).item() - objective(f_minus).item()) / (2 * eps)

rel_error = np.linalg.norm(grad_ad[:n_check] - grad_fd) / np.linalg.norm(grad_fd)
print(f"Checked {n_check}/{n} components")
print(f"AD gradient (first 5): {grad_ad[:5]}")
print(f"FD gradient (first 5): {grad_fd}")
print(f"Relative error: {rel_error:.2e}")

print()
print("=== Optimization (L-BFGS) ===")
f = torch.zeros(n, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.LBFGS([f], lr=1.0, max_iter=50, history_size=10)

def closure():
    optimizer.zero_grad()
    loss = objective(f)
    loss.backward()
    return loss

losses = []
for i in range(20):
    loss = optimizer.step(closure)
    losses.append(loss.item())
    if i % 5 == 0:
        print(f"  iter {i}: loss = {loss.item():.6e}")

u_final = solver.solve(f).detach().numpy()
u_exact = np.sin(np.pi * x_int)
f_exact = np.pi**2 * np.sin(np.pi * x_int)
state_error = np.linalg.norm(u_final - u_exact) / np.linalg.norm(u_exact)
control_error = np.linalg.norm(f.detach().numpy() - f_exact) / np.linalg.norm(f_exact)
print(f"Final state error: {state_error:.4%}")
print(f"Final control error: {control_error:.4%}")
print(f"Final objective: {objective(f).item():.6e}")

print()
print("=== Spectral convergence (p-refinement) ===")
for p in [2, 4, 8, 16]:
    s = DifferentiableSEM1D([(0.0, 0.5), (0.5, 1.0)], p)
    x = s.node_coords[s.interior]
    f_t = torch.tensor(np.pi**2 * np.sin(np.pi * x), dtype=torch.float64)
    u = s.solve(f_t).detach().numpy()
    err = np.linalg.norm(u - np.sin(np.pi * x)) / np.linalg.norm(np.sin(np.pi * x))
    print(f"  p={p:2d}, n={s.n_interior:2d}: L2 error = {err:.2e}")
