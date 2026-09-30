"""Test the differentiable spectral element solver."""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from spectral_element import DifferentiableSEM1D, gll_points

torch.manual_seed(42)

print("=" * 60)
print("Test 1: GLL points correctness")
print("=" * 60)
for p in [2, 4, 8]:
    x, w = gll_points(p)
    print(f"p={p}: {len(x)} points, sum(w)={sum(w):.10f} (should be 2.0)")
    print(f"  x range: [{x[0]:.6f}, {x[-1]:.6f}]")

print("\n" + "=" * 60)
print("Test 2: Poisson solver accuracy (-u'' = f, u(0)=u(1)=0)")
print("=" * 60)
# Exact solution: u = sin(pi*x), f = pi^2 * sin(pi*x)
elements = [(0.0, 0.5), (0.5, 1.0)]  # 2 elements
for p in [4, 8, 16]:
    solver = DifferentiableSEM1D(elements, p)
    x_int = solver.node_coords[solver.interior]
    f_exact = (np.pi**2) * np.sin(np.pi * x_int)
    f_t = torch.tensor(f_exact, dtype=torch.float64)
    u_int = solver.solve(f_t)
    u_exact = np.sin(np.pi * x_int)
    error = np.max(np.abs(u_int.detach().numpy() - u_exact))
    print(f"p={p}, n_interior={solver.n_interior}: max error = {error:.2e}")

print("\n" + "=" * 60)
print("Test 3: Gradient check (AD vs finite difference)")
print("=" * 60)
solver = DifferentiableSEM1D([(0.0, 0.5), (0.5, 1.0)], p=8)
n = solver.n_interior
x_int = solver.node_coords[solver.interior]
u_d = torch.tensor(np.sin(np.pi * x_int), dtype=torch.float64)

def objective(f):
    u = solver.solve(f)
    return 0.5 * ((u - u_d)**2).sum() + 0.001 * 0.5 * (f**2).sum()

f0 = torch.zeros(n, dtype=torch.float64, requires_grad=True)
loss = objective(f0)
loss.backward()
grad_ad = f0.grad.detach().numpy()

# Finite difference gradient
eps = 1e-6
grad_fd = np.zeros(n)
for i in range(n):
    f_plus = f0.detach().clone()
    f_plus[i] += eps
    f_minus = f0.detach().clone()
    f_minus[i] -= eps
    grad_fd[i] = (objective(f_plus).item() - objective(f_minus).item()) / (2 * eps)

rel_error = np.linalg.norm(grad_ad - grad_fd) / np.linalg.norm(grad_fd)
print(f"AD gradient norm: {np.linalg.norm(grad_ad):.6e}")
print(f"FD gradient norm: {np.linalg.norm(grad_fd):.6e}")
print(f"Relative error: {rel_error:.2e}")
print(f"Max abs diff: {np.max(np.abs(grad_ad - grad_fd)):.2e}")

print("\n" + "=" * 60)
print("Test 4: Optimization convergence")
print("=" * 60)
f = torch.zeros(n, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.LBFGS([f], lr=1.0, max_iter=50)

def closure():
    optimizer.zero_grad()
    loss = objective(f)
    loss.backward()
    return loss

for i in range(10):
    optimizer.step(closure)

u_final = solver.solve(f).detach().numpy()
u_exact = np.sin(np.pi * x_int)
state_error = np.linalg.norm(u_final - u_exact) / np.linalg.norm(u_exact)
print(f"Final state error: {state_error:.4%}")
print(f"Final objective: {objective(f).item():.6e}")

print("\nAll tests completed!")
