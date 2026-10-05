import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from spectral_element_2d import DifferentiableSEM2D

print("=== L-shaped domain Poisson ===")
print("Domain: [0,2]x[0,1] U [0,1]x[1,2]")
print("Manufactured solution: u = sin(pi*x/2)*sin(pi*y/2)")
print()

# L-shaped domain: 3 rectangular elements
elements = [
    (0.0, 1.0, 0.0, 1.0),  # bottom-left
    (1.0, 2.0, 0.0, 1.0),  # bottom-right
    (0.0, 1.0, 1.0, 2.0),  # top-left
]

p = 6
solver = DifferentiableSEM2D(elements, p)
print(f"n_nodes = {solver.n_nodes}, n_interior = {solver.n_interior}")
print(f"n_elements = {solver.n_elements}")

# Manufactured solution: u = sin(pi*x/2)*sin(pi*y/2)
# -Delta u = (pi^2/4 + pi^2/4) * u = pi^2/2 * u
x_int = solver.node_coords[solver.interior, 0]
y_int = solver.node_coords[solver.interior, 1]
u_exact = np.sin(np.pi * x_int / 2) * np.sin(np.pi * y_int / 2)
f_exact = (np.pi**2 / 2) * u_exact

f_t = torch.tensor(f_exact, dtype=torch.float64)
u_sol = solver.solve(f_t).detach().numpy()
l2_err = np.linalg.norm(u_sol - u_exact) / np.linalg.norm(u_exact)
max_err = np.max(np.abs(u_sol - u_exact))
print(f"L2 error = {l2_err:.2e}")
print(f"max error = {max_err:.2e}")
print(f"cond(K) = {np.linalg.cond(solver.K_int):.2e}")

# p-refinement on L-shape
print()
print("=== p-refinement on L-shaped domain ===")
for p in [2, 4, 6, 8]:
    s = DifferentiableSEM2D(elements, p)
    xi = s.node_coords[s.interior, 0]
    yi = s.node_coords[s.interior, 1]
    ue = np.sin(np.pi * xi / 2) * np.sin(np.pi * yi / 2)
    fe = (np.pi**2 / 2) * ue
    us = s.solve(torch.tensor(fe, dtype=torch.float64)).detach().numpy()
    err = np.linalg.norm(us - ue) / np.linalg.norm(ue)
    print(f"  p={p}, n={s.n_interior}: L2 error = {err:.2e}")

# Optimization on L-shape
print()
print("=== Optimization on L-shaped domain ===")
alpha = 1e-6
u_d = torch.tensor(u_exact, dtype=torch.float64)
n = solver.n_interior

def objective(f):
    u = solver.solve(f)
    return 0.5 * ((u - u_d)**2).sum() + alpha * 0.5 * (f**2).sum()

f = torch.zeros(n, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.LBFGS([f], lr=1.0, max_iter=200)
def closure():
    optimizer.zero_grad()
    loss = objective(f)
    loss.backward()
    return loss

for i in range(50):
    loss = optimizer.step(closure)

u_final = solver.solve(f).detach().numpy()
state_err = np.linalg.norm(u_final - u_exact) / np.linalg.norm(u_exact)
ctrl_err = np.linalg.norm(f.detach().numpy() - f_exact) / np.linalg.norm(f_exact)
print(f"alpha = {alpha}")
print(f"State error = {state_err:.4%}")
print(f"Control error = {ctrl_err:.4%}")
print(f"Final loss = {objective(f).item():.6e}")
