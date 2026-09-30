import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import json
from spectral_element_3d import DifferentiableSEM3D

print("=== 3D Poisson: -Delta u = f, u = sin(pi x) sin(pi y) sin(pi z) ===")
print("f = 3*pi^2 * u")
print()

# 2x2x2 elements on [0,1]^3
elements = []
for k in range(2):
    for j in range(2):
        for i in range(2):
            elements.append((i*0.5, (i+1)*0.5, j*0.5, (j+1)*0.5, k*0.5, (k+1)*0.5))

results = {}
for p in [2, 4]:
    solver = DifferentiableSEM3D(elements, p)
    x_int = solver.node_coords[solver.interior, 0]
    y_int = solver.node_coords[solver.interior, 1]
    z_int = solver.node_coords[solver.interior, 2]
    u_exact = np.sin(np.pi*x_int) * np.sin(np.pi*y_int) * np.sin(np.pi*z_int)
    f_exact = 3 * np.pi**2 * u_exact
    
    f_t = torch.tensor(f_exact, dtype=torch.float64)
    u_sol = solver.solve(f_t).detach().numpy()
    l2_err = np.linalg.norm(u_sol - u_exact) / np.linalg.norm(u_exact)
    max_err = np.max(np.abs(u_sol - u_exact))
    cond = np.linalg.cond(solver.K_int)
    
    print(f"p={p}, n_interior={solver.n_interior}, n_nodes={solver.n_nodes}")
    print(f"  L2 error = {l2_err:.2e}, max error = {max_err:.2e}, cond(K)={cond:.2e}")
    results[f'p{p}'] = {'n_interior': solver.n_interior, 'l2_error': l2_err, 'max_error': max_err, 'cond': float(cond)}

# Gradient check
print()
print("=== 3D Gradient check (p=2) ===")
solver = DifferentiableSEM3D(elements, p=2)
n = solver.n_interior
x_int = solver.node_coords[solver.interior, 0]
y_int = solver.node_coords[solver.interior, 1]
z_int = solver.node_coords[solver.interior, 2]
u_d = torch.tensor(np.sin(np.pi*x_int)*np.sin(np.pi*y_int)*np.sin(np.pi*z_int), dtype=torch.float64)

alpha = 1e-6
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
results['gradient_error'] = float(rel_err)

# 3D Optimization
print()
print("=== 3D Optimization (p=2, alpha=1e-6) ===")
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
u_exact = np.sin(np.pi*x_int)*np.sin(np.pi*y_int)*np.sin(np.pi*z_int)
f_exact = 3 * np.pi**2 * u_exact
state_err = np.linalg.norm(u_final - u_exact) / np.linalg.norm(u_exact)
ctrl_err = np.linalg.norm(f.detach().numpy() - f_exact) / np.linalg.norm(f_exact)
print(f"State error = {state_err:.4%}")
print(f"Control error = {ctrl_err:.4%}")
print(f"Final loss = {objective(f).item():.6e}")
results['optimization'] = {'state_error': float(state_err), 'control_error': float(ctrl_err)}

# Scalability: more elements
print()
print("=== Scalability: 4x4x4 elements, p=2 ===")
elements_fine = []
for k in range(4):
    for j in range(4):
        for i in range(4):
            elements_fine.append((i*0.25, (i+1)*0.25, j*0.25, (j+1)*0.25, k*0.25, (k+1)*0.25))
solver_fine = DifferentiableSEM3D(elements_fine, p=2)
xi = solver_fine.node_coords[solver_fine.interior, 0]
yi = solver_fine.node_coords[solver_fine.interior, 1]
zi = solver_fine.node_coords[solver_fine.interior, 2]
ue = np.sin(np.pi*xi)*np.sin(np.pi*yi)*np.sin(np.pi*zi)
fe = 3*np.pi**2*ue
us = solver_fine.solve(torch.tensor(fe, dtype=torch.float64)).detach().numpy()
err = np.linalg.norm(us - ue) / np.linalg.norm(ue)
print(f"n_interior={solver_fine.n_interior}, n_nodes={solver_fine.n_nodes}, L2 error={err:.2e}")
results['scalability'] = {'n_interior': solver_fine.n_interior, 'l2_error': float(err)}

with open('results_3d.json', 'w') as fout:
    json.dump(results, fout, indent=2)
print("\nResults saved to results_3d.json")
