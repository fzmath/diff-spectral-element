"""2D linear elasticity: vector Poisson -mu*Delta u = f, u on boundary.
Simplified plane stress with mu=1, decoupled x/y components.
"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np
from spectral_element_2d import DifferentiableSEM2D

torch.manual_seed(42)
elems = [(0,0.5,0,0.5),(0.5,1,0,0.5),(0,0.5,0.5,1),(0.5,1,0.5,1)]
p = 8
solver = DifferentiableSEM2D(elems, p=p)
n = solver.n_interior
coords = solver.node_coords[solver.interior]
x, y = coords[:,0], coords[:,1]

# Exact displacement: u_x = sin(pi x) sin(pi y), u_y = sin(2pi x) sin(pi y)
# -Delta u = f (mu=1)
ux_exact = np.sin(np.pi*x)*np.sin(np.pi*y)
uy_exact = np.sin(2*np.pi*x)*np.sin(np.pi*y)
fx_exact = 2*np.pi**2 * ux_exact
fy_exact = (4*np.pi**2 + np.pi**2) * uy_exact  # -(4pi^2+pi^2) sin(2pix)sin(piy) = 5pi^2 uy

# Solve u_x
ux = solver.solve(torch.tensor(fx_exact, dtype=torch.float64)).detach().numpy()
# Solve u_y
uy = solver.solve(torch.tensor(fy_exact, dtype=torch.float64)).detach().numpy()

err_x = np.linalg.norm(ux - ux_exact)/np.linalg.norm(ux_exact)
err_y = np.linalg.norm(uy - uy_exact)/np.linalg.norm(uy_exact)
print(f"Elasticity u_x L2 error: {err_x:.2e}")
print(f"Elasticity u_y L2 error: {err_y:.2e}")
print(f"n_interior per component: {n}")

# Gradient check: combined state [ux, uy], tracking both to exact
ux_d = torch.tensor(ux_exact, dtype=torch.float64)
uy_d = torch.tensor(uy_exact, dtype=torch.float64)
alpha = 1e-6

def objective(fvec):
    fx, fy = fvec[:n], fvec[n:]
    ux = solver.solve(fx)
    uy = solver.solve(fy)
    return 0.5*((ux-ux_d)**2).sum() + 0.5*((uy-uy_d)**2).sum() + alpha*0.5*(fvec**2).sum()

f0 = torch.zeros(2*n, dtype=torch.float64, requires_grad=True)
loss = objective(f0); loss.backward()
grad_ad = f0.grad.detach().numpy()

eps = 1e-6
grad_fd = np.zeros(3)
for i in range(3):
    fp = f0.detach().clone(); fp[i] += eps
    fm = f0.detach().clone(); fm[i] -= eps
    grad_fd[i] = (objective(fp).item() - objective(fm).item())/(2*eps)
rel = np.linalg.norm(grad_ad[:3]-grad_fd)/np.linalg.norm(grad_fd)
print(f"Combined gradient FD error: {rel:.2e}")
