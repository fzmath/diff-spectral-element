import sys; sys.path.insert(0,'.')
import torch, numpy as np
from spectral_element_2d import DifferentiableSEM2D

torch.manual_seed(42)
# 2x2 elements on (0,1)^2
elems = [(0,0.5,0,0.5),(0.5,1,0,0.5),(0,0.5,0.5,1),(0.5,1,0.5,1)]
solver = DifferentiableSEM2D(elems, p=8)
n = solver.n_interior
coords = solver.node_coords[solver.interior]
x_int, y_int = coords[:,0], coords[:,1]
u_d = np.sin(np.pi*x_int)*np.sin(np.pi*y_int)
u_d_t = torch.tensor(u_d, dtype=torch.float64)

# Forward solve
f_exact = 2*np.pi**2 * u_d
f_t = torch.tensor(f_exact, dtype=torch.float64)
u = solver.solve(f_t).detach().numpy()
l2_err = np.linalg.norm(u - u_d)/np.linalg.norm(u_d)
print(f'2D p=8 forward L2 error: {l2_err:.2e}')
print(f'n_interior: {n}')

# Gradient check
eps = 1e-6
def obj(ft):
    uu = solver.solve(ft)
    return 0.5*((uu-u_d_t)**2).sum() + 1e-6*0.5*(ft**2).sum()

f0 = torch.zeros(n, dtype=torch.float64, requires_grad=True)
loss = obj(f0); loss.backward()
grad_ad = f0.grad.detach().numpy()
grad_fd = np.zeros(5)
for i in range(5):
    fp = f0.detach().clone(); fp[i] += eps
    fm = f0.detach().clone(); fm[i] -= eps
    grad_fd[i] = (obj(fp).item() - obj(fm).item())/(2*eps)
rel = np.linalg.norm(grad_ad[:5]-grad_fd)/np.linalg.norm(grad_fd)
print(f'2D gradient rel error (5 comps): {rel:.2e}')
