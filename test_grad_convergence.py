import sys; sys.path.insert(0,'.')
import torch, numpy as np
from spectral_element import DifferentiableSEM1D

torch.manual_seed(42)
print("p | n | state_L2_err | grad_FD_rel_err")
for p in [2,4,6,8,10,12]:
    solver = DifferentiableSEM1D([(0.0,0.5),(0.5,1.0)], p=p)
    n = solver.n_interior
    x = solver.node_coords[solver.interior]
    u_d = torch.tensor(np.sin(np.pi*x), dtype=torch.float64)
    f_exact = np.pi**2 * np.sin(np.pi*x)

    # State error
    u = solver.solve(torch.tensor(f_exact, dtype=torch.float64)).detach().numpy()
    state_err = np.linalg.norm(u - u_d.numpy())/np.linalg.norm(u_d.numpy())

    # Gradient FD check
    alpha = 1e-6
    def obj(f):
        uu = solver.solve(f)
        return 0.5*((uu-u_d)**2).sum() + alpha*0.5*(f**2).sum()
    f0 = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    loss = obj(f0); loss.backward()
    grad_ad = f0.grad.detach().numpy()
    eps = 1e-6
    n_check = min(3, n)
    grad_fd = np.zeros(n_check)
    for i in range(n_check):
        fp = f0.detach().clone(); fp[i] += eps
        fm = f0.detach().clone(); fm[i] -= eps
        grad_fd[i] = (obj(fp).item() - obj(fm).item())/(2*eps)
    grad_err = np.linalg.norm(grad_ad[:n_check]-grad_fd)/np.linalg.norm(grad_fd)
    print(f"{p} | {n} | {state_err:.2e} | {grad_err:.2e}")
