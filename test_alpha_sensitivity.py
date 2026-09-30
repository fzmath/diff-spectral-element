import sys; sys.path.insert(0,'.')
import torch, numpy as np
from spectral_element import DifferentiableSEM1D

torch.manual_seed(42)
solver = DifferentiableSEM1D([(0.0,0.5),(0.5,1.0)], p=8)
n = solver.n_interior
x = solver.node_coords[solver.interior]
u_d = torch.tensor(np.sin(np.pi*x), dtype=torch.float64)

# True control for this u_d: f = -u'' = pi^2 sin(pi x) at interior nodes
f_true_np = np.pi**2 * np.sin(np.pi*x)

print("alpha | state_err% | ctrl_rel_err% | obj_val | iters")
for alpha in [1e-2, 1e-4, 1e-6, 1e-8]:
    def objective(f):
        u = solver.solve(f)
        return 0.5*((u-u_d)**2).sum() + alpha*0.5*(f**2).sum()
    f = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([f], lr=1.0, max_iter=100, history_size=20,
                             tolerance_grad=1e-12, tolerance_change=1e-14)
    def closure():
        opt.zero_grad(); loss = objective(f); loss.backward(); return loss
    opt.step(closure)
    u = solver.solve(f).detach().numpy()
    f_np = f.detach().numpy()
    state_err = np.linalg.norm(u - u_d.numpy())/np.linalg.norm(u_d.numpy())*100
    ctrl_err = np.linalg.norm(f_np - f_true_np)/np.linalg.norm(f_true_np)*100
    obj = objective(f).item()
    print(f"{alpha:.0e} | {state_err:.3f} | {ctrl_err:.2f} | {obj:.4e}")
