import sys; sys.path.insert(0,'.')
import torch, numpy as np
from spectral_element import DifferentiableSEM1D

print("seed | state_err% | obj")
state_errs = []
for seed in range(10):
    torch.manual_seed(seed)
    solver = DifferentiableSEM1D([(0.0,0.5),(0.5,1.0)], p=8)
    n = solver.n_interior
    x = solver.node_coords[solver.interior]
    u_d = torch.tensor(np.sin(np.pi*x), dtype=torch.float64)
    alpha = 1e-6
    def objective(f):
        u = solver.solve(f)
        return 0.5*((u-u_d)**2).sum() + alpha*0.5*(f**2).sum()
    f = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([f], lr=1.0, max_iter=50, history_size=10,
                             tolerance_grad=1e-12, tolerance_change=1e-14)
    def closure():
        opt.zero_grad(); loss = objective(f); loss.backward(); return loss
    opt.step(closure)
    u = solver.solve(f).detach().numpy()
    se = np.linalg.norm(u-u_d.numpy())/np.linalg.norm(u_d.numpy())*100
    state_errs.append(se)
    print(f"{seed} | {se:.4f} | {objective(f).item():.4e}")

state_errs = np.array(state_errs)
print(f"\nMean state error: {state_errs.mean():.4f} +/- {state_errs.std():.4f}%")
print(f"Min: {state_errs.min():.4f}%, Max: {state_errs.max():.4f}%")
