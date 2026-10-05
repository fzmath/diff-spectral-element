"""Debug Newton convergence for 2D Burgers."""
import torch, numpy as np
from sem_lib import build_2d, to_torch, full_differential_2d

torch.set_default_dtype(torch.float64)
p, E = 4, 2
K, M, x, y, nodes_1d, n1d, Dg = build_2d(p, E)
Kt, Mt = to_torch(K, M)
Dx, Dy = full_differential_2d(n1d, Dg, list(range(K.shape[0])))
Dxt, Dyt = torch.tensor(Dx), torch.tensor(Dy)

def residual(u, f, nu):
    return nu*Kt@u + Mt@(u*(Dxt@u)) + Mt@(u*(Dyt@u)) - Mt@f

def jacobian(u, nu):
    return nu*Kt + Mt@torch.diag(Dxt@u) + Mt@torch.diag(u)@Dxt + Mt@torch.diag(Dyt@u) + Mt@torch.diag(u)@Dyt

f0 = torch.tensor(0.05*np.sin(torch.tensor(np.pi*x))*np.sin(torch.tensor(np.pi*y)))
# continuation from nu=1.0 down to 0.05, full Newton steps
u = torch.zeros_like(f0)
for nu in [1.0, 0.5, 0.1, 0.05]:
    print(f"--- nu={nu} (continuation) ---")
    for it in range(30):
        R = residual(u, f0, nu)
        rn = float(torch.norm(R))
        if it < 4 or it % 5 == 0:
            print(f"  it={it:2d} res={rn:.3e} |u|max={float(u.abs().max()):.3f}")
        if rn < 1e-12: break
        J = jacobian(u, nu)
        du = torch.linalg.solve(J, -R)
        u = u + du
