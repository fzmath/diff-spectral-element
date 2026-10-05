"""Supplementary: L-shaped domain optimization (CPC manuscript Sec. 5.4)."""
import json, os
import numpy as np, torch
from sem_lib import build_lshape, to_torch
torch.set_default_dtype(torch.float64)
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
p = 8
K, M, x, y = build_lshape(p)
Kt, Mt = to_torch(K, M)
u_d = torch.sin(torch.pi * torch.tensor(x)) * torch.sin(torch.pi * torch.tensor(y))
alpha = 1e-6

def fun(fn):
    ff = torch.tensor(fn)
    uu = torch.linalg.solve(Kt, Mt @ ff)
    return float(0.5 * torch.sum((uu - u_d) ** 2) + 0.5 * alpha * torch.sum(ff ** 2))

def jac(fn):
    ff = torch.tensor(fn, requires_grad=True)
    uu = torch.linalg.solve(Kt, Mt @ ff)
    J = 0.5 * torch.sum((uu - u_d) ** 2) + 0.5 * alpha * torch.sum(ff ** 2)
    return torch.autograd.grad(J, ff)[0].numpy()

res = minimize(fun, np.zeros(K.shape[0]), jac=jac, method="L-BFGS-B",
               options={"maxiter": 100, "gtol": 1e-8})
uu = torch.linalg.solve(Kt, Mt @ torch.tensor(res.x))
state_err = float(torch.norm(uu - u_d) / torch.norm(u_d)) * 100
out = {"iters": int(res.nit), "state_err_pct": round(state_err, 4),
       "n_int": int(K.shape[0]), "alpha": alpha}
with open(os.path.join(OUT, "verify_lshape_opt.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(out)
