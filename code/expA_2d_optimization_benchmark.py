"""
expA_2d_optimization_benchmark.py
Experiment A (CPC manuscript Sec. 5.3): complete L-BFGS optimization on the
2D Poisson control problem comparing AD, hand-coded discrete adjoint, and
central finite differences, plus FD step-size sensitivity.

Outputs: results/expA_opt_benchmark.json, results/expA_fd_sensitivity.json
"""
import json
import time
import os
import numpy as np
import torch

from sem_lib import build_2d, to_torch, fd_gradient_central, l2_rel

torch.set_default_dtype(torch.float64)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ problem
p, E = 4, 2
ALPHA = 1e-6
K, M, x, y, nodes_1d, n1d, Dg = build_2d(p, E)
Kt, Mt = to_torch(K, M)
n = K.shape[0]
u_d = torch.sin(torch.tensor(np.pi * x)) * torch.sin(torch.tensor(np.pi * y))
# Discrete "true" control: the source that reproduces u_d exactly (alpha->0 limit)
f_true = torch.linalg.solve(Mt, Kt @ u_d)


def objective(f):
    u = torch.linalg.solve(Kt, Mt @ f)
    return 0.5 * torch.sum((u - u_d) ** 2) + 0.5 * ALPHA * torch.sum(f ** 2)


def solve_state(f):
    return torch.linalg.solve(Kt, Mt @ f)


def grad_ad(f):
    f = f.clone().requires_grad_(True)
    J = objective(f)
    g = torch.autograd.grad(J, f)[0]
    return g.detach()


def grad_hand_adjoint(f):
    """Discrete adjoint: K^T lambda = -(u-u_d), grad = alpha*f - M*lambda."""
    u = solve_state(f)
    lam = torch.linalg.solve(Kt.T, -(u - u_d))
    return ALPHA * f - Mt @ lam


def run_lbfgs(grad_fn, x0, label):
    """Run L-BFGS via scipy with a common interface for all gradient sources."""
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    njev = [0]
    history = {"iters": [], "objectives": [], "grad_norms": []}

    def jac(f_np):
        njev[0] += 1
        return grad_fn(torch.tensor(f_np)).numpy()

    def fun(f_np):
        return float(objective(torch.tensor(f_np)))

    def callback(xk):
        fk = torch.tensor(xk)
        history["iters"].append(len(history["iters"]) + 1)
        history["objectives"].append(float(objective(fk)))
        history["grad_norms"].append(float(torch.norm(grad_fn(fk))))

    res = minimize(fun, x0, jac=jac, method="L-BFGS-B", callback=callback,
                   options={"maxiter": 300, "gtol": 1e-6, "ftol": 1e-16})
    elapsed = time.perf_counter() - t0
    f_opt = torch.tensor(res.x)
    u_opt = solve_state(f_opt)
    g_opt = grad_fn(f_opt)
    state_err = float(torch.norm(u_opt - u_d) / torch.norm(u_d)) * 100.0
    ctrl_err = float(torch.norm(f_opt - f_true) / torch.norm(f_true)) * 100.0
    return {
        "label": label,
        "iterations": int(res.nit),
        "gradient_calls": njev[0],
        "total_time_s": round(elapsed, 4),
        "final_objective": float(res.fun),
        "gradient_norm": float(torch.norm(g_opt)),
        "state_error_pct": round(state_err, 6),
        "control_error_pct": round(ctrl_err, 4),
        "converged": bool(res.success),
        "history": history,
    }


def main():
    x0 = np.zeros(n)
    results = [
        run_lbfgs(grad_ad, x0, "AD"),
        run_lbfgs(grad_hand_adjoint, x0, "Hand-coded adjoint"),
        run_lbfgs(lambda f: fd_gradient_central(objective, f, 1e-6), x0, "Finite difference"),
    ]
    with open(os.path.join(OUT, "expA_opt_benchmark.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # convergence trajectories for figures (objective and gradient norm)
    traj = {r["label"]: r.pop("history") for r in results}
    with open(os.path.join(OUT, "expA_trajectory.json"), "w") as fh:
        json.dump(traj, fh, indent=2)
    with open(os.path.join(OUT, "expA_opt_benchmark.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # FD step-size sensitivity (single gradient at a fixed reference point)
    rng = np.random.default_rng(0)
    f_ref = torch.tensor(rng.random(n) * 0.1)
    ad_g = grad_ad(f_ref)
    fd_table = []
    for hfd in [1e-4, 1e-5, 1e-6, 1e-7]:
        fd_g = fd_gradient_central(objective, f_ref, hfd)
        err = float(torch.norm(ad_g - fd_g) / torch.norm(fd_g))
        fd_table.append({"h_FD": hfd, "rel_gradient_error": err})
    with open(os.path.join(OUT, "expA_fd_sensitivity.json"), "w") as fh:
        json.dump(fd_table, fh, indent=2)

    print(json.dumps(results, indent=2))
    print(json.dumps(fd_table, indent=2))


if __name__ == "__main__":
    main()
