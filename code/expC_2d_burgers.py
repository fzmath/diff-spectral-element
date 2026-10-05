"""
expC_2d_burgers.py
Experiment C (CPC manuscript Sec. 5.11): 2D steady viscous Burgers-type
residual  -nu Δu + u u_x + u u_y = f  on (0,1)^2, p=4, 2x2 elements.
The Newton solve is wrapped in a custom PyTorch autograd Function; the
backward pass solves the adjoint Jacobian system via the implicit function
theorem. AD gradient is compared against central finite differences.

Outputs: results/expC_2d_burgers.json
"""
import json
import os
import numpy as np
import torch

from sem_lib import build_2d, to_torch, full_differential_2d, fd_gradient_central

torch.set_default_dtype(torch.float64)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

p, E = 4, 4
ALPHA = 1e-6
K, M, x, y, nodes_1d, n1d, Dg = build_2d(p, E)
Kt, Mt = to_torch(K, M)
Dx, Dy = full_differential_2d(n1d, Dg, list(range(K.shape[0])))
Dxt, Dyt = torch.tensor(Dx), torch.tensor(Dy)
n = K.shape[0]

u_d = torch.sin(torch.tensor(np.pi * x)) * torch.sin(torch.tensor(np.pi * y))


def residual(u, f, nu):
    """Discrete residual R(u) = nu K u + M(u u_x + u u_y) - M f."""
    return (nu * Kt @ u + Mt @ (u * (Dxt @ u)) + Mt @ (u * (Dyt @ u)) - Mt @ f)


def jacobian(u, nu):
    """Jacobian J(u) = nu K + M diag(Dx u) + M diag(u) Dx + M diag(Dy u) + M diag(u) Dy."""
    return (nu * Kt + Mt @ torch.diag(Dxt @ u) + Mt @ torch.diag(u) @ Dxt
            + Mt @ torch.diag(Dyt @ u) + Mt @ torch.diag(u) @ Dyt)


def newton_solve(f, nu, u0=None, tol=1e-13, maxit=100):
    """Newton solve for R(u,f)=0 with backtracking damping on residual increase.
    u0 = initial guess (default zero)."""
    u = torch.zeros_like(f) if u0 is None else u0.clone()
    for it in range(maxit):
        R = residual(u, f, nu)
        rn = float(torch.norm(R))
        if rn < tol:
            return u, it
        J = jacobian(u, nu)
        du = torch.linalg.solve(J, -R)
        alpha = 1.0
        u_try = u + du
        while float(torch.norm(residual(u_try, f, nu))) > rn and alpha > 1e-4:
            alpha *= 0.5
            u_try = u + alpha * du
        u = u_try
        if torch.norm(du) < 1e-13 * (torch.norm(u) + 1.0):
            return u, it + 1
    return u, maxit


class BurgersSolve(torch.autograd.Function):
    """Differentiable Newton solve: forward = Newton, backward = adjoint system."""

    @staticmethod
    def forward(ctx, f, nu):
        with torch.no_grad():
            u, it = newton_solve(f, nu)
        ctx.save_for_backward(u)
        ctx.nu = nu
        ctx.iters = it
        return u

    @staticmethod
    def backward(ctx, grad_u):
        (u,) = ctx.saved_tensors
        J = jacobian(u, ctx.nu)                       # solve J^T lambda = grad_u
        lam = torch.linalg.solve(J.T, grad_u)
        g = Mt @ lam                                  # du/df = J^{-1} M  =>  (du/df)^T = M J^{-T}
        return g, None


def objective(f, nu):
    u = BurgersSolve.apply(f, nu)
    return 0.5 * torch.sum((u - u_d) ** 2) + 0.5 * ALPHA * torch.sum(f ** 2)


def main():
    rng = np.random.default_rng(1)
    f0 = torch.tensor(0.05 * np.sin(np.pi * np.asarray(x)) * np.sin(np.pi * np.asarray(y)))
    results = []
    u_prev = None
    for nu in [1e-1, 5e-2]:
        # continuation: start from the converged state of the previous (larger) nu
        u, it = newton_solve(f0, nu, u0=u_prev)
        converged = float(torch.norm(residual(u, f0, nu))) < 1e-10
        if not converged:
            print(f"nu={nu}: Newton did NOT converge (res={float(torch.norm(residual(u,f0,nu))):.2e})")
            results.append({"nu": nu, "newton": "not converged", "newton_iters": it,
                            "grad_fd_err": None})
            continue
        u_prev = u
        # AD gradient through the custom autograd Function
        f = f0.clone().requires_grad_(True)
        J = objective(f, nu)
        g_ad = torch.autograd.grad(J, f)[0].detach()
        # central finite differences (same objective path)
        g_fd = fd_gradient_central(lambda ff: objective(ff, nu), f0, 1e-6)
        err = float(torch.norm(g_ad - g_fd) / torch.norm(g_fd))
        print(f"nu={nu}: newton_iters={it}, grad_fd_err={err:.3e}")
        results.append({"nu": nu, "newton": "converged", "newton_iters": int(it),
                        "grad_fd_err": err})

    with open(os.path.join(OUT, "expC_2d_burgers.json"), "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
