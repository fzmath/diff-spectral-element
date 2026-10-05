"""
verify_all.py — Re-runs all retained numerical experiments of the manuscript
with the unified DSEM library (sem_lib, GLL quadrature mass matrix), so that
every table in the paper shares one discretization convention and all raw
numbers are archived as JSON.

Experiments (manuscript sections):
  5.1  1D Poisson p-refinement + gradient accuracy + alpha sensitivity
  5.2  2D heat conduction
  5.4  L-shaped domain
  5.6  3D Poisson
  5.7  hand-coded adjoint vs AD (single gradient, timing)
  5.8  SEM vs linear FEM
  5.9  matrix-free CG gradient accuracy
  5.10 1D Burgers gradient accuracy across nu
"""
import json
import os
import time
import numpy as np
import torch

from sem_lib import (gll_points, diff_matrix_gll, build_1d, build_2d,
                     build_lshape, build_3d, to_torch, fd_gradient_central)

torch.set_default_dtype(torch.float64)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

ALPHA = 1e-6


def save(name, data):
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[saved] {name}")


# ------------------------------------------------------------ 5.1 1D Poisson
def run_1d():
    rows_p = []
    rows_g = []
    for p in [2, 4, 6, 8, 10, 16]:
        K, M, nodes, Dg = build_1d(p, 2)
        Kt, Mt = to_torch(K, M)
        n = K.shape[0]
        x = torch.tensor(nodes[1:-1])
        u_ex = torch.sin(torch.pi * x)
        f = torch.pi ** 2 * u_ex          # -u'' = pi^2 sin(pi x)
        u = torch.linalg.solve(Kt, Mt @ f)
        err = float(torch.norm(u - u_ex) / torch.norm(u_ex))
        kap = float(np.linalg.cond(K))
        rows_p.append({"p": p, "n_int": n, "L2_rel": err, "kappa_K": kap})
        if p <= 10:
            # gradient accuracy via AD vs central FD
            def obj(ff):
                uu = torch.linalg.solve(Kt, Mt @ ff)
                return 0.5 * torch.sum((uu - u_ex) ** 2) + 0.5 * ALPHA * torch.sum(ff ** 2)
            f0 = f.clone().requires_grad_(True)
            g_ad = torch.autograd.grad(obj(f0), f0)[0].detach()
            g_fd = fd_gradient_central(obj, f, 1e-6)
            gerr = float(torch.norm(g_ad - g_fd) / torch.norm(g_fd))
            rows_g.append({"p": p, "n_int": n, "L2_rel": err, "grad_fd_err": gerr})
    # alpha sensitivity (p=8)
    K, M, nodes, Dg = build_1d(8, 2)
    Kt, Mt = to_torch(K, M)
    x = torch.tensor(nodes[1:-1])
    u_ex = torch.sin(torch.pi * x)
    f_true = torch.linalg.solve(Mt, Kt @ u_ex)
    rows_a = []
    for alpha in [1e-2, 1e-4, 1e-6, 1e-8]:
        from scipy.optimize import minimize

        def fun(fn):
            ff = torch.tensor(fn)
            uu = torch.linalg.solve(Kt, Mt @ ff)
            return float(0.5 * torch.sum((uu - u_ex) ** 2) + 0.5 * alpha * torch.sum(ff ** 2))

        def jac(fn):
            ff = torch.tensor(fn, requires_grad=True)
            uu = torch.linalg.solve(Kt, Mt @ ff)
            J = 0.5 * torch.sum((uu - u_ex) ** 2) + 0.5 * alpha * torch.sum(ff ** 2)
            return torch.autograd.grad(J, ff)[0].numpy()

        res = minimize(fun, np.zeros(15), jac=jac, method="L-BFGS-B",
                       options={"maxiter": 100, "gtol": 1e-9})
        f_opt = torch.tensor(res.x)
        u_opt = torch.linalg.solve(Kt, Mt @ f_opt)
        state_err = float(torch.norm(u_opt - u_ex) / torch.norm(u_ex)) * 100
        ctrl_err = float(torch.norm(f_opt - f_true) / torch.norm(f_true)) * 100
        rows_a.append({"alpha": alpha, "state_err_pct": round(state_err, 3),
                       "ctrl_err_pct": round(ctrl_err, 3), "objective": res.fun})
    save("verify_1d.json", {"p_refinement": rows_p, "gradient_accuracy": rows_g,
                            "alpha_sensitivity": rows_a})
    # 1D optimization at alpha=1e-6: iterations + seed variance
    from scipy.optimize import minimize
    K, M, nodes, Dg = build_1d(8, 2)
    Kt, Mt = to_torch(K, M)
    x = torch.tensor(nodes[1:-1])
    u_d = torch.sin(torch.pi * x)
    alpha = 1e-6
    state_errors = []
    iters_list = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        f0 = rng.normal(0, 0.1, K.shape[0])

        def fun(fn):
            ff = torch.tensor(fn)
            uu = torch.linalg.solve(Kt, Mt @ ff)
            return float(0.5 * torch.sum((uu - u_d) ** 2) + 0.5 * alpha * torch.sum(ff ** 2))

        def jac(fn):
            ff = torch.tensor(fn, requires_grad=True)
            uu = torch.linalg.solve(Kt, Mt @ ff)
            J = 0.5 * torch.sum((uu - u_d) ** 2) + 0.5 * alpha * torch.sum(ff ** 2)
            return torch.autograd.grad(J, ff)[0].numpy()

        res = minimize(fun, f0, jac=jac, method="L-BFGS-B",
                       options={"maxiter": 100, "gtol": 1e-8})
        ff = torch.tensor(res.x)
        uu = torch.linalg.solve(Kt, Mt @ ff)
        state_errors.append(float(torch.norm(uu - u_d) / torch.norm(u_d)) * 100)
        iters_list.append(res.nit)
    save("verify_1d_opt.json", {
        "alpha": alpha, "iters": iters_list,
        "state_err_pct": [round(s, 4) for s in state_errors],
        "mean_state_err_pct": round(float(np.mean(state_errors)), 4),
        "std_state_err_pct": round(float(np.std(state_errors)), 6)})


# ------------------------------------------------------------ 5.2 2D Poisson
def run_2d():
    rows = []
    for p in [4, 8]:
        K, M, x, y, nodes_1d, n1d, Dg = build_2d(p, 2)
        Kt, Mt = to_torch(K, M)
        n = K.shape[0]
        u_ex = torch.sin(torch.pi * torch.tensor(x)) * torch.sin(torch.pi * torch.tensor(y))
        f = 2.0 * torch.pi ** 2 * u_ex
        u = torch.linalg.solve(Kt, Mt @ f)
        err = float(torch.norm(u - u_ex) / torch.norm(u_ex))
        kap = float(np.linalg.cond(K))
        gerr = None
        if p == 8:
            # AD vs FD gradient on the tracking objective
            u_d = u_ex
            def obj(ff):
                uu = torch.linalg.solve(Kt, Mt @ ff)
                return 0.5 * torch.sum((uu - u_d) ** 2) + 0.5 * ALPHA * torch.sum(ff ** 2)
            f0 = f.clone().requires_grad_(True)
            g_ad = torch.autograd.grad(obj(f0), f0)[0].detach()
            g_fd = fd_gradient_central(obj, f, 1e-6)
            gerr = float(torch.norm(g_ad - g_fd) / torch.norm(g_fd))
        rows.append({"p": p, "n_int": n, "L2_rel": err, "kappa_K": kap,
                     "grad_fd_err": gerr})
    save("verify_2d.json", rows)
    # decoupled vector-valued system (Sec. 5.5): u_y with 2pi frequency
    p = 8
    K, M, x, y, nodes_1d, n1d, Dg = build_2d(p, 2)
    Kt, Mt = to_torch(K, M)
    u_y = torch.sin(2 * torch.pi * torch.tensor(x)) * torch.sin(torch.pi * torch.tensor(y))
    f_y = 5.0 * torch.pi ** 2 * u_y
    uy = torch.linalg.solve(Kt, Mt @ f_y)
    err_y = float(torch.norm(uy - u_y) / torch.norm(u_y))
    # combined gradient over both components
    u_x = torch.sin(torch.pi * torch.tensor(x)) * torch.sin(torch.pi * torch.tensor(y))
    u_dv = torch.cat([u_x, u_y])
    Mv = torch.block_diag(Mt, Mt); Kv = torch.block_diag(Kt, Kt)
    def obj_v(ff):
        uu = torch.linalg.solve(Kv, Mv @ ff)
        return 0.5 * torch.sum((uu - u_dv) ** 2) + 0.5 * ALPHA * torch.sum(ff ** 2)
    fv = torch.cat([2.0 * torch.pi ** 2 * u_x, f_y])
    fv_in = fv.clone().requires_grad_(True)
    gv_ad = torch.autograd.grad(obj_v(fv_in), fv_in)[0].detach()
    gv_fd = fd_gradient_central(obj_v, fv, 1e-6)
    gerr_v = float(torch.norm(gv_ad - gv_fd) / torch.norm(gv_fd))
    save("verify_vector.json", {"u_y_L2_rel": err_y,
                                "combined_grad_fd_err": gerr_v,
                                "n_int_per_comp": int(K.shape[0])})


# ------------------------------------------------------------ 5.4 L-shaped
def run_lshape():
    rows = []
    for p in [2, 4, 6, 8]:
        K, M, x, y = build_lshape(p)
        Kt, Mt = to_torch(K, M)
        n = K.shape[0]
        u_ex = torch.sin(torch.pi * torch.tensor(x)) * torch.sin(torch.pi * torch.tensor(y))
        f = 2.0 * torch.pi ** 2 * u_ex
        u = torch.linalg.solve(Kt, Mt @ f)
        err = float(torch.norm(u - u_ex) / torch.norm(u_ex))
        rows.append({"p": p, "n_int": n, "L2_rel": err})
    save("verify_lshape.json", rows)


# ------------------------------------------------------------ 5.6 3D Poisson
def run_3d():
    rows = []
    for p in [2, 3, 4]:
        K, M, x, y, z = build_3d(p, 2)
        Kt, Mt = to_torch(K, M)
        n = K.shape[0]
        u_ex = (torch.sin(torch.pi * torch.tensor(x)) *
                torch.sin(torch.pi * torch.tensor(y)) *
                torch.sin(torch.pi * torch.tensor(z)))
        f = 3.0 * torch.pi ** 2 * u_ex
        u = torch.linalg.solve(Kt, Mt @ f)
        err = float(torch.norm(u - u_ex) / torch.norm(u_ex))
        kap = float(np.linalg.cond(K))
        gerr = None
        if p == 4:
            def obj(ff):
                uu = torch.linalg.solve(Kt, Mt @ ff)
                return 0.5 * torch.sum((uu - u_ex) ** 2) + 0.5 * ALPHA * torch.sum(ff ** 2)
            f0 = f.clone().requires_grad_(True)
            g_ad = torch.autograd.grad(obj(f0), f0)[0].detach()
            g_fd = fd_gradient_central(obj, f, 1e-6)
            gerr = float(torch.norm(g_ad - g_fd) / torch.norm(g_fd))
        rows.append({"p": p, "n_int": n, "L2_rel": err, "kappa_K": kap,
                     "grad_fd_err": gerr})
    save("verify_3d.json", rows)
    # 3D optimization (alpha=1e-6, p=4) and 4^3 mesh with p=2
    from scipy.optimize import minimize
    K, M, x, y, z = build_3d(4, 2)
    Kt, Mt = to_torch(K, M)
    u_d = (torch.sin(torch.pi * torch.tensor(x)) *
           torch.sin(torch.pi * torch.tensor(y)) *
           torch.sin(torch.pi * torch.tensor(z)))
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
    # 4^3 mesh, p=2
    K4, M4, x4, y4, z4 = build_3d(2, 4)
    u_d4 = (torch.sin(torch.pi * torch.tensor(x4)) *
            torch.sin(torch.pi * torch.tensor(y4)) *
            torch.sin(torch.pi * torch.tensor(z4)))
    f4 = 3.0 * torch.pi ** 2 * u_d4
    u4 = torch.linalg.solve(torch.tensor(K4), torch.tensor(M4) @ f4)
    err4 = float(torch.norm(u4 - u_d4) / torch.norm(u_d4))
    save("verify_3d_opt.json", {"opt_iters": int(res.nit),
                                "opt_state_err_pct": round(state_err, 4),
                                "mesh4_L2_rel": err4,
                                "mesh4_n_int": int(K4.shape[0])})


# ------------------------------------------------- 5.7 hand adjoint vs AD
def run_adjoint():
    K, M, nodes, Dg = build_1d(8, 2)
    Kt, Mt = to_torch(K, M)
    n = K.shape[0]
    x = torch.tensor(nodes[1:-1])
    u_d = torch.sin(torch.pi * x)
    f = torch.zeros(n)

    def obj(ff):
        uu = torch.linalg.solve(Kt, Mt @ ff)
        return 0.5 * torch.sum((uu - u_d) ** 2) + 0.5 * ALPHA * torch.sum(ff ** 2)

    # AD timing
    t0 = time.perf_counter()
    for _ in range(50):
        ff = f.clone().requires_grad_(True)
        g_ad = torch.autograd.grad(obj(ff), ff)[0]
    t_ad = (time.perf_counter() - t0) / 50 * 1e3

    # hand adjoint timing
    def grad_hand(ff):
        uu = torch.linalg.solve(Kt, Mt @ ff)
        lam = torch.linalg.solve(Kt.T, -(uu - u_d))
        return ALPHA * ff - Mt @ lam

    t0 = time.perf_counter()
    for _ in range(50):
        g_hand = grad_hand(f)
    t_hand = (time.perf_counter() - t0) / 50 * 1e3

    gerr = float(torch.norm(g_ad - g_hand) / torch.norm(g_hand))
    save("verify_adjoint.json", {"grad_rel_err": gerr,
                                 "ad_ms_per_eval": round(t_ad, 4),
                                 "hand_ms_per_eval": round(t_hand, 4),
                                 "ratio": round(t_ad / t_hand, 2)})


# ------------------------------------------------------------ 5.8 SEM vs FEM
def run_fem_compare():
    # 1D linear FEM, n interior nodes
    def fem_solve(n):
        h = 1.0 / (n + 1)
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] += 2.0 / h
            if i > 0:
                K[i, i - 1] -= 1.0 / h
                K[i - 1, i] -= 1.0 / h
        x = np.linspace(h, 1 - h, n)
        f = np.pi ** 2 * np.sin(np.pi * x)
        b = h * f
        u = np.linalg.solve(K, b)
        u_ex = np.sin(np.pi * x)
        return float(np.sqrt(np.mean((u - u_ex) ** 2)))
    fem_err = fem_solve(40)
    # SEM p=4, E=10
    K, M, nodes, Dg = build_1d(4, 10)
    Kt, Mt = to_torch(K, M)
    x = torch.tensor(nodes[1:-1])
    u_ex = torch.sin(torch.pi * x)
    f = torch.pi ** 2 * u_ex
    u = torch.linalg.solve(Kt, Mt @ f)
    sem_err = float(torch.sqrt(torch.mean((u - u_ex) ** 2)))
    save("verify_fem.json", {"fem_n40_rms": fem_err, "sem_p4_E10_rms": sem_err,
                             "sem_n_int": int(K.shape[0])})


# ------------------------------------------------- 5.9 matrix-free CG accuracy
def run_cg_accuracy():
    from expB_matrixfree_scaling import ElementWiseMatvec, cg_matrixfree
    rows = []
    for E in [20, 50, 100, 200]:
        K, M, nodes, Dg = build_1d(4, E)
        n = K.shape[0]
        b = M @ np.ones(n)
        A = ElementWiseMatvec(E, 4)
        x_cg, it = cg_matrixfree(A, b, rtol=1e-10)
        x_dir = np.linalg.solve(K, b)
        err = float(np.linalg.norm(x_cg - x_dir) / np.linalg.norm(x_dir))
        rows.append({"n_int": n, "cg_iters": it, "sol_rel_err": err})
    save("verify_cg.json", rows)


# -------------------------------------------------------- 5.10 1D Burgers
def run_burgers_1d():
    p, E = 16, 2
    K, M, nodes, Dg = build_1d(p, E)
    Kt, Mt, Dgt = to_torch(K, M), None, None
    Kt, Mt = to_torch(K, M)
    Dt = torch.tensor(Dg)
    n = K.shape[0]

    def residual(u, f, nu):
        return nu * Kt @ u + Mt @ (u * (Dt @ u)) - Mt @ f

    def jacobian(u, nu):
        return nu * Kt + Mt @ torch.diag(Dt @ u) + Mt @ torch.diag(u) @ Dt

    def newton(f, nu, u0=None):
        u = torch.zeros_like(f) if u0 is None else u0.clone()
        for it in range(100):
            R = residual(u, f, nu)
            if float(torch.norm(R)) < 1e-12:
                return u, it
            J = jacobian(u, nu)
            du = torch.linalg.solve(J, -R)
            u = u + du
        return u, 100

    class Burgers1D(torch.autograd.Function):
        @staticmethod
        def forward(ctx, f, nu):
            with torch.no_grad():
                u, it = newton(f, nu)
            ctx.save_for_backward(u)
            ctx.nu = nu
            return u

        @staticmethod
        def backward(ctx, grad_u):
            (u,) = ctx.saved_tensors
            J = jacobian(u, ctx.nu)
            lam = torch.linalg.solve(J.T, grad_u)
            return Mt @ lam, None

    x = torch.tensor(nodes[1:-1])
    u_d = torch.sin(torch.pi * x)
    f0 = torch.tensor(0.5 * np.sin(torch.pi * nodes[1:-1]))

    rows = []
    u_prev = None
    for nu in [1e-1, 1e-2, 1e-3]:
        u, it = newton(f0, nu, u0=u_prev)
        conv = float(torch.norm(residual(u, f0, nu))) < 1e-10
        if not conv:
            rows.append({"nu": nu, "newton": "not converged", "grad_fd_err": None})
            continue
        u_prev = u

        def obj(ff):
            uu = Burgers1D.apply(ff, nu)
            return 0.5 * torch.sum((uu - u_d) ** 2) + 0.5 * ALPHA * torch.sum(ff ** 2)

        fg = f0.clone().requires_grad_(True)
        g_ad = torch.autograd.grad(obj(fg), fg)[0].detach()
        g_fd = fd_gradient_central(obj, f0, 1e-6)
        err = float(torch.norm(g_ad - g_fd) / torch.norm(g_fd))
        rows.append({"nu": nu, "newton": "converged", "newton_iters": int(it),
                     "grad_fd_err": err})
        print(f"burgers nu={nu}: iters={it}, grad_fd_err={err:.2e}")
    save("verify_burgers.json", rows)


if __name__ == "__main__":
    run_1d()
    run_2d()
    run_lshape()
    run_3d()
    run_adjoint()
    run_fem_compare()
    run_cg_accuracy()
    run_burgers_1d()
