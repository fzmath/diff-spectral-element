"""
verify_hessian_cond.py — reduced-Hessian condition number (CPC manuscript
Sec. 4.2 / Theorem 2). For the 1D Poisson control problem with Tikhonov
regularization, computes kappa(K) and kappa(H) for H = alpha I + M K^{-T} K^{-1} M,
and the effective scaling with p at fixed h and with h at fixed p.
Outputs: results/verify_hessian_cond.json
"""
import json, os
import numpy as np
from sem_lib import build_1d

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 1e-6


def kappa_hessian(p, E, alpha=ALPHA):
    K, M, nodes, Dg = build_1d(p, E)
    n = K.shape[0]
    Kinv = np.linalg.inv(K)
    H = alpha * np.eye(n) + M @ Kinv.T @ Kinv @ M
    return float(np.linalg.cond(K)), float(np.linalg.cond(H)), n


def main():
    rows = []
    # p-refinement at fixed h (E = 2 elements)
    for p in [2, 4, 8, 16]:
        kapK, kapH, n = kappa_hessian(p, 2)
        rows.append({"vary": "p", "p": p, "E": 2, "n_int": n,
                     "kappa_K": round(kapK, 2), "kappa_H": round(kapH, 2)})
        print(f"p={p:2d}  kappa(K)={kapK:10.2f}  kappa(H)={kapH:12.2f}")
    # h-refinement at fixed p (E elements), log2-halving
    for E in [2, 4, 8, 16]:
        kapK, kapH, n = kappa_hessian(4, E)
        rows.append({"vary": "h", "p": 4, "E": E, "n_int": n,
                     "kappa_K": round(kapK, 2), "kappa_H": round(kapH, 2)})
        print(f"E={E:2d}  kappa(K)={kapK:10.2f}  kappa(H)={kapH:12.2f}")
    # alpha scan at fixed p, E
    for alpha in [1e-2, 1e-4, 1e-6, 1e-8]:
        kapK, kapH, n = kappa_hessian(8, 2, alpha)
        rows.append({"vary": "alpha", "alpha": alpha, "p": 8, "E": 2, "n_int": n,
                     "kappa_K": round(kapK, 2), "kappa_H": round(kapH, 2)})
        print(f"alpha={alpha:.0e}  kappa(K)={kapK:10.2f}  kappa(H)={kapH:12.2f}")
    with open(os.path.join(OUT, "verify_hessian_cond.json"), "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
