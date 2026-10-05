"""
make_figures.py — regenerates all manuscript figures from results/*.json
into PDF (matplotlib PdfPages, one page per figure).

Figures:
  fig1_lbfgs_convergence.pdf   objective vs L-BFGS iteration (log scale)
  fig2_gradient_norm.pdf       gradient norm vs L-BFGS iteration (log scale)
  fig3_matrixfree_scaling.pdf  log-log solver runtime vs n (dense/sparse/CG)
  fig4_fd_sensitivity.pdf      FD step-size sensitivity (log-log)
  fig5_hessian_cond.pdf        reduced-Hessian condition number vs kappa(K)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def make_all():
    with PdfPages(os.path.join(FIG, "fig1_lbfgs_convergence.pdf")) as pdf:
        traj = load("expA_trajectory.json")
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for label in ["AD", "Hand-coded adjoint", "Finite difference"]:
            t = traj[label]
            ax.semilogy(t["iters"], t["objectives"], marker="o", ms=3, lw=1.2, label=label)
        ax.set_xlabel("L-BFGS iteration")
        ax.set_ylabel("Objective $J$")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    with PdfPages(os.path.join(FIG, "fig2_gradient_norm.pdf")) as pdf:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for label in ["AD", "Hand-coded adjoint", "Finite difference"]:
            t = traj[label]
            ax.semilogy(t["iters"], t["grad_norms"], marker="s", ms=3, lw=1.2, label=label)
        ax.set_xlabel("L-BFGS iteration")
        ax.set_ylabel(r"$\|\nabla J\|_2$")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    with PdfPages(os.path.join(FIG, "fig3_matrixfree_scaling.pdf")) as pdf:
        rows = load("expB_scaling.json")
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        nn = [r["n_int"] for r in rows]
        for key, lbl, mk in [("dense_ms", "Dense direct", "o"),
                             ("sparse_ms", "Sparse direct", "s"),
                             ("cg_ms", "Matrix-free CG", "^")]:
            x = [r["n_int"] for r in rows if r[key] is not None]
            y = [r[key] for r in rows if r[key] is not None]
            ax.loglog(x, y, marker=mk, lw=1.2, label=lbl)
        ax.set_xlabel(r"Interior DOF $n$")
        ax.set_ylabel("Solve time (ms)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    with PdfPages(os.path.join(FIG, "fig4_fd_sensitivity.pdf")) as pdf:
        rows = load("expA_fd_sensitivity.json")
        h = [r["h_FD"] for r in rows]
        e = [r["rel_gradient_error"] for r in rows]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.loglog(h, e, marker="o", lw=1.4)
        ax.set_xlabel(r"FD step $h_{\mathrm{FD}}$")
        ax.set_ylabel(r"Relative AD$-$FD gradient error")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    with PdfPages(os.path.join(FIG, "fig5_hessian_cond.pdf")) as pdf:
        rows = load("verify_hessian_cond.json")
        p_rows = [r for r in rows if r["vary"] == "p"]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.loglog([r["kappa_K"] for r in p_rows], [r["kappa_H"] for r in p_rows],
                  marker="o", lw=1.4, label=r"$p=2,4,8,16$ ($h$ fixed)")
        ax.axhline(p_rows[-1]["kappa_H"], ls=":", color="gray",
                   label=r"regularization-dominated saturation ($\alpha=10^{-6}$)")
        ax.set_xlabel(r"$\kappa_2(K)$")
        ax.set_ylabel(r"$\kappa_2(H)$")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print("figures written to", FIG)


if __name__ == "__main__":
    make_all()
