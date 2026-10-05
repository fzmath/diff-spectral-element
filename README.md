# DSEM — Differentiable Spectral Element Software

PyTorch-based differentiable spectral element method (SEM) for
PDE-constrained inverse and optimization problems.

This repository contains the exact code and data used for the manuscript

> **Differentiable spectral element software for PDE-constrained inverse
> problems: automatic adjoint consistency and reduced Hessian conditioning**
> Fuchang Wang, Huirong Cao. Computer Physics Communications (submitted).

## What it does

A unified, high-order spectral element library (`code/sem_lib.py`) that
supports:

- 1D / 2D / 3D tensor-product GLL spectral elements (diagonal GLL mass
  matrix, strong Dirichlet conditions);
- forward solves, reverse-mode automatic differentiation (PyTorch), and
  hand-coded discrete adjoints, with AD–adjoint consistency verified to
  machine precision;
- dense, sparse, and element-wise matrix-free solution modes;
- PDE-constrained optimization (L-BFGS) with Tikhonov regularization,
  including a reduced-Hessian conditioning analysis;
- representative nonlinear steady residuals (steady Burgers, 2D nonlinear
  diffusion).

It is intended as a *reusable differentiable high-order PDE optimization
architecture*: for a new residual, one supplies the element residual and
the framework obtains gradients automatically, without deriving or
maintaining a problem-specific continuous or discrete adjoint.

## Layout

| Path | Contents |
|---|---|
| `code/` | Final unified-library scripts (reproduce every table and figure) |
| `results/` | Machine-readable JSON outputs used by the manuscript |
| `figures/` | Figure scripts and the vector figures (PDF) |
| `oms_draft/` | Earlier draft scripts (consistent-mass formulation, OMS-era) retained for provenance only — **not** used in the CPC manuscript |
| `CITATION.cff` | Citation metadata (Zenodo DOI will be added at release) |
| `environment.yml` | Conda environment (Python 3.11, PyTorch ≥ 2.1, NumPy 2.x) |

## Reproduction

```
conda env create -f environment.yml
conda activate dsem-cpc
cd code
python verify_all.py              # retained verification tables
python verify_lshape_opt.py       # L-shaped-domain optimization
python verify_hessian_cond.py     # reduced-Hessian conditioning (Table)
python calibrate_B_scaling.py     # mu_max(B) ~ p^4 h^-2 calibration (Thm 2)
python expA_2d_optimization_benchmark.py   # AD vs hand adjoint vs FD
python expB_matrixfree_scaling.py          # matrix-free scaling, log-log fits
python expC_2d_burgers.py                  # 2D steady nonlinear residual
cd ../figures && python make_figures.py    # all figures from results/*.json
```

All arithmetic is float64 (CPU). Timings in the paper were measured on an
Intel Core i7-8850H, 32 GB RAM, single process, and are reported for that
environment.

## Archiving

The exact commit behind the manuscript is tagged as Release `v1.0.0` and
deposited to Zenodo (DOI to be registered at release). MIT license.

## Citation

See `CITATION.cff` (BibTeX-friendly fields). When the Zenodo DOI is
minted, cite the Zenodo record; until then, cite the GitHub repository.
