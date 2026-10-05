"""
sem_lib.py — Common utilities for the DSEM (Differentiable Spectral Element)
experiments in the Computer Physics Communications submission.

Conventions
-----------
- GLL quadrature mass matrix (diagonal, positive weights) is used throughout,
  consistent with Assumption (A2) of Theorem 2 in the manuscript.
- K, M are the interior stiffness/mass matrices after strong Dirichlet BC
  enforcement (boundary rows/columns removed).
- Node ordering in 2D/3D: tensor-product grid, first index fastest.
- All gradients use PyTorch float64 reverse-mode AD.
"""
import numpy as np
import torch
from scipy.special import legendre, eval_legendre

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- GLL basis
def gll_points(p):
    """Gauss--Lobatto--Legendre nodes and weights on [-1,1]."""
    P = legendre(p)
    dP = P.deriv()
    interior = np.sort(dP.roots.real)
    xi = np.concatenate([[-1.0], interior, [1.0]])
    w = np.array([2.0 / (p * (p + 1) * eval_legendre(p, x) ** 2) for x in xi])
    return xi, w


def diff_matrix_gll(xi):
    """Spectral differentiation matrix for the GLL nodes."""
    n = len(xi)
    c = 1.0 / np.array([np.prod([xi[j] - xi[k] for k in range(n) if k != j])
                        for j in range(n)])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (c[j] / c[i]) / (xi[i] - xi[j])
        D[i, i] = -np.sum(D[i, :])
    return D


# ------------------------------------------------------------------ 1D mesh
def build_1d(p, E, sparse=False):
    """1D Poisson on (0,1): E elements of degree p.

    If sparse=True, returns scipy CSC interior K, M (for large E) and Dg=None;
    otherwise returns numpy dense K_int, M_int, nodes_1d, and D_int.
    """
    xi, w = gll_points(p)
    Dref = diff_matrix_gll(xi)
    Je = 0.5 / E                      # half-width of a uniform element
    n1d = E * p + 1
    nodes_1d = np.zeros(n1d)
    for e in range(E):
        idx = list(range(e * p, e * p + p + 1))
        nodes_1d[idx] = e / E + Je * (xi + 1)

    if sparse:
        from scipy.sparse import lil_matrix
        K = lil_matrix((n1d, n1d))
        M = lil_matrix((n1d, n1d))
        for e in range(E):
            idx = list(range(e * p, e * p + p + 1))
            Ke = (1.0 / Je) * (Dref.T @ np.diag(w) @ Dref)
            Me = Je * np.diag(w)
            for a in range(p + 1):
                for b in range(p + 1):
                    K[idx[a], idx[b]] += Ke[a, b]
                    M[idx[a], idx[b]] += Me[a, b]
        interior = list(range(1, n1d - 1))
        return K[np.ix_(interior, interior)].tocsc(), \
               M[np.ix_(interior, interior)].tocsc(), nodes_1d, None

    K = np.zeros((n1d, n1d))
    M = np.zeros((n1d, n1d))
    Dg = np.zeros((n1d, n1d))          # global differentiation (assembled)
    for e in range(E):
        idx = list(range(e * p, e * p + p + 1))
        Ke = (1.0 / Je) * (Dref.T @ np.diag(w) @ Dref)
        Me = Je * np.diag(w)
        De = (1.0 / Je) * Dref
        for a in range(p + 1):
            for b in range(p + 1):
                K[idx[a], idx[b]] += Ke[a, b]
                M[idx[a], idx[b]] += Me[a, b]
                Dg[idx[a], idx[b]] += De[a, b]
    interior = list(range(1, n1d - 1))
    return (K[np.ix_(interior, interior)], M[np.ix_(interior, interior)],
            nodes_1d, Dg[np.ix_(interior, interior)])


# ------------------------------------------------------------------ 2D mesh
def build_2d(p, E):
    """2D Poisson on (0,1)^2: E x E elements of degree p.

    Returns K_int, M_int, interior x/y coordinates, nodes_1d, n1d, and
    global 1D differentiation matrix Dg (n1d x n1d, assembled).
    Node ordering j*n1d+i (x fastest).
    """
    xi, w = gll_points(p)
    Dref = diff_matrix_gll(xi)
    Je = 0.5 / E
    n1d = E * p + 1
    nodes_1d = np.zeros(n1d)
    for e in range(E):
        idx = list(range(e * p, e * p + p + 1))
        nodes_1d[idx] = e / E + Je * (xi + 1)

    M1d = Je * np.diag(w)                      # physical GLL mass (1D)
    K1d = (1.0 / Je) * (Dref.T @ np.diag(w) @ Dref)   # physical stiffness (1D)
    D1d = (1.0 / Je) * Dref                    # physical derivative (1D ref)

    n2d = n1d ** 2
    K = np.zeros((n2d, n2d))
    M = np.zeros((n2d, n2d))
    for ex in range(E):
        for ey in range(E):
            idx = []
            for j in range(p + 1):
                for i in range(p + 1):
                    idx.append((ey * p + j) * n1d + ex * p + i)
            Ke = np.kron(K1d, M1d) + np.kron(M1d, K1d)
            Me = np.kron(M1d, M1d)
            for a in range((p + 1) ** 2):
                for b in range((p + 1) ** 2):
                    K[idx[a], idx[b]] += Ke[a, b]
                    M[idx[a], idx[b]] += Me[a, b]

    boundary = set()
    for i in range(n1d):
        for j in range(n1d):
            x, y = nodes_1d[i], nodes_1d[j]
            if x < 1e-10 or x > 1 - 1e-10 or y < 1e-10 or y > 1 - 1e-10:
                boundary.add(j * n1d + i)
    interior = sorted(set(range(n2d)) - boundary)
    # interior node coordinates (x fastest ordering, matching K)
    xx = np.zeros(n2d); yy = np.zeros(n2d)
    for j in range(n1d):
        for i in range(n1d):
            xx[j * n1d + i] = nodes_1d[i]
            yy[j * n1d + i] = nodes_1d[j]

    # global 1D differentiation assembled over elements (interior only)
    Dg = np.zeros((n1d, n1d))
    for e in range(E):
        idx = list(range(e * p, e * p + p + 1))
        for a in range(p + 1):
            for b in range(p + 1):
                Dg[idx[a], idx[b]] += D1d[a, b]

    return (K[np.ix_(interior, interior)], M[np.ix_(interior, interior)],
            xx[interior], yy[interior], nodes_1d, n1d, Dg)


def full_differential_2d(n1d, Dg, interior):
    """Global 2D differentiation matrices Dx, Dy restricted to interior DOFs.

    Node ordering j*n1d+i (x fastest):  Dx = kron(I, Dg), Dy = kron(Dg, I).
    Returns Dx_int, Dy_int acting on interior-indexed vectors.
    """
    Dx = np.kron(np.eye(n1d), Dg)
    Dy = np.kron(Dg, np.eye(n1d))
    return Dx[np.ix_(interior, interior)], Dy[np.ix_(interior, interior)]


# ------------------------------------------------------------------ L-shape
def build_lshape(p):
    """L-shaped domain (0,2)x(0,1) U (0,1)x(1,2), 3 elements of degree p.

    Returns K_int, M_int, interior x/y coords. Reentrant-corner edges are
    strongly enforced as Dirichlet.
    """
    elems = [(0.0, 1.0, 0.0, 1.0), (1.0, 2.0, 0.0, 1.0), (0.0, 1.0, 1.0, 2.0)]
    xi, w = gll_points(p)
    Dref = diff_matrix_gll(xi)
    n_loc = p + 1

    node_map = {}
    coords = []
    elem_nodes = []
    for (x0, x1, y0, y1) in elems:
        Je = 0.5
        enodes = []
        for j in range(n_loc):
            for i in range(n_loc):
                x = 0.5 * (x1 - x0) * (xi[i] + 1) + x0
                y = 0.5 * (y1 - y0) * (xi[j] + 1) + y0
                key = (round(x, 12), round(y, 12))
                if key not in node_map:
                    node_map[key] = len(coords)
                    coords.append([x, y])
                enodes.append(node_map[key])
        elem_nodes.append(enodes)
    n_nodes = len(coords)
    coords = np.array(coords)

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    boundary = set()
    for i in range(n_nodes):
        x, y = coords[i]
        on_outer = (abs(x - x_min) < 1e-12 or abs(x - x_max) < 1e-12 or
                    abs(y - y_min) < 1e-12 or abs(y - y_max) < 1e-12)
        if on_outer:
            boundary.add(i)
            continue
        # reentrant-corner edges: x=1 (1<y<2) or y=1 (1<x<2)
        if (abs(x - 1.0) < 1e-10 and y > 1.0) or (abs(y - 1.0) < 1e-10 and x > 1.0):
            boundary.add(i)
    interior = sorted(set(range(n_nodes)) - boundary)

    K = np.zeros((n_nodes, n_nodes))
    M = np.zeros((n_nodes, n_nodes))
    for (x0, x1, y0, y1), enodes in zip(elems, elem_nodes):
        Je = 0.5
        M1d = Je * np.diag(w)
        K1d = (1.0 / Je) * (Dref.T @ np.diag(w) @ Dref)
        Ke = np.kron(K1d, M1d) + np.kron(M1d, K1d)
        Me = np.kron(M1d, M1d)
        for a in range(n_loc ** 2):
            for b in range(n_loc ** 2):
                K[enodes[a], enodes[b]] += Ke[a, b]
                M[enodes[a], enodes[b]] += Me[a, b]

    return (K[np.ix_(interior, interior)], M[np.ix_(interior, interior)],
            coords[interior, 0], coords[interior, 1])


# ------------------------------------------------------------------ 3D mesh
def build_3d(p, E):
    """3D Poisson on (0,1)^3: E x E x E elements of degree p.

    Returns K_int, M_int and interior coordinates (x, y, z), with node
    ordering ((k*n1d + j)*n1d + i) (x fastest).
    """
    xi, w = gll_points(p)
    Dref = diff_matrix_gll(xi)
    Je = 0.5 / E
    n1d = E * p + 1
    nodes_1d = np.zeros(n1d)
    for e in range(E):
        idx = list(range(e * p, e * p + p + 1))
        nodes_1d[idx] = e / E + Je * (xi + 1)
    M1d = Je * np.diag(w)
    K1d = (1.0 / Je) * (Dref.T @ np.diag(w) @ Dref)
    n3d = n1d ** 3
    K = np.zeros((n3d, n3d))
    M = np.zeros((n3d, n3d))
    for ex in range(E):
        for ey in range(E):
            for ez in range(E):
                idx = []
                for k in range(p + 1):
                    for j in range(p + 1):
                        for i in range(p + 1):
                            idx.append(((ez * p + k) * n1d + (ey * p + j)) * n1d + ex * p + i)
                Ke = (np.kron(K1d, np.kron(M1d, M1d)) +
                      np.kron(M1d, np.kron(K1d, M1d)) +
                      np.kron(M1d, np.kron(M1d, K1d)))
                Me = np.kron(M1d, np.kron(M1d, M1d))
                for a in range((p + 1) ** 3):
                    for b in range((p + 1) ** 3):
                        K[idx[a], idx[b]] += Ke[a, b]
                        M[idx[a], idx[b]] += Me[a, b]
    boundary = set()
    for i in range(n1d):
        for j in range(n1d):
            for k in range(n1d):
                x, y, z = nodes_1d[i], nodes_1d[j], nodes_1d[k]
                if (x < 1e-10 or x > 1 - 1e-10 or y < 1e-10 or y > 1 - 1e-10 or
                        z < 1e-10 or z > 1 - 1e-10):
                    boundary.add((k * n1d + j) * n1d + i)
    interior = sorted(set(range(n3d)) - boundary)
    xx = np.zeros(n3d); yy = np.zeros(n3d); zz = np.zeros(n3d)
    for k in range(n1d):
        for j in range(n1d):
            for i in range(n1d):
                idx = (k * n1d + j) * n1d + i
                xx[idx] = nodes_1d[i]; yy[idx] = nodes_1d[j]; zz[idx] = nodes_1d[k]
    return (K[np.ix_(interior, interior)], M[np.ix_(interior, interior)],
            xx[interior], yy[interior], zz[interior])


# ------------------------------------------------------------- torch helpers
def to_torch(K, M):
    return (torch.tensor(K, dtype=torch.float64),
            torch.tensor(M, dtype=torch.float64))


def l2_rel(u, u_ex):
    return float(torch.norm(u - u_ex) / torch.norm(u_ex))


def fd_gradient_central(fun, f0, h=1e-6):
    """Central finite-difference gradient of fun(f) (numpy or torch vector)."""
    f0 = f0.detach().cpu().numpy() if isinstance(f0, torch.Tensor) else np.asarray(f0)
    n = f0.size
    g = np.zeros(n)
    for i in range(n):
        fp = f0.copy(); fm = f0.copy()
        fp[i] += h; fm[i] -= h
        g[i] = (fun(torch.tensor(fp)) - fun(torch.tensor(fm))) / (2 * h)
    return torch.tensor(g, dtype=torch.float64)
