"""
Differentiable Spectral Element Method (DSEM) for PDE-constrained optimization.
"""
import torch
import numpy as np
from numpy.polynomial import legendre


def gll_points(p):
    """Gauss-Lobatto-Legendre points and weights for polynomial order p."""
    if p == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    coeffs = np.zeros(p + 2)
    coeffs[p] = 1.0
    Pp = legendre.Legendre(coeffs)
    Pp_deriv = Pp.deriv()
    inner_roots = Pp_deriv.roots()
    x = np.concatenate([[-1.0], np.sort(inner_roots.real), [1.0]])
    P_vals = Pp(x)
    w = 2.0 / (p * (p + 1) * P_vals**2)
    return x, w


def differentiation_matrix_gll(x):
    """Differentiation matrix using exact Lagrange formula.
    D[i,j] = L'_j(x_i)"""
    n = len(x)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # L'_j(x_i) = prod_{k!=j, k!=i} (x_i - x_k) / prod_{k!=j} (x_j - x_k)
                num = 1.0
                den = 1.0
                for k in range(n):
                    if k != j and k != i:
                        num *= (x[i] - x[k])
                for k in range(n):
                    if k != j:
                        den *= (x[j] - x[k])
                D[i, j] = num / den
            else:
                D[i, i] = sum(1.0 / (x[i] - x[k]) for k in range(n) if k != i)
    return D


class SpectralElement1D:
    """1D spectral element on [x0, x1] with polynomial order p."""
    
    def __init__(self, x0, x1, p):
        self.x0 = x0
        self.x1 = x1
        self.p = p
        self.x_ref, self.w_ref = gll_points(p)
        self.x_phys = 0.5 * (x1 - x0) * (self.x_ref + 1) + x0
        self.jac = 0.5 * (x1 - x0)
        self.D_ref = differentiation_matrix_gll(self.x_ref)
        self.D = self.D_ref / self.jac
        # Mass matrix (diagonal)
        self.M = np.diag(self.w_ref * self.jac)
        # Stiffness matrix: K = jac * D_ref^T * diag(w) * D_ref
        self.M_ref_diag = self.w_ref
        self.K = (1.0 / self.jac) * (self.D_ref.T * self.w_ref[np.newaxis, :]) @ self.D_ref


class DifferentiableSEM1D:
    """Differentiable 1D spectral element solver for -u'' = f with Dirichlet BCs."""
    
    def __init__(self, elements, p):
        self.elements = [SpectralElement1D(x0, x1, p) for x0, x1 in elements]
        self.p = p
        self.n_elements = len(elements)
        # Build global node numbering
        self.global_nodes = []
        node_coords = []
        node_map = {}
        for e, elem in enumerate(self.elements):
            elem_nodes = []
            for i, x in enumerate(elem.x_phys):
                key = round(x, 12)
                if key not in node_map:
                    node_map[key] = len(node_coords)
                    node_coords.append(x)
                elem_nodes.append(node_map[key])
            self.global_nodes.append(elem_nodes)
        self.n_nodes = len(node_coords)
        self.node_coords = np.array(node_coords)
        # Interior nodes
        self.interior = [i for i in range(self.n_nodes) 
                        if abs(self.node_coords[i] - self.node_coords[0]) > 1e-12 
                        and abs(self.node_coords[i] - self.node_coords[-1]) > 1e-12]
        self.n_interior = len(self.interior)
        self._build_global_matrices()
    
    def _build_global_matrices(self):
        self.M_global = np.zeros((self.n_nodes, self.n_nodes))
        self.K_global = np.zeros((self.n_nodes, self.n_nodes))
        for e, elem in enumerate(self.elements):
            nodes = self.global_nodes[e]
            for i in range(self.p + 1):
                for j in range(self.p + 1):
                    self.M_global[nodes[i], nodes[j]] += elem.M[i, j]
                    self.K_global[nodes[i], nodes[j]] += elem.K[i, j]
        self.M_int = self.M_global[np.ix_(self.interior, self.interior)]
        self.K_int = self.K_global[np.ix_(self.interior, self.interior)]
        self.M_int_t = torch.tensor(self.M_int, dtype=torch.float64)
        self.K_int_t = torch.tensor(self.K_int, dtype=torch.float64)
    
    def solve(self, f_interior):
        rhs = self.M_int_t @ f_interior
        u_int = torch.linalg.solve(self.K_int_t, rhs)
        return u_int
    
    def get_full_solution(self, u_interior):
        u_full = torch.zeros(self.n_nodes, dtype=u_interior.dtype)
        for idx, node in enumerate(self.interior):
            u_full[node] = u_interior[idx]
        return u_full
