"""
2D Differentiable Spectral Element Method.
Tensor-product spectral elements on rectangular meshes.
"""
import torch
import numpy as np
from spectral_element import gll_points, differentiation_matrix_gll


class SpectralElement2D:
    """2D spectral element on [x0,x1] x [y0,y1] with polynomial order p."""
    
    def __init__(self, x0, x1, y0, y1, p):
        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = y0, y1
        self.p = p
        # 1D GLL points
        self.xi, self.w = gll_points(p)
        # Physical coordinates (tensor product)
        self.x_phys = 0.5 * (x1 - x0) * (self.xi + 1) + x0
        self.y_phys = 0.5 * (y1 - y0) * (self.xi + 1) + y0
        self.jac_x = 0.5 * (x1 - x0)
        self.jac_y = 0.5 * (y1 - y0)
        self.jac = self.jac_x * self.jac_y
        # 1D differentiation and mass matrices
        self.D1d = differentiation_matrix_gll(self.xi)
        self.M1d = np.diag(self.w)
        # 2D operators via tensor product
        # Mass: M = M_x ⊗ M_y (Kronecker product)
        self.M = np.kron(self.M1d * self.jac_x, self.M1d * self.jac_y)
        # Stiffness: K = K_x ⊗ M_y + M_x ⊗ K_y
        self.K1d = (1.0 / self.jac_x) * (self.D1d.T * self.w[np.newaxis, :]) @ self.D1d
        self.K1d_y = (1.0 / self.jac_y) * (self.D1d.T * self.w[np.newaxis, :]) @ self.D1d
        self.Mx = self.M1d * self.jac_x
        self.My = self.M1d * self.jac_y
        self.K = np.kron(self.K1d, self.My) + np.kron(self.Mx, self.K1d_y)
        # Node coordinates in flattened order (x varies fastest)
        self.n_nodes_elem = (p + 1) ** 2
        self.coords = np.zeros((self.n_nodes_elem, 2))
        idx = 0
        for j in range(p + 1):
            for i in range(p + 1):
                self.coords[idx, 0] = self.x_phys[i]
                self.coords[idx, 1] = self.y_phys[j]
                idx += 1


class DifferentiableSEM2D:
    """Differentiable 2D spectral element solver for -Delta u = f with Dirichlet BCs."""
    
    def __init__(self, elements, p):
        """
        elements: list of (x0, x1, y0, y1) tuples
        p: polynomial order
        """
        self.elements = [SpectralElement2D(x0, x1, y0, y1, p) for x0, x1, y0, y1 in elements]
        self.p = p
        self.n_elements = len(elements)
        # Build global node numbering
        self.global_nodes = []
        node_coords = []
        node_map = {}
        for e, elem in enumerate(self.elements):
            elem_nodes = []
            for idx in range(elem.n_nodes_elem):
                x, y = elem.coords[idx]
                key = (round(x, 12), round(y, 12))
                if key not in node_map:
                    node_map[key] = len(node_coords)
                    node_coords.append([x, y])
                elem_nodes.append(node_map[key])
            self.global_nodes.append(elem_nodes)
        self.n_nodes = len(node_coords)
        self.node_coords = np.array(node_coords)
        # Find boundary and interior nodes
        x_min = self.node_coords[:, 0].min()
        x_max = self.node_coords[:, 0].max()
        y_min = self.node_coords[:, 1].min()
        y_max = self.node_coords[:, 1].max()
        # Find boundary and interior nodes.
        # Outer bounding box edges are Dirichlet. Additionally, any node that
        # lies on an element edge but is NOT shared with another element is a
        # reentrant-corner boundary and also gets Dirichlet BC.
        self.boundary = []
        self.interior = []
        # Count how many elements each node belongs to
        elem_count = {}
        for e_nodes in self.global_nodes:
            for n in e_nodes:
                elem_count[n] = elem_count.get(n, 0) + 1
        # For each element edge, mark its nodes; a node on an edge but shared
        # by only one element is a reentrant-corner boundary.
        for i in range(self.n_nodes):
            x, y = self.node_coords[i]
            on_outer = (abs(x - x_min) < 1e-12 or abs(x - x_max) < 1e-12 or
                        abs(y - y_min) < 1e-12 or abs(y - y_max) < 1e-12)
            if on_outer:
                self.boundary.append(i)
                continue
            # Check if this node lies on any element's edge (not interior)
            on_elem_edge = False
            for e, elem in enumerate(self.elements):
                # Is this node on the boundary of element e?
                xi_idx = np.argmin(np.abs(elem.x_phys - x))
                yi_idx = np.argmin(np.abs(elem.y_phys - y))
                if (abs(x - elem.x_phys[xi_idx]) < 1e-10 and
                    abs(y - elem.y_phys[yi_idx]) < 1e-10):
                    if (xi_idx == 0 or xi_idx == elem.p or
                        yi_idx == 0 or yi_idx == elem.p):
                        on_elem_edge = True
                        break
            if on_elem_edge and elem_count.get(i, 0) == 1:
                self.boundary.append(i)
            else:
                self.interior.append(i)
        self.n_interior = len(self.interior)
        self._build_global_matrices()
    
    def _build_global_matrices(self):
        self.M_global = np.zeros((self.n_nodes, self.n_nodes))
        self.K_global = np.zeros((self.n_nodes, self.n_nodes))
        for e, elem in enumerate(self.elements):
            nodes = self.global_nodes[e]
            for i in range(elem.n_nodes_elem):
                for j in range(elem.n_nodes_elem):
                    self.M_global[nodes[i], nodes[j]] += elem.M[i, j]
                    self.K_global[nodes[i], nodes[j]] += elem.K[i, j]
        self.M_int = self.M_global[np.ix_(self.interior, self.interior)]
        self.K_int = self.K_global[np.ix_(self.interior, self.interior)]
        self.M_int_t = torch.tensor(self.M_int, dtype=torch.float64)
        self.K_int_t = torch.tensor(self.K_int, dtype=torch.float64)
    
    def solve(self, f_interior):
        """Solve -Delta u = f with zero Dirichlet BC."""
        rhs = self.M_int_t @ f_interior
        u_int = torch.linalg.solve(self.K_int_t, rhs)
        return u_int
    
    def get_full_solution(self, u_interior):
        u_full = torch.zeros(self.n_nodes, dtype=u_interior.dtype)
        for idx, node in enumerate(self.interior):
            u_full[node] = u_interior[idx]
        return u_full
