"""
3D Differentiable Spectral Element Method.
Tensor-product spectral elements on hexahedral meshes.
"""
import torch
import numpy as np
from spectral_element import gll_points, differentiation_matrix_gll


class SpectralElement3D:
    """3D spectral element on [x0,x1] x [y0,y1] x [z0,z1] with polynomial order p."""
    
    def __init__(self, x0, x1, y0, y1, z0, z1, p):
        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = y0, y1
        self.z0, self.z1 = z0, z1
        self.p = p
        self.xi, self.w = gll_points(p)
        self.jac_x = 0.5 * (x1 - x0)
        self.jac_y = 0.5 * (y1 - y0)
        self.jac_z = 0.5 * (z1 - z0)
        self.jac = self.jac_x * self.jac_y * self.jac_z
        # 1D operators
        self.D1d = differentiation_matrix_gll(self.xi)
        self.M1d = np.diag(self.w)
        self.K1d_x = (1.0 / self.jac_x) * (self.D1d.T * self.w[np.newaxis, :]) @ self.D1d
        self.K1d_y = (1.0 / self.jac_y) * (self.D1d.T * self.w[np.newaxis, :]) @ self.D1d
        self.K1d_z = (1.0 / self.jac_z) * (self.D1d.T * self.w[np.newaxis, :]) @ self.D1d
        self.Mx = self.M1d * self.jac_x
        self.My = self.M1d * self.jac_y
        self.Mz = self.M1d * self.jac_z
        # 3D mass: M = Mx ⊗ My ⊗ Mz
        self.M = np.kron(np.kron(self.Mx, self.My), self.Mz)
        # 3D stiffness: K = Kx⊗My⊗Mz + Mx⊗Ky⊗Mz + Mx⊗My⊗Kz
        self.K = (np.kron(np.kron(self.K1d_x, self.My), self.Mz) +
                  np.kron(np.kron(self.Mx, self.K1d_y), self.Mz) +
                  np.kron(np.kron(self.Mx, self.My), self.K1d_z))
        # Node coordinates (x varies fastest, then y, then z)
        self.n_nodes_elem = (p + 1) ** 3
        self.coords = np.zeros((self.n_nodes_elem, 3))
        x_phys = 0.5 * (x1 - x0) * (self.xi + 1) + x0
        y_phys = 0.5 * (y1 - y0) * (self.xi + 1) + y0
        z_phys = 0.5 * (z1 - z0) * (self.xi + 1) + z0
        idx = 0
        for k in range(p + 1):
            for j in range(p + 1):
                for i in range(p + 1):
                    self.coords[idx, 0] = x_phys[i]
                    self.coords[idx, 1] = y_phys[j]
                    self.coords[idx, 2] = z_phys[k]
                    idx += 1


class DifferentiableSEM3D:
    """Differentiable 3D spectral element solver for -Delta u = f with Dirichlet BCs."""
    
    def __init__(self, elements, p):
        """
        elements: list of (x0, x1, y0, y1, z0, z1) tuples
        p: polynomial order
        """
        self.elements = [SpectralElement3D(x0, x1, y0, y1, z0, z1, p) 
                         for x0, x1, y0, y1, z0, z1 in elements]
        self.p = p
        self.n_elements = len(elements)
        # Build global node numbering
        self.global_nodes = []
        node_coords = []
        node_map = {}
        for e, elem in enumerate(self.elements):
            elem_nodes = []
            for idx in range(elem.n_nodes_elem):
                x, y, z = elem.coords[idx]
                key = (round(x, 12), round(y, 12), round(z, 12))
                if key not in node_map:
                    node_map[key] = len(node_coords)
                    node_coords.append([x, y, z])
                elem_nodes.append(node_map[key])
            self.global_nodes.append(elem_nodes)
        self.n_nodes = len(node_coords)
        self.node_coords = np.array(node_coords)
        # Boundary/interior split
        x_min, x_max = self.node_coords[:, 0].min(), self.node_coords[:, 0].max()
        y_min, y_max = self.node_coords[:, 1].min(), self.node_coords[:, 1].max()
        z_min, z_max = self.node_coords[:, 2].min(), self.node_coords[:, 2].max()
        self.boundary = []
        self.interior = []
        for i in range(self.n_nodes):
            x, y, z = self.node_coords[i]
            if (abs(x - x_min) < 1e-12 or abs(x - x_max) < 1e-12 or
                abs(y - y_min) < 1e-12 or abs(y - y_max) < 1e-12 or
                abs(z - z_min) < 1e-12 or abs(z - z_max) < 1e-12):
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
        rhs = self.M_int_t @ f_interior
        u_int = torch.linalg.solve(self.K_int_t, rhs)
        return u_int
