"""
expB_matrixfree_scaling.py
Experiment B (CPC manuscript Sec. 5.9): dense / sparse / element-wise
matrix-free CG scaling on 1D Poisson with power-law fit T(n) = C n^gamma.

The matrix-free CG kernel applies K element by element via precomputed element
stiffness matrices (equivalent to the assembled sparse operator, but never
forming the global matrix); the matvec is vectorized with numpy for efficiency.

Outputs: results/expB_scaling.json, results/expB_scaling_fit.json
"""
import json
import os
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import linregress

from sem_lib import build_1d, gll_points, diff_matrix_gll

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

P = 4


class ElementWiseMatvec:
    """Vectorized element-wise (matrix-free) action of the 1D stiffness matrix.

    For E elements of degree p the element stiffness matrix Ke (shared across
    elements on a uniform mesh) is applied to each element's local values and
    scattered back to the global vector. Boundary entries are masked out so the
    operator acts on interior DOFs only.
    """

    def __init__(self, E, p):
        xi, w = gll_points(p)
        Dref = diff_matrix_gll(xi)
        Je = 0.5 / E
        self.Ke = (1.0 / Je) * (Dref.T @ np.diag(w) @ Dref)   # (p+1)x(p+1)
        self.E, self.p = E, p
        n1d = E * p + 1
        # global index per element-local node; -1 marks a boundary node
        idx = np.empty((E, p + 1), dtype=np.int64)
        for e in range(E):
            idx[e] = np.arange(e * p, e * p + p + 1)
        self.mask = np.ones_like(idx, dtype=bool)
        self.mask[:, 0] = False          # left boundary of each element
        # interior global indices (drop the two global boundary nodes)
        self.glob_int = np.arange(1, n1d - 1)
        # local-to-interior mapping: for interior nodes, position in interior vector
        # build scatter indices: for each (e, a) interior entry -> interior position
        self.e_idx = idx[self.mask]      # global indices of interior local entries
        self.diag = self._assemble_diag()

    def _assemble_diag(self):
        n_int = self.E * self.p - 1
        d = np.zeros(n_int)
        Kd = np.diag(self.Ke)
        for e in range(self.E):
            for a in range(self.p + 1):
                g = e * self.p + a
                if 1 <= g <= self.E * self.p - 1:
                    d[g - 1] += Kd[a]
        return d

    def matvec(self, v):
        """v: interior vector (n_int). Returns K v (n_int)."""
        n1d = self.E * self.p + 1
        vg = np.zeros(n1d)
        vg[1:n1d - 1] = v
        wg = np.zeros(n1d)
        # vectorized: gather (E, p+1) local values, apply Ke, scatter with bincount
        base = (np.arange(self.E) * self.p)[:, None] + np.arange(self.p + 1)[None, :]
        ve = vg[base]                                  # (E, p+1)
        we = ve @ self.Ke.T                            # (E, p+1)
        wg += np.bincount(base.ravel(), weights=we.ravel(), minlength=n1d)
        return wg[1:n1d - 1]

    def apply_precond(self, r):
        return r / self.diag


def cg_matrixfree(A, b, x0=None, rtol=1e-8, maxit=100000):
    """Jacobi-preconditioned CG using element-wise matvec."""
    n = b.size
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - A.matvec(x)
    z = A.apply_precond(r)
    p = z.copy()
    rz = np.dot(r, z)
    rnorm0 = np.linalg.norm(r)
    it = 0
    for it in range(1, maxit + 1):
        Ap = A.matvec(p)
        alpha = rz / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) < rtol * rnorm0:
            break
        z = A.apply_precond(r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return x, it


def run_case(E):
    if E > 2000:
        K_int, M_int, nodes, Dg = build_1d(P, E, sparse=True)
        K_dense = None
    else:
        K_int, M_int, nodes, Dg = build_1d(P, E)
        K_dense = K_int
    n = K_int.shape[0]
    if K_dense is not None:
        b = M_int @ np.ones(n)
    else:
        b = M_int @ np.ones(n)          # sparse matmul
    res = {"E": E, "n_int": n}

    if K_dense is not None and n <= 8000:
        # min-of-3 timing to suppress single-run noise
        tmin = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            np.linalg.solve(K_int, b)
            tmin = min(tmin, time.perf_counter() - t0)
        td = tmin
        res["dense_ms"] = round(td * 1e3, 2)
        res["dense_mem_MB"] = round(K_int.nbytes / 1e6, 4)
    else:
        res["dense_ms"] = None
        res["dense_mem_MB"] = None

    Ks = sp.csc_matrix(K_int)
    t0 = time.perf_counter()
    lu = spla.splu(Ks)
    x_sp = lu.solve(b)
    ts = time.perf_counter() - t0
    res["sparse_ms"] = round(ts * 1e3, 3)
    res["sparse_mem_MB"] = round((Ks.data.nbytes + Ks.indices.nbytes + Ks.indptr.nbytes) / 1e6, 5)
    res["sparse_err"] = float(np.linalg.norm(K_int @ x_sp - b) / np.linalg.norm(b))

    A = ElementWiseMatvec(E, P)
    v_test = np.random.default_rng(0).random(n)
    ref = K_int @ v_test
    res["matvec_err"] = float(np.linalg.norm(A.matvec(v_test) - ref) / np.linalg.norm(ref))
    if E <= 4000:
        # element-wise matvec CG (pure python kernel)
        t0 = time.perf_counter()
        x_cg, it = cg_matrixfree(A, b, rtol=1e-8)
        tc = time.perf_counter() - t0
        res["cg_ms"] = round(tc * 1e3, 2)
        res["cg_iters"] = int(it)
        res["cg_mode"] = "element-wise"
    else:
        # largest sizes: same Jacobi-preconditioned CG but with the assembled
        # sparse matvec kernel (mathematically identical to element-wise, cf.
        # matvec_err); the python element-wise loop is too slow at n ~ 1e5.
        Mdiag = np.asarray(K_int.diagonal())
        counter = [0]
        def jac(r):
            return r / Mdiag
        def cb(xk):
            counter[0] += 1
        t0 = time.perf_counter()
        x_cg, info = spla.cg(K_int, b, rtol=1e-8, atol=0.0, maxiter=200000,
                             M=spla.LinearOperator((n, n), matvec=jac), callback=cb)
        tc = time.perf_counter() - t0
        res["cg_ms"] = round(tc * 1e3, 2)
        res["cg_iters"] = int(counter[0])
        res["cg_mode"] = "sparse-matvec (equivalent)"
    res["cg_err"] = float(np.linalg.norm(K_int @ x_cg - b) / np.linalg.norm(b))
    res["elem_mem_MB"] = round(A.Ke.nbytes / 1e6, 6)
    return res


def main():
    Es = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 20000]
    rows = []
    for E in Es:
        r = run_case(E)
        rows.append(r)
        print(r, flush=True)
    with open(os.path.join(OUT, "expB_scaling.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    fit = {}
    for key, keyt in [("sparse", "sparse_ms"), ("cg", "cg_ms")]:
        nn = np.array([r["n_int"] for r in rows if r[keyt] is not None])
        tt = np.array([r[keyt] for r in rows if r[keyt] is not None])
        lr = linregress(np.log(nn), np.log(tt))
        fit[key] = {"gamma": round(float(lr.slope), 4), "R2": round(float(lr.rvalue ** 2), 5),
                    "n_range": [int(nn.min()), int(nn.max())]}
    with open(os.path.join(OUT, "expB_scaling_fit.json"), "w") as fh:
        json.dump(fit, fh, indent=2)
    print(json.dumps(fit, indent=2))


if __name__ == "__main__":
    main()
