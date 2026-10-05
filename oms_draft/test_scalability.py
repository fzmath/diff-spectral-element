import sys; sys.path.insert(0,'.')
import torch, numpy as np, time, scipy.sparse as sp
import scipy.sparse.linalg as spla
from spectral_element import DifferentiableSEM1D

print("n_elem | n_int | dense_ms | sparse_ms | CG_ms | dense_mem_MB | sparse_mem_MB")
for n_elem in [10, 20, 50, 100, 200]:
    elems = [(i/n_elem, (i+1)/n_elem) for i in range(n_elem)]
    solver = DifferentiableSEM1D(elems, p=4)
    n = solver.n_interior
    K = np.array(solver.K_int)
    M = np.array(solver.M_int)
    f = np.ones(n)*0.1

    # Dense direct
    t0 = time.time()
    for _ in range(10):
        u = np.linalg.solve(K, M@f)
    dense_ms = (time.time()-t0)/10*1000
    dense_mem = K.nbytes/1e6

    # Sparse direct
    Ksp = sp.csr_matrix(K)
    t0 = time.time()
    for _ in range(10):
        u = spla.spsolve(Ksp, M@f)
    sparse_ms = (time.time()-t0)/10*1000
    sparse_mem = Ksp.data.nbytes/1e6

    # CG (matrix-free approximation: just K^T K products)
    t0 = time.time()
    for _ in range(10):
        b = M@f
        u_cg, info = spla.cg(Ksp, b, rtol=1e-10, maxiter=200)
    cg_ms = (time.time()-t0)/10*1000

    print(f"{n_elem} | {n} | {dense_ms:.2f} | {sparse_ms:.2f} | {cg_ms:.2f} | {dense_mem:.3f} | {sparse_mem:.4f}")
