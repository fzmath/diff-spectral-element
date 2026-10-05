import sys; sys.path.insert(0,'.')
import numpy as np, torch, time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from spectral_element import DifferentiableSEM1D

print("n | CG_fwd_iters | CG_bwd_iters | CG_time_ms | grad_rel_err")
for n_elem in [20, 50, 100, 200]:
    elems = [(i/n_elem, (i+1)/n_elem) for i in range(n_elem)]
    solver = DifferentiableSEM1D(elems, p=4)
    n = solver.n_interior
    x = solver.node_coords[solver.interior]
    u_d = np.sin(np.pi*x)
    
    K = np.array(solver.K_int)
    M = np.array(solver.M_int)
    Ksp = sp.csr_matrix(K)
    
    # Forward: Ku = Mf, f=0 -> u=0
    b_fwd = M @ np.zeros(n)
    # Adjoint: K lambda = -(u - u_d) = u_d
    b_adj = -(np.zeros(n) - u_d)
    
    # Forward CG
    t0 = time.time()
    u_fwd, info_fwd = spla.cg(Ksp, b_fwd, rtol=1e-10, maxiter=500)
    it_fwd = "converged" if info_fwd == 0 else f"it={info_fwd}"
    
    # Backward CG
    lam, info_bwd = spla.cg(Ksp, b_adj, rtol=1e-10, maxiter=500)
    it_bwd = "converged" if info_bwd == 0 else f"it={info_bwd}"
    cg_ms = (time.time()-t0)/10*1000
    
    # Gradient: alpha f - M lambda
    alpha = 1e-6
    grad_cg = alpha*np.zeros(n) - M @ lam
    
    # Direct solve for comparison
    lam_direct = np.linalg.solve(K, b_adj)
    grad_direct = alpha*np.zeros(n) - M @ lam_direct
    
    grad_err = np.linalg.norm(grad_cg - grad_direct) / np.linalg.norm(grad_direct)
    
    print(f"{n} | {it_fwd} | {it_bwd} | {cg_ms:.1f} | {grad_err:.2e}")
