import sys; sys.path.insert(0,'.')
import torch, numpy as np, time
from spectral_element import DifferentiableSEM1D

print("n | CG_fwd_iters | CG_bwd_iters | CG_time_ms | grad_err")
for n_elem in [20, 50, 100, 200]:
    elems = [(i/n_elem, (i+1)/n_elem) for i in range(n_elem)]
    solver = DifferentiableSEM1D(elems, p=4)
    n = solver.n_interior
    x = solver.node_coords[solver.interior]
    u_d = torch.tensor(np.sin(np.pi*x), dtype=torch.float64)
    K = solver.K_int_t
    M = solver.M_int_t
    alpha = 1e-6
    
    # CG solve with Jacobi preconditioner
    def cg_solve(A, b, tol=1e-8, maxit=500):
        M_jac = 1.0 / torch.diag(A)
        x = torch.zeros_like(b)
        r = b - A @ x
        z = M_jac * r
        p = z.clone()
        rz_old = (r*z).sum()
        for it in range(maxit):
            Ap = A @ p
            alpha_step = rz_old / (p*Ap).sum()
            x = x + alpha_step * p
            r = r - alpha_step * Ap
            if torch.norm(r) < tol * torch.norm(b):
                break
            z = M_jac * r
            rz_new = (r*z).sum()
            p = z + (rz_new/rz_old) * p
            rz_old = rz_new
        return x, it+1
    
    # Forward: Ku = Mf, f = 0 -> u=0
    f = torch.zeros(n, dtype=torch.float64)
    b_fwd = M @ f
    u_fwd, it_fwd = cg_solve(K, b_fwd)
    
    # Adjoint: K^T lambda = -(u - u_d) = u_d (since u=0)
    b_adj = -(u_fwd - u_d)
    lam, it_bwd = cg_solve(K, b_adj)  # K symmetric
    
    # Gradient
    grad = alpha*f - M @ lam
    
    # Compare with direct solve gradient
    u_direct = torch.linalg.solve(K, b_fwd)
    lam_direct = torch.linalg.solve(K, b_adj)
    grad_direct = alpha*f - M @ lam_direct
    
    grad_err = torch.norm(grad - grad_direct).item() / torch.norm(grad_direct).item()
    
    # Timing
    t0 = time.time()
    for _ in range(10):
        u_fwd, _ = cg_solve(K, b_fwd)
    t_cg = (time.time()-t0)/10*1000
    
    print(f"{n} | {it_fwd} | {it_bwd} | {t_cg:.1f} | {grad_err:.2e}")
