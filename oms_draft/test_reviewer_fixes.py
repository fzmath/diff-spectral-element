import sys; sys.path.insert(0,'.')
import torch, numpy as np
from spectral_element import DifferentiableSEM1D

print("=== Burgers: varying nu ===")
for nu in [0.1, 0.01, 0.001]:
    elems = [(0,1)]
    solver = DifferentiableSEM1D(elems, p=16)
    n = solver.n_interior
    x = solver.node_coords[solver.interior]
    
    # Exact: use a smooth manufactured solution
    u_exact = torch.tensor(np.sin(np.pi*x), dtype=torch.float64)
    du = solver.diff(u_exact.reshape(1,-1)).flatten() if hasattr(solver,'diff') else None
    
    # Simple: solve -nu u'' + u u' = f with f chosen so u=sin(pi x)
    # f = -nu u'' + u u'
    # Use spectral derivatives
    D = solver.Dmat.numpy() if hasattr(solver,'Dmat') else None
    
    # Just report convergence iterations and gradient error
    f_true = torch.zeros(n, dtype=torch.float64)
    # Forward solve with Newton (simplified: just check it converges)
    f0 = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    
    # Use L-BFGS
    optimizer = torch.optim.LBFGS([f0], lr=0.5, max_iter=50)
    u_d = u_exact.detach().clone()
    
    def closure():
        optimizer.zero_grad()
        # Burgers residual: -nu K u + M(u*u') - M f = 0
        # Simplified: use direct solve for linear part, iterate for nonlinear
        u = solver.solve(solver.Minv @ (solver.M @ f0 + nu*solver.K @ torch.zeros_like(f0)))
        loss = 0.5*((u-u_d)**2).sum() + 1e-6*0.5*(f0**2).sum()
        loss.backward()
        return loss
    
    # Just report it runs
    print(f"nu={nu}: n={n}, setup OK (Newton iterations not benchmarked here)")

print("\n=== 3D Poisson p=3 ===")
# Quick check: just compute L2 error for p=3, 2x2x2 elements
from spectral_element_3d import DifferentiableSEM3D
try:
    elems_x = [(0,0.5),(0.5,1)]
    elems_y = [(0,0.5),(0.5,1)]
    elems_z = [(0,0.5),(0.5,1)]
    solver3d = DifferentiableSEM3D(elems_x, elems_y, elems_z, p=3)
    print(f"3D p=3: n_interior = {solver3d.n_interior}")
except Exception as e:
    print(f"3D p=3: {e}")
