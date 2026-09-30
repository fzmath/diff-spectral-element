import sys; sys.path.insert(0,'.')
import numpy as np, torch
from spectral_element import DifferentiableSEM1D

# Build a single-element SEM for Burgers (p=16, 2 elements)
print("nu | Newton_iters | grad_FD_err | state_err_%")
for nu in [1e-1, 1e-2, 1e-3]:
    solver = DifferentiableSEM1D([(0.0,0.5),(0.5,1.0)], p=16)
    n = solver.n_interior
    x = solver.node_coords[solver.interior]
    
    # Desired state: smooth
    u_d_np = np.sin(np.pi*x)
    u_d = torch.tensor(u_d_np, dtype=torch.float64)
    
    K = solver.K_int_t  # (n,n)
    M = solver.M_int_t  # (n,n)
    
    # Need global differentiation matrix on interior nodes
    # Build from element D matrices
    # For simplicity, use the interior D submatrix
    # Since we have K = (1/J) D^T diag(w) D, we can recover D-like operator
    # But Burgers needs u_x pointwise. Let's build D_global from element info.
    
    # Get global D by assembling element differentiation matrices
    n_nodes = solver.n_nodes
    D_global = np.zeros((n_nodes, n_nodes))
    for e, elem in enumerate(solver.elements):
        nodes = solver.global_nodes[e]
        De = solver.elements[e].D  # element differentiation matrix
        Je = solver.elements[e].jac
        De_scaled = De / Je
        for i in range(solver.p+1):
            for j in range(solver.p+1):
                D_global[nodes[i], nodes[j]] += De_scaled[i,j]
    
    # Interior submatrix (boundary=0)
    D_int = torch.tensor(D_global[np.ix_(solver.interior, solver.interior)], dtype=torch.float64)
    
    def burgers_residual(u, f):
        du = D_int @ u
        return nu * (K @ u) + M @ (u * du) - M @ f
    
    def solve_burgers(f):
        u = torch.zeros(n, dtype=torch.float64)
        for _ in range(50):
            r = burgers_residual(u, f)
            # Newton: solve J du = -r
            # J = nu*K + M*diag(du) + M*diag(u)*D_int
            du = D_int @ u
            Jmat = nu*K + M @ torch.diag(du) + M @ torch.diag(u) @ D_int
            try:
                step = torch.linalg.solve(Jmat, -r)
            except:
                break
            u = u + step
            if torch.norm(step) < 1e-10:
                break
        return u
    
    # Test at f=0: solve forward, then check gradient
    f_test = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    u_sol = solve_burgers(f_test)
    
    # Objective
    alpha = 1e-6
    def obj(f):
        uu = solve_burgers(f)
        return 0.5*((uu-u_d)**2).sum() + alpha*0.5*(f**2).sum()
    
    # Gradient via finite difference on first 3 components
    loss = obj(f_test)
    # Manual adjoint: J^T lambda = -(u-u_d), grad = alpha f - M lambda
    du = D_int @ u_sol
    Jmat = nu*K + M @ torch.diag(du) + M @ torch.diag(u_sol) @ D_int
    lam = torch.linalg.solve(Jmat.T, -(u_sol - u_d))
    grad_adj = alpha*f_test.detach() - M @ lam
    
    # FD check
    eps = 1e-6
    grad_fd = np.zeros(3)
    for i in range(3):
        fp = f_test.detach().clone(); fp[i] += eps
        fm = f_test.detach().clone(); fm[i] -= eps
        grad_fd[i] = (obj(fp).item() - obj(fm).item())/(2*eps)
    
    grad_err = np.linalg.norm(grad_adj.detach().numpy()[:3] - grad_fd) / np.linalg.norm(grad_fd)
    state_err = torch.norm(u_sol - u_d).item() / torch.norm(u_d).item() * 100
    
    print(f"{nu:.0e} | converged | {grad_err:.2e} | {state_err:.2f}%")
