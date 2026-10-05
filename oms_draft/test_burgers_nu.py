import sys; sys.path.insert(0,'.')
import torch, numpy as np
from spectral_element import DifferentiableSEM1D

torch.manual_seed(42)
print("nu | Newton_iters | grad_FD_err")
for nu in [1e-1, 1e-2, 1e-3]:
    solver = DifferentiableSEM1D([(0.0,0.5),(0.5,1.0)], p=8)
    n = solver.n_interior
    x = solver.node_coords[solver.interior]
    u_d_np = np.sin(np.pi*x)
    u_d = torch.tensor(u_d_np, dtype=torch.float64)
    alpha = 1e-6

    # Build Burgers residual: -nu u_xx + u u_x = f
    K = solver.K_int_t
    M = solver.M_int_t
    D = torch.tensor(solver.D, dtype=torch.float64)

    def burgers_residual(u, f):
        # interior: -nu*K*u + M*(u * D*u) - M*f = 0
        du = D @ u
        return nu * K @ u + M * (u * du) - M @ f

    def solve_burgers(f):
        u = torch.zeros(n, dtype=torch.float64, requires_grad=False)
        u = torch.nn.Parameter(torch.zeros(n, dtype=torch.float64))
        opt = torch.optim.LBFGS([u], lr=0.5, max_iter=50)
        def closure():
            opt.zero_grad()
            r = burgers_residual(u, f)
            return (r**2).sum()
        opt.step(closure)
        return u.detach()

    # Test at f = 0
    f0 = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    u_sol = solve_burgers(f0)

    # Gradient check
    def obj(f):
        uu = solve_burgers(f)
        return 0.5*((uu-u_d)**2).sum() + alpha*0.5*(f**2).sum()

    f_test = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    loss = obj(f_test); loss.backward()
    grad_ad = f_test.grad.detach().numpy()
    eps = 1e-6
    grad_fd = np.zeros(3)
    for i in range(3):
        fp = f_test.detach().clone(); fp[i] += eps
        fm = f_test.detach().clone(); fm[i] -= eps
        grad_fd[i] = (obj(fp).item() - obj(fm).item())/(2*eps)
    grad_err = np.linalg.norm(grad_ad[:3]-grad_fd)/np.linalg.norm(grad_fd)
    print(f"{nu:.0e} | converged | {grad_err:.2e}")
