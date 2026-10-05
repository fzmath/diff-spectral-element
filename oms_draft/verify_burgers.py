import numpy as np
from scipy.special import roots_legendre, eval_legendre, legendre
from scipy.optimize import minimize

def gll(p):
    P = legendre(p)
    dP = P.deriv()
    interior = np.sort(dP.roots.real)
    xi = np.concatenate([[-1.0], interior, [1.0]])
    w = np.array([2.0/(p*(p+1)*eval_legendre(p, x)**2) for x in xi])
    return xi, w

# Burgers: -nu u_xx + u u_x = f, u(0)=u(1)=0
# Use p=16, 2 elements as in paper
p = 16; E = 2
xi, w = gll(p)
n = p+1
c = 1.0/np.array([np.prod([xi[j]-xi[k] for k in range(n) if k!=j]) for j in range(n)])
D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i != j: D[i,j] = (c[j]/c[i])/(xi[i]-xi[j])
    D[i,i] = -np.sum(D[i,:])

xg, wg = roots_legendre(30)
L = np.zeros((len(xg), n))
for j in range(n):
    prod = np.ones(len(xg))
    for k in range(n):
        if k != j: prod *= (xg - xi[k])/(xi[j] - xi[k])
    L[:,j] = prod
Mc = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        Mc[i,j] = np.sum(wg * L[:,i] * L[:,j])
Ke = D.T @ np.diag(w) @ D

he = 1.0/E; Je = he/2
Me = Je * Mc; Ke = Ke/Je
n_nodes = E*p+1
K = np.zeros((n_nodes,n_nodes)); M = np.zeros((n_nodes,n_nodes))
nodes = np.zeros(n_nodes)
for e in range(E):
    idx = list(range(e*p,e*p+p+1))
    x_elem = e*he + Je*(xi+1)
    for ii in range(p+1):
        nodes[idx[ii]] = x_elem[ii]
        for jj in range(p+1):
            K[idx[ii],idx[jj]] += Ke[ii,jj]
            M[idx[ii],idx[jj]] += Me[ii,jj]

Kint = K[1:-1,1:-1]; Mint = M[1:-1,1:-1]
Dint = D[1:-1,1:-1]  # derivative at interior nodes (approx)
x = nodes[1:-1]

print("=== Burgers gradient verification ===")
for nu in [0.1, 0.01, 0.001]:
    # Newton solve: R(u) = nu K u + M*(u*(D u)) - M f = 0
    # Choose f such that u_exact = sin(pi x)
    u_ex = np.sin(np.pi*x)
    Du = np.gradient(u_ex, x)  # approximate derivative
    f = nu*Kint@u_ex + Mint@(u_ex*Du)  # RHS

    # Newton iteration
    u = np.zeros(len(x))
    for it in range(50):
        Du = np.gradient(u, x)
        R = nu*Kint@u + Mint@(u*Du) - Mint@f
        if np.max(np.abs(R)) < 1e-10:
            break
        # Jacobian
        J = nu*Kint + Mint@np.diag(Du) + Mint@np.diag(u)@np.diag(np.gradient(np.ones_like(u), x))
        # Simplified: J = nu K + M diag(Du) + M diag(u) D
        J = nu*Kint + Mint@np.diag(Du) + Mint@np.diag(u)@np.gradient(np.eye(len(x)), x, axis=1)
        du = np.linalg.solve(J, -R)
        u = u + du
    print(f"nu={nu}: Newton {it+1} iters, residual={np.max(np.abs(R)):.2e}")

    # Gradient check: minimize 0.5||u-u_d||^2 + alpha/2||f||^2
    u_d = np.sin(np.pi*x)
    alpha = 1e-6
    # At f=0, compute gradient via adjoint
    def newton_solve(f):
        u = np.zeros(len(x))
        J = None
        for it in range(50):
            Du = np.gradient(u, x)
            R = nu*Kint@u + Mint@(u*Du) - Mint@f
            if np.max(np.abs(R)) < 1e-10: break
            J = nu*Kint + Mint@np.diag(Du) + Mint@np.diag(u)@np.gradient(np.eye(len(x)), x, axis=1)
            du = np.linalg.solve(J, -R)
            u = u + du
        if J is None:
            Du = np.gradient(u, x)
            J = nu*Kint + Mint@np.diag(Du) + Mint@np.diag(u)@np.gradient(np.eye(len(x)), x, axis=1)
        return u, J

    u, J = newton_solve(np.zeros(len(x)))
    # Adjoint: J^T lambda = -(u-u_d)
    lam = np.linalg.solve(J.T, -(u-u_d))
    g_ad = -Mint@lam + alpha*np.zeros(len(x))

    # FD gradient
    h = 1e-6
    g_fd = np.zeros(5)
    for i in range(5):
        fp = np.zeros(len(x)); fp[i] += h
        fm = np.zeros(len(x)); fm[i] -= h
        up, _ = newton_solve(fp)
        um, _ = newton_solve(fm)
        Jp = 0.5*np.sum((up-u_d)**2) + alpha*0.5*np.sum(fp**2)
        Jm = 0.5*np.sum((um-u_d)**2) + alpha*0.5*np.sum(fm**2)
        g_fd[i] = (Jp-Jm)/(2*h)
    gerr = np.linalg.norm(g_ad[:5]-g_fd)/np.linalg.norm(g_ad[:5])
    print(f"  gradient FD error = {gerr:.2e}")
