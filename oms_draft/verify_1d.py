import numpy as np
from scipy.special import roots_legendre
from scipy.optimize import minimize

def build_1d(p, E=2, alpha=1e-6):
    """Build 1D SEM matrices for -u''=pi^2 sin(pi x) on [0,1]."""
    # GLL nodes
    from numpy.polynomial.legendre import legder, legval
    # Use scipy for GLL nodes
    from scipy.special import roots_legendre as rl
    # GLL nodes: roots of (1-x^2)*P_p'(x)
    # Use known formula
    xi = np.sort(np.concatenate([[-1.0], rl(p-1, mu=1)[0] if p>1 else [], [1.0]]))
    # Actually GLL = roots of derivative of Legendre P_p
    from scipy.special import eval_legendre, legendre
    P = legendre(p)
    dP = P.deriv()
    interior = np.sort(dP.roots.real)
    xi = np.concatenate([[-1.0], interior, [1.0]])
    # Weights
    w = np.zeros(p+1)
    for i in range(p+1):
        w[i] = 2.0 / (p*(p+1) * eval_legendre(p, xi[i])**2)

    n = p+1
    # Barycentric weights
    c = 1.0/np.array([np.prod([xi[j]-xi[k] for k in range(n) if k!=j]) for j in range(n)])
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j: D[i,j] = (c[j]/c[i])/(xi[i]-xi[j])
        D[i,i] = -np.sum(D[i,:])

    # Consistent mass
    xg, wg = rl(20)
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
    Me = Je * Mc
    Ke = Ke / Je
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
    return K[1:-1,1:-1], M[1:-1,1:-1], nodes[1:-1]

# Table 1: 1D convergence
print("=== Table 1: 1D L2 error ===")
for p in [2, 4, 8, 16]:
    K, M, x = build_1d(p, E=2)
    f = np.pi**2 * np.sin(np.pi*x)
    u = np.linalg.solve(K, M@f)
    u_ex = np.sin(np.pi*x)
    err = np.sqrt(np.mean((u-u_ex)**2))
    kappa = np.linalg.cond(K)
    print(f"p={p:2d}: n_int={len(x):3d}, L2={err:.2e}, kappa(K)={kappa:.2e}")

# Table 2: condition numbers
print("\n=== Table 2: condition numbers ===")
alpha = 1e-6
for p in [2, 4, 8, 16]:
    K, M, x = build_1d(p, E=2)
    H = alpha*np.eye(len(x)) + M @ np.linalg.solve(K, np.linalg.solve(K.T, M))
    print(f"p={p:2d}: kappa(K)={np.linalg.cond(K):.2e}, kappa(H)={np.linalg.cond(H):.2e}")

# Table 3: gradient convergence
print("\n=== Table 3: gradient FD error ===")
for p in [2, 4, 8]:
    K, M, x = build_1d(p, E=2)
    u_d = np.sin(np.pi*x)
    alpha = 1e-6
    def obj(f):
        u = np.linalg.solve(K, M@f)
        return 0.5*np.sum((u-u_d)**2) + alpha*0.5*np.sum(f**2)
    def grad(f):
        u = np.linalg.solve(K, M@f)
        return M @ np.linalg.solve(K.T, u-u_d) + alpha*f
    f0 = np.zeros(len(x))
    g_ad = grad(f0)
    h = 1e-6
    g_fd = np.zeros(len(x))
    for i in range(min(5, len(x))):
        fp = f0.copy(); fp[i] += h
        fm = f0.copy(); fm[i] -= h
        g_fd[i] = (obj(fp)-obj(fm))/(2*h)
    gerr = np.linalg.norm(g_ad[:5]-g_fd[:5])/np.linalg.norm(g_ad[:5])
    u = np.linalg.solve(K, M@(np.pi**2*np.sin(np.pi*x)))
    state_err = np.sqrt(np.mean((u-np.sin(np.pi*x))**2))
    print(f"p={p:2d}: state_err={state_err:.2e}, grad_FD_err={gerr:.2e}")
