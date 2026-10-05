import numpy as np
from scipy.special import roots_legendre, eval_legendre, legendre

def gll_nodes_weights(p):
    if p == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    P = legendre(p)
    dP = P.deriv()
    interior = np.sort(dP.roots.real)
    xi = np.concatenate([[-1.0], interior, [1.0]])
    w = np.array([2.0/(p*(p+1)*eval_legendre(p, x)**2) for x in xi])
    return xi, w

def build_1d(p, E=2):
    xi, w = gll_nodes_weights(p)
    n = p+1
    c = 1.0/np.array([np.prod([xi[j]-xi[k] for k in range(n) if k!=j]) for j in range(n)])
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j: D[i,j] = (c[j]/c[i])/(xi[i]-xi[j])
        D[i,i] = -np.sum(D[i,:])
    # Consistent mass
    xg, wg = roots_legendre(20)
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
    return K[1:-1,1:-1], M[1:-1,1:-1], nodes[1:-1]

print("=== Table 1: 1D convergence (E=2, consistent mass) ===")
for p in [2, 4, 6, 8, 10, 16]:
    K, M, x = build_1d(p, E=2)
    f = np.pi**2 * np.sin(np.pi*x)
    u = np.linalg.solve(K, M@f)
    err = np.sqrt(np.mean((u-np.sin(np.pi*x))**2))
    kappa = np.linalg.cond(K)
    print(f"p={p:2d}, n={len(x):3d}, L2={err:.3e}, kappa={kappa:.3e}")

print("\n=== Table 2: condition numbers (E=2, alpha=1e-6) ===")
alpha = 1e-6
for p in [2, 4, 8, 16]:
    K, M, x = build_1d(p, E=2)
    H = alpha*np.eye(len(x)) + M @ np.linalg.solve(K, np.linalg.solve(K.T, M))
    print(f"p={p:2d}, kappa(K)={np.linalg.cond(K):.3e}, kappa(H)={np.linalg.cond(H):.3e}")

print("\n=== Table 2: h-refinement (p=4) ===")
for E in [2, 4, 8, 16]:
    K, M, x = build_1d(4, E=E)
    H = alpha*np.eye(len(x)) + M @ np.linalg.solve(K, np.linalg.solve(K.T, M))
    h = 1.0/E
    print(f"E={E:2d}, h={h:.4f}, kappa(K)={np.linalg.cond(K):.3e}, kappa(H)={np.linalg.cond(H):.3e}")
