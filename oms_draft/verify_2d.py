import numpy as np
from scipy.special import roots_legendre, eval_legendre, legendre

def gll(p):
    P = legendre(p)
    dP = P.deriv()
    interior = np.sort(dP.roots.real)
    xi = np.concatenate([[-1.0], interior, [1.0]])
    w = np.array([2.0/(p*(p+1)*eval_legendre(p, x)**2) for x in xi])
    return xi, w

def build_1d_mats(p):
    xi, w = gll(p)
    n = p+1
    c = 1.0/np.array([np.prod([xi[j]-xi[k] for k in range(n) if k!=j]) for j in range(n)])
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j: D[i,j] = (c[j]/c[i])/(xi[i]-xi[j])
        D[i,i] = -np.sum(D[i,:])
    # consistent mass via high-order quadrature
    xg, wg = roots_legendre(20)
    L = np.zeros((len(xg), n))
    for j in range(n):
        prod = np.ones(len(xg))
        for k in range(n):
            if k != j: prod *= (xg - xi[k])/(xi[j] - xi[k])
        L[:,j] = prod
    M1d = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M1d[i,j] = np.sum(wg * L[:,i] * L[:,j])
    K1d = D.T @ np.diag(w) @ D
    return xi, w, M1d, K1d

def build_2d_square(p, E=2):
    """2D Poisson on (0,1)^2 with E x E elements."""
    xi, w, M1d, K1d = build_1d_mats(p)
    Je = 0.5/E  # Jacobian for each element
    # Element matrices (scaled)
    Me = Je * np.kron(M1d, M1d)  # mass
    Ke = (1/Je)*(np.kron(K1d, M1d) + np.kron(M1d, K1d))  # stiffness

    # Nodes: E*p+1 per direction
    n1d = E*p + 1
    n2d = n1d**2
    # Global node positions
    nodes_1d = np.zeros(n1d)
    for e in range(E):
        idx = list(range(e*p, e*p+p+1))
        nodes_1d[idx] = e/E + Je*(xi+1)

    K = np.zeros((n2d, n2d))
    M = np.zeros((n2d, n2d))
    for ex in range(E):
        for ey in range(E):
            idx = []
            for j in range(p+1):
                for i in range(p+1):
                    idx.append((ey*p+j)*n1d + ex*p+i)
            for ii in range((p+1)**2):
                for jj in range((p+1)**2):
                    K[idx[ii], idx[jj]] += Ke[ii,jj]
                    M[idx[ii], idx[jj]] += Me[ii,jj]

    # Boundary nodes (on edge of [0,1]^2)
    boundary = set()
    for i in range(n1d):
        for j in range(n1d):
            x = nodes_1d[i]; y = nodes_1d[j]
            if x < 1e-10 or x > 1-1e-10 or y < 1e-10 or y > 1-1e-10:
                boundary.add(j*n1d + i)
    interior = sorted(set(range(n2d)) - boundary)

    xx, yy = np.meshgrid(nodes_1d, nodes_1d)
    return K[np.ix_(interior,interior)], M[np.ix_(interior,interior)], xx.flatten()[interior], yy.flatten()[interior]

print("=== 2D Poisson: -Delta u = 2*pi^2 sin(pi x) sin(pi y) ===")
for p in [2, 4, 6, 8]:
    K, M, x, y = build_2d_square(p, E=2)
    f = 2*np.pi**2 * np.sin(np.pi*x)*np.sin(np.pi*y)
    u = np.linalg.solve(K, M@f)
    u_ex = np.sin(np.pi*x)*np.sin(np.pi*y)
    err = np.sqrt(np.mean((u-u_ex)**2))
    kappa = np.linalg.cond(K)
    print(f"p={p:2d}, n_int={len(x):4d}, L2={err:.3e}, kappa(K)={kappa:.3e}")
