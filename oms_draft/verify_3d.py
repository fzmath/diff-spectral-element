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
    return xi, M1d, K1d

def build_3d(p, E=2):
    xi, M1d, K1d = build_1d_mats(p)
    Je = 0.5/E
    Me = Je**3 * np.kron(np.kron(M1d,M1d),M1d)  # 3D mass
    # K = Je*(Kx*My*Mz + Mx*Ky*Mz + Mx*My*Kz)
    Ke3 = Je*(np.kron(np.kron(K1d,M1d),M1d) + np.kron(np.kron(M1d,K1d),M1d) + np.kron(np.kron(M1d,M1d),K1d))

    n1d = E*p+1
    n3d = n1d**3
    nodes_1d = np.zeros(n1d)
    for e in range(E):
        idx = list(range(e*p,e*p+p+1))
        nodes_1d[idx] = e/E + Je*(xi+1)

    K = np.zeros((n3d, n3d))
    M = np.zeros((n3d, n3d))
    for ex in range(E):
        for ey in range(E):
            for ez in range(E):
                idx = []
                for k in range(p+1):
                    for j in range(p+1):
                        for i in range(p+1):
                            idx.append((ez*p+k)*n1d**2 + (ey*p+j)*n1d + ex*p+i)
                for ii in range((p+1)**3):
                    for jj in range((p+1)**3):
                        K[idx[ii], idx[jj]] += Ke3[ii,jj]
                        M[idx[ii], idx[jj]] += Me[ii,jj]

    # boundary
    boundary = set()
    for i in range(n1d):
        for j in range(n1d):
            for k in range(n1d):
                x=nodes_1d[i]; y=nodes_1d[j]; z=nodes_1d[k]
                if x<1e-10 or x>1-1e-10 or y<1e-10 or y>1-1e-10 or z<1e-10 or z>1-1e-10:
                    boundary.add(k*n1d**2 + j*n1d + i)
    interior = sorted(set(range(n3d)) - boundary)
    xx, yy, zz = np.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing='ij')
    return K[np.ix_(interior,interior)], M[np.ix_(interior,interior)], \
           xx.flatten()[interior], yy.flatten()[interior], zz.flatten()[interior]

print("=== 3D Poisson ===")
for p in [2, 3, 4]:
    K, M, x, y, z = build_3d(p, E=2)
    f = 3*np.pi**2 * np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    u = np.linalg.solve(K, M@f)
    u_ex = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    err = np.sqrt(np.mean((u-u_ex)**2))
    kappa = np.linalg.cond(K)
    print(f"p={p}, n_int={len(x)}, L2={err:.3e}, kappa={kappa:.3e}")
