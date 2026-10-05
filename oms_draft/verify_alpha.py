import numpy as np
from scipy.special import roots_legendre, eval_legendre, legendre
from scipy.optimize import minimize

def gll_nodes_weights(p):
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

# Table 4: alpha sensitivity, p=8
K, M, x = build_1d(8, E=2)
u_d = np.sin(np.pi*x)
print("alpha | state_err(%) | control_err(%) | obj")
for alpha in [1e-2, 1e-4, 1e-6, 1e-8]:
    def obj(f):
        u = np.linalg.solve(K, M@f)
        return 0.5*np.sum((u-u_d)**2) + alpha*0.5*np.sum(f**2)
    def grad(f):
        u = np.linalg.solve(K, M@f)
        return M @ np.linalg.solve(K.T, u-u_d) + alpha*f
    f0 = np.zeros(len(x))
    res = minimize(obj, f0, method='L-BFGS-B', jac=grad, options={'maxiter':100})
    f_opt = res.x
    u_opt = np.linalg.solve(K, M@f_opt)
    state_err = np.sqrt(np.mean((u_opt-u_d)**2))/np.sqrt(np.mean(u_d**2)) * 100
    # true control = pi^2 sin(pi x)
    f_true = np.pi**2 * np.sin(np.pi*x)
    ctrl_err = np.linalg.norm(f_opt-f_true)/np.linalg.norm(f_true)*100
    print(f"{alpha:.0e} | {state_err:.2f} | {ctrl_err:.2f} | {obj(f_opt):.4e}")
