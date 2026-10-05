import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

xi = np.array([-1.0, -np.sqrt(3/7), 0.0, np.sqrt(3/7), 1.0])
w = np.array([1/10, 49/90, 32/45, 49/90, 1/10])
p = 4; n = p+1

c = 1.0/np.array([np.prod([xi[j]-xi[k] for k in range(n) if k!=j]) for j in range(n)])
D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i != j: D[i,j] = (c[j]/c[i])/(xi[i]-xi[j])
    D[i,i] = -np.sum(D[i,:])

# Consistent mass
from scipy.special import roots_legendre
xg, wg = roots_legendre(20)
L = np.zeros((len(xg), n))
for j in range(n):
    prod = np.ones(len(xg))
    for k in range(n):
        if k != j: prod *= (xg - xi[k])/(xi[j] - xi[k])
    L[:,j] = prod
Mc_ref = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        Mc_ref[i,j] = np.sum(wg * L[:,i] * L[:,j])
Ke_ref = D.T @ np.diag(w) @ D

def build(E):
    he = 1.0/E; Je = he/2
    Me = Je * Mc_ref
    Ke = Ke_ref / Je
    n_nodes = E*p+1
    K = np.zeros((n_nodes,n_nodes)); M = np.zeros((n_nodes,n_nodes))
    nodes = np.zeros(n_nodes)
    for e in range(E):
        idx = list(range(e*p,e*p+p+1))
        # Actual GLL positions on this element
        x_elem = e*he + Je*(xi + 1)
        for ii in range(p+1):
            nodes[idx[ii]] = x_elem[ii]
            for jj in range(p+1):
                K[idx[ii],idx[jj]] += Ke[ii,jj]
                M[idx[ii],idx[jj]] += Me[ii,jj]
    return K, M, nodes

print("Convergence:")
for E in [2, 5, 10, 20]:
    K, M, nodes = build(E)
    f_rhs = np.pi**2 * np.sin(np.pi*nodes)
    Kint = K[1:-1,1:-1]; Mint = M[1:-1,1:-1]
    rhs = Mint @ f_rhs[1:-1]
    u = np.linalg.solve(Kint, rhs)
    err = np.sqrt(np.mean((u-np.sin(np.pi*nodes[1:-1]))**2))
    print(f"  E={E:3d}, n_int={len(nodes)-2:4d}, RMS={err:.2e}")

# Optimization
E = 10; K, M, nodes = build(E)
Kint = K[1:-1,1:-1]; Mint = M[1:-1,1:-1]
u_d = np.sin(np.pi*nodes[1:-1])
alpha = 1e-6
errors = []; objs = []
def objective(f):
    u = np.linalg.solve(Kint, Mint @ f)
    J = 0.5*np.sum((u-u_d)**2) + alpha*0.5*np.sum(f**2)
    g = Mint @ np.linalg.solve(Kint.T, u-u_d) + alpha*f
    errors.append(np.sqrt(np.mean((u-u_d)**2)))
    objs.append(J)
    return J, g
f0 = np.zeros(len(nodes)-2)
res = minimize(objective, f0, method='L-BFGS-B', jac=True, options={'maxiter': 30, 'ftol': 1e-14})
print(f"\nOptimization: {len(errors)} iters, final error={errors[-1]:.4e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.semilogy(errors, 'b-o', markersize=3)
ax1.set_xlabel('Iteration'); ax1.set_ylabel('State RMS error')
ax1.set_title('L-BFGS convergence (SEM, p=4, E=10)'); ax1.grid(True, alpha=0.3)
ax2.semilogy(objs, 'r-o', markersize=3)
ax2.set_xlabel('Iteration'); ax2.set_ylabel('Objective J')
ax2.set_title('Objective decrease'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'H:\2026科研\OMS-WFC20261004\figures\fig_conv_opt.pdf', dpi=150)
print("Saved")
