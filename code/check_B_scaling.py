"""Check the p,h scaling of mu_max(B), mu_min(B), kappa(H) in 1D."""
import numpy as np
from sem_lib import build_1d

def stats(p, E, alpha=1e-12):
    K, M, nodes, Dg = build_1d(p, E)
    # M is diagonal (GLL quadrature): B = M^{-1/2} K M^{-1/2} with D=diag(sqrt(M))
    d = np.sqrt(np.diag(M))
    B = K / np.outer(d, d)
    mu = np.linalg.eigvalsh(B)
    Kinv = np.linalg.inv(K)
    H = alpha * np.eye(K.shape[0]) + M @ Kinv.T @ Kinv @ M
    kapH = np.linalg.cond(H)
    return mu.max(), mu.min(), kapH

print("p-refinement, E=2 (h=0.5 fixed):")
for p in [2, 4, 8, 16]:
    mumax, mumin, kapH = stats(p, 2)
    print(f"  p={p:3d}  mu_max={mumax:12.4e}  mu_min={mumin:12.4e}  kappa(H,a=1e-12)={kapH:12.4e}")
print("h-refinement, p=4:")
for E in [2, 4, 8, 16]:
    mumax, mumin, kapH = stats(4, E)
    print(f"  E={E:3d}  mu_max={mumax:12.4e}  mu_min={mumin:12.4e}  kappa(H,a=1e-12)={kapH:12.4e}")
