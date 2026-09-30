import sys; sys.path.insert(0,'.')
import torch, numpy as np, time
from spectral_element import DifferentiableSEM1D

torch.manual_seed(42)
solver = DifferentiableSEM1D([(0.0,0.5),(0.5,1.0)], p=8)
n = solver.n_interior
x = solver.node_coords[solver.interior]
u_d_np = np.sin(np.pi*x)
u_d = torch.tensor(u_d_np, dtype=torch.float64)
alpha = 1e-6
K = np.array(solver.K_int)
M = np.array(solver.M_int)

# Hand adjoint
f = np.ones(n)*0.1
u = np.linalg.solve(K, M@f)
lam = np.linalg.solve(K.T, -(u - u_d_np))
g_hand = alpha*f - M@lam

# AD
ft = torch.tensor(f, dtype=torch.float64, requires_grad=True)
ut = solver.solve(ft)
loss = 0.5*((ut-u_d)**2).sum() + alpha*0.5*(ft**2).sum()
loss.backward()
g_ad = ft.grad.numpy()

rel_err = np.linalg.norm(g_hand - g_ad)/np.linalg.norm(g_ad)
print(f'Hand-adjoint vs AD gradient relative error: {rel_err:.2e}')

# Timing
t0=time.time()
for _ in range(100):
    u = np.linalg.solve(K, M@f)
    lam = np.linalg.solve(K.T, -(u-u_d_np))
    g = alpha*f - M@lam
t_hand=(time.time()-t0)/100*1000

t0=time.time()
for _ in range(100):
    ft2 = torch.tensor(f, dtype=torch.float64, requires_grad=True)
    ut2 = solver.solve(ft2)
    l = 0.5*((ut2-u_d)**2).sum() + alpha*0.5*(ft2**2).sum()
    l.backward()
t_ad=(time.time()-t0)/100*1000
print(f'Hand adjoint: {t_hand:.2f} ms/eval')
print(f'AD: {t_ad:.2f} ms/eval')
