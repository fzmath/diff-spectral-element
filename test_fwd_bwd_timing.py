import sys; sys.path.insert(0,'.')
import torch, numpy as np, time
from spectral_element import DifferentiableSEM1D

print("n_elem | n_int | fwd_ms | bwd_ms | fwd+bwd_ms")
for n_elem in [10, 20, 50, 100, 200, 500]:
    elems = [(i/n_elem, (i+1)/n_elem) for i in range(n_elem)]
    solver = DifferentiableSEM1D(elems, p=4)
    n = solver.n_interior
    x = solver.node_coords[solver.interior]
    u_d = torch.tensor(np.sin(np.pi*x), dtype=torch.float64)
    alpha = 1e-6

    def obj(f):
        u = solver.solve(f)
        return 0.5*((u-u_d)**2).sum() + alpha*0.5*(f**2).sum()

    # Warmup
    f0 = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    obj(f0).backward()

    # Forward timing
    t0 = time.time()
    for _ in range(20):
        f = torch.zeros(n, dtype=torch.float64, requires_grad=True)
        u = solver.solve(f)
        loss = 0.5*((u-u_d)**2).sum() + alpha*0.5*(f**2).sum()
    fwd_ms = (time.time()-t0)/20*1000

    # Forward+backward timing
    t0 = time.time()
    for _ in range(20):
        f = torch.zeros(n, dtype=torch.float64, requires_grad=True)
        loss = obj(f)
        loss.backward()
    total_ms = (time.time()-t0)/20*1000
    bwd_ms = total_ms - fwd_ms

    print(f"{n_elem} | {n} | {fwd_ms:.2f} | {bwd_ms:.2f} | {total_ms:.2f}")
