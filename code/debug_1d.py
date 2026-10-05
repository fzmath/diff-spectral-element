import numpy as np, torch
from sem_lib import build_1d
torch.set_default_dtype(torch.float64)
K, M, nodes, Dg = build_1d(8, 2)
print("n_int =", K.shape[0])
print("nodes:", nodes)
x = torch.tensor(nodes[1:-1])
u_ex = torch.sin(torch.pi * x)
f = 2.0 * torch.pi ** 2 * u_ex
Kt, Mt = torch.tensor(K), torch.tensor(M)
u = torch.linalg.solve(Kt, Mt @ f)
print("u[:6]  :", u[:6].numpy())
print("u_ex[:6]:", u_ex[:6].numpy())
print("err:", float(torch.norm(u - u_ex) / torch.norm(u_ex)))
# check equation residual: K u = M f ?
res = Kt @ u - Mt @ f
print("eq residual:", float(torch.norm(res)))
print("K[0,:5]:", K[0,:5])
print("M diag :", np.diag(M))
