import torch as tc
from methods.cygen import jacobian_normF2, laplacian, compatibility_jacnorm
from arch.mlp import MLP

test_type = "laplacian"

if test_type == "jacobian":
    dim_z = 4
    dim_x = 2
    dim_h = 2
    f_x1z = MLP([dim_z, dim_h, dim_x])

    z = tc.randn(dim_z, requires_grad=True)
    x = f_x1z(z)
    print(jacobian_normF2(x, z).data)
    x = f_x1z(z)
    print([jacobian_normF2(x, z, 1000).data for _ in range(1)])

elif test_type == "laplacian":
    dim_z = 100
    f = MLP([dim_z, 3, 2])
    z = tc.randn(dim_z, requires_grad=True)
    x = f(z).sum()
    print(laplacian(x, z).data)
    x = f(z).sum()
    print(laplacian(x, z, 1000).data)

elif test_type == "compatible":
    n_order = 3
    dim = 2
    sigma_x1z = 1.
    sigma_z1x = 1.
    n_sample = 100
    n_mc = 1
    n_iter = 100
    lr = 1e-2
    device = tc.device("cuda:0")

    param_x1z = tc.ones(n_order + 1, requires_grad=True, device=device)
    param_z1x = tc.ones(n_order + 1, requires_grad=True, device=device)
    optim = tc.optim.SGD([param_x1z, param_z1x], lr)
    def logp_x1z(x, z):
        return -(x - tc.tensordot(
                    param_x1z,
                    tc.stack([tc.ones_like(z), z] + [z**p for p in range(2, n_order+1)]),
                    dims=([0], [0])
                ))**2 / sigma_x1z**2 / 2

    def logq_z1x(z, x):
        return -(z - tc.tensordot(
                    param_z1x,
                    tc.stack([tc.ones_like(x), x] + [x**p for p in range(2, n_order+1)]),
                    dims=([0], [0])
                ))**2 / sigma_z1x**2 / 2

    z = tc.randn(n_sample, dim, requires_grad=True, device=device)
    x = tc.randn(n_sample, dim, requires_grad=True, device=device)
    for it in range(n_iter):
        optim.zero_grad()
        logp = logp_x1z(x, z)
        logq = logq_z1x(z, x)
        loss = compatibility_jacnorm(logp, logq, x, z, n_mc)
        loss.backward()
        optim.step()
        print(param_x1z.data, param_z1x.data)

