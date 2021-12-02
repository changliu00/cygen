import torch as tc
from timeit import timeit

device = 'cpu'
number = 10000

x = tc.randn(30, 40, requires_grad=True, device=device)
w = tc.randn(30, 40, requires_grad=True, device=device)
eta = tc.randn(30, 40, requires_grad=False, device=device)

def run_proj_gradop():
    y = 1 / (1 + (1+(-w*x).exp()).log()**2 )
    g = tc.autograd.grad(y, x, eta, create_graph=True)[0]
    return tc.autograd.grad(g.sum(), w)[0]

def run_proj_inprod():
    y = 1 / (1 + (1+(-w*x).exp()).log()**2 )
    g = tc.autograd.grad((y*eta).sum(), x, create_graph=True)[0]
    return tc.autograd.grad(g.sum(), w)[0]

def run_sum_gradop():
    y = 1 / (1 + (1+(-w*x).exp()).log()**2 )
    g = tc.autograd.grad(y, x, tc.ones_like(y), create_graph=True)[0]
    return tc.autograd.grad(g.sum(), w)[0]

def run_sum_inprod():
    y = 1 / (1 + (1+(-w*x).exp()).log()**2 )
    g = tc.autograd.grad(y.sum(), x, create_graph=True)[0]
    return tc.autograd.grad(g.sum(), w)[0]

if __name__ == '__main__':
    print('Proj with eta:')
    print('Using `grad_outputs`:')
    print(timeit(lambda: run_proj_gradop(), number=number))
    print('Using `inner product`:')
    print(timeit(lambda: run_proj_inprod(), number=number))
    print(tc.allclose(run_proj_gradop(), run_proj_inprod()))

    print('Sum:')
    print('Using `grad_outputs`:')
    print(timeit(lambda: run_sum_gradop(), number=number))
    print('Using `inner product`:')
    print(timeit(lambda: run_sum_inprod(), number=number))
    print(tc.allclose(run_sum_gradop(), run_sum_inprod()))

