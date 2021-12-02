#!/usr/bin/env python3.6
''' Multi-Layer Perceptron Architecture.
'''
import os
import json
import torch as tc
import torch.nn as nn

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

def iter_modules_params(*modules):
    for md in modules:
        if hasattr(md, 'parameters') and callable(md.parameters):
            for param in md.parameters():
                yield param

class ViewLayer(nn.Module):
    def __init__(self, shape):
        self.shape = tc.Size(shape)

    def forward(self, x):
        if x.ndim < 2: return x.view(*self.shape)
        else: return x.view(-1, *self.shape)

def mlp_constructor(dims, actv = "Sigmoid", lastactv = True): # `Sequential()`, or `Sequential(*[])`, is the identity map for any shape!
    if type(actv) is str: actv = getattr(nn, actv)
    if len(dims) <= 1: return nn.Sequential()
    else: return nn.Sequential(*(
        sum([[nn.Linear(dims[i], dims[i+1]), actv()] for i in range(len(dims)-2)], []) + \
        [nn.Linear(dims[-2], dims[-1])] + ([actv()] if lastactv else [])
    ))

class MLPBase(nn.Module):
    def save(self, path): tc.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(tc.load(path))
        self.eval()
    def load_or_save(self, filename):
        dirname = "init_models_mlp/"
        os.makedirs(dirname, exist_ok=True)
        path = dirname + filename
        if os.path.exists(path): self.load(path)
        else: self.save(path)

class MLP(MLPBase):
    def __init__(self, dims, actv = "Sigmoid"):
        if type(actv) is str: actv = getattr(nn, actv)
        super(MLP, self).__init__()
        self.f_x2y = mlp_constructor(dims, actv, lastactv = False)
    def forward(self, x): return self.f_x2y(x).squeeze(-1)

class MLPsvy1x(MLPBase):
    def __init__(self, dim_x, dims_postx2prev, dim_v, dim_parav, dims_postv2s, dims_posts2prey, dim_y, actv = "Sigmoid"):
        if type(actv) is str: actv = getattr(nn, actv)
        dim_prev, dim_s = dims_postx2prev[-1], dims_postv2s[-1]
        super(MLPsvy1x, self).__init__()
        self.dim_x, self.dim_v, self.dim_s, self.dim_y = dim_x, dim_v, dim_s, dim_y
        self.shape_x, self.shape_v, self.shape_s = (dim_x,), (dim_v,), (dim_s,)
        self.dims_postx2prev, self.dim_parav, self.dims_postv2s, self.dims_posts2prey, self.actv \
                = dims_postx2prev, dim_parav, dims_postv2s, dims_posts2prey, actv
        self.f_x2prev = mlp_constructor([dim_x] + dims_postx2prev, actv)
        self.f_prev2v = nn.Sequential( nn.Linear(dim_prev, dim_v), actv() )
        self.f_prev2parav = nn.Sequential( nn.Linear(dim_prev, dim_parav), actv() )
        self.f_vparav2s = mlp_constructor([dim_v + dim_parav] + dims_postv2s, actv)
        self.f_s2y = mlp_constructor([dim_s] + dims_posts2prey + [dim_y], actv, lastactv = False)

    def v1x(self, x): return self.f_prev2v(self.f_x2prev(x))
    def s1vx(self, v, x):
        parav = self.f_prev2parav(self.f_x2prev(x))
        return self.f_vparav2s(tc.cat([v, parav], dim=-1))
    def s1x(self, x):
        prev = self.f_x2prev(x)
        v = self.f_prev2v(prev)
        parav = self.f_prev2parav(prev)
        return self.f_vparav2s(tc.cat([v, parav], dim=-1))
    def y1s(self, s): return self.f_s2y(s).squeeze(-1) # squeeze for binary y

    def ys1x(self, x):
        s = self.s1x(x)
        return self.y1s(s), s
    def forward(self, x):
        return self.y1s(self.s1x(x))

class MLPx1sv(MLPBase):
    def __init__(self, dim_s = None, dims_pres2parav = None, dim_v = None, dims_prev2postx = None, dim_x = None,
            actv = None, *, discr = None):
        if dim_s is None: dim_s = discr.dim_s
        if dim_v is None: dim_v = discr.dim_v
        if dim_x is None: dim_x = discr.dim_x
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        if dims_pres2parav is None: dims_pres2parav = discr.dims_postv2s[::-1][1:] + [discr.dim_parav]
        if dims_prev2postx is None: dims_prev2postx = discr.dims_postx2prev[::-1]
        super(MLPx1sv, self).__init__()
        self.dim_s, self.dim_v, self.dim_x = dim_s, dim_v, dim_x
        self.dims_pres2parav, self.dims_prev2postx, self.actv = dims_pres2parav, dims_prev2postx, actv
        self.f_s2parav = mlp_constructor([dim_s] + dims_pres2parav, actv)
        self.f_vparav2x = mlp_constructor([dim_v + dims_pres2parav[-1]] + dims_prev2postx + [dim_x], actv)

    def x1sv(self, s, v): return self.f_vparav2x(tc.cat([v, self.f_s2parav(s)], dim=-1))
    def forward(self, s, v): return self.x1sv(s, v)

class MLPx1s(MLPBase):
    def __init__(self, dim_s = None, dims_pres2postx = None, dim_x = None,
            actv = None, *, discr = None):
        if dim_s is None: dim_s = discr.dim_s
        if dim_x is None: dim_x = discr.dim_x
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        if dims_pres2postx is None:
            dims_pres2postx = discr.dims_postv2s[::-1][1:] + [discr.dim_v + discr.dim_parav] + discr.dims_postx2prev[::-1]
        super(MLPx1s, self).__init__()
        self.dim_s, self.dim_x, self.dims_pres2postx, self.actv = dim_s, dim_x, dims_pres2postx, actv
        self.f_s2x = mlp_constructor([dim_s] + dims_pres2postx + [dim_x], actv)

    def x1s(self, s): return self.f_s2x(s)
    def forward(self, s): return self.x1s(s)

class MLPv1s(MLPBase):
    def __init__(self, dim_s = None, dims_pres2postv = None, dim_v = None,
            actv = None, *, discr = None):
        if dim_s is None: dim_s = discr.dim_s
        if dim_v is None: dim_v = discr.dim_v
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        if dims_pres2postv is None: dims_pres2postv = discr.dims_postv2s[::-1][1:]
        super(MLPv1s, self).__init__()
        self.dim_s, self.dim_v, self.dims_pres2postv, self.actv = dim_s, dim_v, dims_pres2postv, actv
        self.f_s2v = mlp_constructor([dim_s] + dims_pres2postv + [dim_v], actv)

    def v1s(self, s): return self.f_s2v(s)
    def forward(self, s): return self.v1s(s)

def create_discr_from_json(stru_name: str, dim_x: int, dim_y: int, actv: str=None, jsonfile: str="mlpstru.json"):
    stru = json.load(open(jsonfile))['MLPsvy1x'][stru_name]
    if actv is not None: stru['actv'] = actv
    return MLPsvy1x(dim_x=dim_x, dim_y=dim_y, **stru)

def create_gen_from_json(model_type: str="MLPx1sv", discr: MLPsvy1x=None, stru_name: str=None, dim_x: int=None, actv: str=None, jsonfile: str="mlpstru.json"):
    if stru_name is None:
        return eval(model_type)(dim_x=dim_x, discr=discr, actv=actv)
    else:
        stru = json.load(open(jsonfile))[model_type][stru_name]
        if actv is not None: stru['actv'] = actv
        return eval(model_type)(dim_x=dim_x, discr=discr, **stru)

