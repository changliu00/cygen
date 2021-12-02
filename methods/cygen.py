#!/usr/bin/env python3.6
import math
from contextlib import suppress
import warnings
import torch as tc

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

###### Basic tools ######

NoneVars = ["None", "none", None, "Null", "null", "No", "no", "False", "false", False, "0", 0]

def solve_vec(b, A):
    return tc.solve(b.unsqueeze(-1), A)[0].squeeze(-1)

def smart_grad(outputs, inputs, grad_outputs = None, retain_graph = None, create_graph = None,
        only_inputs = True, allow_unused = False):
    if create_graph is None: create_graph = tc.is_grad_enabled()
    if retain_graph is None: retain_graph = create_graph
    gradients = tc.autograd.grad(outputs, inputs, grad_outputs, retain_graph, create_graph, only_inputs, allow_unused)
    if isinstance(inputs, tc.Tensor): return gradients[0]
    else: return gradients

def track_var(var: tc.Tensor, track: bool):
    if track: return var if var.requires_grad else var.detach().requires_grad_(True)
    else: return var.detach() if var.requires_grad else var

###### Jacobian and Laplacian ######

def jacobian_normF2(y, x, n_mc = 0):
    """
    `y` is vector-valued. `x` requires grad.
    If `n_mc` == 0, use exact calculation. If `n_mc` > 0, use the Hutchinson's estimator.
    """
    if n_mc > 0: # Hutchinson's estimator
        ls_gradx_yproj = [smart_grad(y, x, grad_outputs=tc.randn_like(y), retain_graph=True)
                for _ in range(n_mc)]
        ls_quad = [(gradx_yproj**2).sum() for gradx_yproj in ls_gradx_yproj]
        return tc.stack(ls_quad).mean()
    elif n_mc == 0: # exact calculation
        with tc.enable_grad(): ls_y = y.flatten().unbind()
        return tc.stack([(smart_grad(yi, x, retain_graph=True)**2).sum() for yi in ls_y]).sum()

def jacobian(y, x, ndim_batch = 0):
    """
    `y` is vector-valued. `x` requires grad.
    `y.shape` = shape_batch + shape_y, `x.shape` = shape_batch + shape_x,
    where len(shape_batch) = ndim_batch.
    Output shape: shape_batch + shape_x + shape_y.
    This is the 'transpose' to the functional form `autograd.functional.jacobian`.
    """
    shape_batch = y.shape[:ndim_batch]
    assert shape_batch == x.shape[:ndim_batch]
    shape_y = y.shape[ndim_batch:]
    shape_x = x.shape[ndim_batch:]

    with tc.enable_grad():
        y_sum = y.sum(dim=tuple(range(ndim_batch))) if ndim_batch else y
        ls_y = y_sum.flatten().unbind()
    jac_flat = tc.stack([smart_grad(yi, x, retain_graph=True) for yi in ls_y], dim=-1) # shape_batch + shape_x + (shape_y.prod(),)
    return jac_flat.reshape(shape_batch + shape_x + shape_y)

def directional_jacobian(y, x, v):
    """
    Vector-Jacobian product `( v(x)^T (grad_x y(x)^T) )^T`,
    or the tensor form of `tc.autograd.functional.jvp` (their Jacobian is transposed).
    Based on `grad_eta (v(x)^T grad_x (eta^T y(x))) = grad_eta (v(x)^T (grad_x y(x)^T) eta) = ( v(x)^T (grad_x y(x)^T) )^T`.
    """
    eta = tc.zeros_like(y, requires_grad=True) # The value of `eta` does not matter
    with tc.enable_grad():
        gradx_yproj = smart_grad(y, x, grad_outputs=eta)
    return smart_grad(gradx_yproj, eta, grad_outputs=v)

def laplacian(y, x, n_mc = 0, gradx_y = None):
    """
    `y` is a scalar, or is summed up first. `x` requires grad.
    `gradx_y` overwrites `y`. It should have the same shape as `x`.
    If `n_mc` == 0, use exact calculation. If `n_mc` > 0, use the Hutchinson's estimator.
    """
    if gradx_y is None:
        with tc.enable_grad(): gradx_y = smart_grad(y.sum(), x)
    if n_mc > 0: # Hutchinson's estimator
        ls_eta = [tc.randn_like(gradx_y) for _ in range(n_mc)]
        ls_quad = [(smart_grad(gradx_y, x, grad_outputs=eta, retain_graph=True) * eta).sum() for eta in ls_eta]
        return tc.stack(ls_quad).mean()
    elif n_mc == 0: # exact calculation
        with tc.enable_grad(): ls_gradx_y = gradx_y.flatten().unbind()
        return tc.stack([smart_grad(gradxi_y, x, retain_graph=True).flatten()[i]
                for i, gradxi_y in enumerate(ls_gradx_y)]).sum()

###### Losses in standalone form ######

def compatibility_jacnorm(logp, logq, x, z, n_mc = 1):
    """ `x` and `z` require grad. All inputs have a matching batch size. """
    with tc.enable_grad():
        gradx_logratio = smart_grad(logp.sum() - logq.sum(), x)
    return jacobian_normF2(gradx_logratio, z, n_mc) / logp.numel()

###### Losses assembled: avoid repeated gradient evaluation ######

class SampleGradContainer:
    def __init__(self, eval_logp, eval_logq, draw_rho):
        for k,v in locals().items():
            if k != "self": setattr(self, k, v)
        self._x = self._z = self._logp_sum = self._logq_sum = None
        self._n_batch = self._shape_batch = None
        self._dc_grad = dict()
        self._has_evaled = False

    def reset(self):
        self._has_evaled = False

    def raise_if_hasnot_evaled(self):
        if not self._has_evaled: raise RuntimeError("Do `eval_if_hasnot(x)` first")

    def eval_if_hasnot(self, x, z = None, track_x = True, track_z = True):
        if self._has_evaled: return
        if z is None:
            with tc.no_grad():
                z = self.draw_rho(x, 1).squeeze(0) # without reparameterization
        with tc.enable_grad():
            x = track_var(x, track_x); z = track_var(z, track_z)
            logp = self.eval_logp(x,z)
            self._logp_sum = logp.sum()
            self._logq_sum = self.eval_logq(z,x).sum()
        self._x = x; self._z = z
        self._n_batch = logp.numel()
        self._shape_batch = logp.shape
        self._dc_grad = dict(x_logp=None, z_logp=None, x_logq=None, z_logq=None)
        self._has_evaled = True

    def get_grad(self, query):
        self.raise_if_hasnot_evaled()
        res = self._dc_grad[query]
        if res is None:
            res = smart_grad(getattr(self, query[1:]+"_sum"), getattr(self, "_"+query[0]), retain_graph=True)
            self._dc_grad[query] = res
        return res

    # def n_batch(self):
    #     self.raise_if_hasnot_evaled()
    #     return self._n_batch

    # def get_var(self, var_name):
    #     self.raise_if_hasnot_evaled()
    #     if var_name == "x": return self._x
    #     elif var_name == "z": return self._z
    #     else: raise KeyError(f"unknown `var_name` {var_name}")

class CyGen:
    def __init__(self, cmtype, pxtype, eval_logp, eval_logq,
            draw_q = None, draw_rho = None,
            w_cm = 1., w_px = 1.,
            n_mc_cm = 1, n_mc_px = 1
        ):
        for k,v in locals().items():
            if k != "self": setattr(self, k, v)
        self._sgc_q = SampleGradContainer(eval_logp, eval_logq, draw_q) if draw_q is not None else None
        self._sgc_rho = self._sgc_q if (
                draw_rho is draw_q or draw_rho is None
            ) else SampleGradContainer(eval_logp, eval_logq, draw_rho)

    def compatibility(self, x):
        if self.cmtype in NoneVars: return tc.zeros([])
        elif self.cmtype in {"jacnorm_x", "jacnorm_z"}:
            sgc = self._sgc_rho
            sgc.eval_if_hasnot(x)
            var1st = self.cmtype[-1]
            var2nd = sgc._z if var1st == "x" else sgc._x
            with tc.enable_grad():
                grad1_logratio = sgc.get_grad(var1st+"_logp") - sgc.get_grad(var1st+"_logq")
            return jacobian_normF2(grad1_logratio, var2nd, self.n_mc_cm) / sgc._n_batch
        else: raise KeyError(f"unknown `cmtype` '{self.cmtype}'")

    def pxmatch(self, x):
        if self.pxtype in NoneVars: return tc.zeros([])
        elif self.pxtype == "nllhmarg":
            return -self.llhmarg(x, self.n_mc_px).mean()
        else: raise KeyError(f"unknown `pxtype` '{self.pxtype}'")

    def getlosses(self, x):
        self._sgc_rho.reset(); self._sgc_q.reset()
        cmloss = self.compatibility(x)
        pxloss = self.pxmatch(x)
        return self.w_cm * cmloss + self.w_px * pxloss, cmloss, pxloss

    def getloss(self, x):
        return self.getlosses(x)[0]

    def llhmarg(self, x, n_mc = None):
        if n_mc is None: n_mc = self.n_mc_px
        z = self.draw_q(x, n_mc) # (n_mc, n_batch). Exactly `draw_q`. With reparameterization
        logp_stackz = self.eval_logp(x, z) # (n_mc, n_batch). Broadcast for `x.expand((n_mc,)+x.shape)`
        logpx = -tc.logsumexp(-logp_stackz, dim=0) + math.log(n_mc)
        return logpx

    def generate(self, gentype, draw_p, n_iter = None, z0 = None, x0 = None,
            stepsize = 1e-3, anneal = None, freeze = None, *, x_range = None, no_grad = True
        ):
        ''' Number of samples is determined by `z0` or `x0`.
            `stepsize`, `anneal`, `freeze` only effective for 'langv-z' and 'langv-x'.
            `x_range` only effective for 'langv-x'. For other `gentype`, `draw_p` is responsible for the range.
        '''
        if gentype == "gibbs":
            with tc.no_grad() if no_grad else suppress():
                z = self.draw_q(x0, 1).squeeze(0) if z0 is None else z0
                for _ in range(n_iter):
                    x = draw_p(z, 1).squeeze(0)
                    z = self.draw_q(x, 1).squeeze(0)
                x = draw_p(z, 1).squeeze(0)
        elif gentype == "xhat":
            x, z = self.generate("gibbs", draw_p, 0, z0, x0, no_grad=no_grad)
        elif gentype in {"langv-z", "langv-x"}:
            var0, varname, varcond, other0, othername, othercond = (
                x0, "x", "p", z0, "z", "q") if gentype[-1] == "x" else (
                z0, "z", "q", x0, "x", "p")
            drawvar, drawother = (draw_p, self.draw_q) if varname == "x" else (self.draw_q, draw_p)
            clamp_x = (varname == "x") and (x_range is not None)
            with tc.no_grad() if no_grad else suppress():
                if tc.is_grad_enabled(): warnings.warn("behavior undefined when `is_grad_enabled`")
                var = drawvar(other0, 1).squeeze(0) if var0 is None else var0.clone() # If `no_grad`, then no need of `detach`
                sgc = self._sgc_q
                for itr in range(n_iter):
                    eval_args = {"x": var} if varname == "x" else {
                            "z": var, "x": drawother(var, 1).squeeze(0), "track_x": False}
                    sgc.reset(); sgc.eval_if_hasnot(**eval_args)
                    gradvar_logpvar = sgc.get_grad(varname+"_log"+varcond) - sgc.get_grad(varname+"_log"+othercond) # Won't create graph (no such need)
                    anneal_fac = 1. - math.exp(-(anneal/n_iter) * (itr+1)) if bool(anneal) and anneal > 0 else 1.
                    freeze_fac = math.exp(-(freeze/n_iter) * (itr+1)) if bool(freeze) and freeze > 0 else 1.
                    delta_var = (anneal_fac * stepsize) * gradvar_logpvar + math.sqrt(freeze_fac * 2*stepsize) * tc.randn_like(var)
                    if tc.is_grad_enabled():
                        var = var + delta_var
                        if clamp_x: var = var.clamp(*x_range)
                    else:
                        var += delta_var
                        if clamp_x: var.clamp_(*x_range)
                other = drawother(var, 1).squeeze(0)
                x, z = (var, other) if varname == "x" else (other, var)
        return x, z

class DAE(CyGen):
    def __init__(self, eval_logp, draw_p, draw_q, n_gibbs = 0):
        CyGen.__init__(self, "None", "None", eval_logp, None, draw_q=draw_q)
        self.draw_p, self.n_gibbs = draw_p, n_gibbs

    def getlosses(self, x):
        x, z = self.generate("gibbs", self.draw_p, n_iter=self.n_gibbs, z0=None, x0=x, no_grad=False)
        return -self.eval_logp(x, z).mean(),

    def getloss(self, x): return self.getlosses(x)[0]

###### Losses assembled: for flows with no tractable inverse evaluation ######

class SampleGradContainer_FlowqNoInv:
    def __init__(self, eval_logp, draw_q0, eval_z1eps_logqt):
        """
        To draw a sample `z` given `x` and evaluate `log q(z|x)` on this `z` (i.e., `eval_if_hasnot(x)`),
        - Draw an `eps` from `q0(.|x)` with `no_grad`, then `requires_grad`.
          Use `eval_z1eps_logqt(x,eps)` to foward and evaluate this `eps`:
        - `z = f(x,eps)` is the flow output. `f(x,.)` is invertible, but the inverse `eps = f^{-1}(x,z)` is intractable.
        - `log qt(x,eps) := log q(z=f(x,eps)|x) = log q0(eps|x) - log |jac_eps f(x,eps)|`.
        - (Optional) `jac_eps f(x,eps)` if feasible from the forward process. Otherwise use autograd.
        Note: For functions `h(x,z)` and `z=f(x,eps)`, let `ht(x,eps) := h(x,z=f(x,eps))`. Then,
          `[grad_z h(x,z)]|_(x,f(x,eps)) = [jac_eps f(x,eps)]^{-1} [grad_eps ht(x,eps)]`,
          `[grad_x h(x,z)]|_(x,f(x,eps)) = [grad_x ht(x,eps)] - [jac_x f(x,eps)] [grad_z h(x,z)]|_(x,f(x,eps))`.
        """
        for k,v in locals().items():
            if k != "self": setattr(self, k, v)
        self._x = self._z = self._eps = None
        self._logp_sum = self._logqt_sum = None
        self._n_batch = self._shape_batch = None
        self._dc_grad = dict()
        self._has_evaled = False

    def reset(self):
        self._has_evaled = False

    def raise_if_hasnot_evaled(self):
        if not self._has_evaled: raise RuntimeError("Do `eval_if_hasnot(x)` first")

    def eval_if_hasnot(self, x, eps = None, track_x = True, track_eps = True, track_z = True):
        if self._has_evaled: return
        if eps is None:
            with tc.no_grad():
                eps = self.draw_q0(x, 1).squeeze(0) # without reparameterization
        with tc.enable_grad():
            x = track_var(x, track_x); eps = track_var(eps, track_eps)
            out = self.eval_z1eps_logqt(x, eps)
            self._out_z, logqt = out[:2]
            self._logqt_sum = logqt.sum()
            z = track_var(self._out_z, track_z)
            self._logp_sum = self.eval_logp(x,z).sum()
        self._x = x; self._z = z; self._eps = eps
        self._n_batch = logqt.numel()
        self._shape_batch = logqt.shape
        self._dc_grad = dict(x_logp=None, z_logp=None, x_logq=None, z_logq=None,
                x_logqt=None, eps_logqt=None, eps_z=None) # `x_z` is evaluated with a vector product
        if len(out) > 2: self._dc_grad['eps_z'] = out[2]
        self._has_evaled = True

    def get_grad(self, query):
        self.raise_if_hasnot_evaled()
        res = self._dc_grad[query]
        if res is None:
            if query in {"x_logp", "z_logp", "x_logqt", "eps_logqt"}:
                pos = query.find("_")
                res = smart_grad(getattr(self, query[pos:]+"_sum"), getattr(self, "_"+query[:pos]), retain_graph=True)
            elif query == "eps_z":
                res = jacobian(self._out_z, self._eps, len(self._shape_batch)) # shape_batch + shape_eps + shape_z
            elif query == "z_logq":
                res = solve_vec(self.get_grad("eps_logqt"), self.get_grad("eps_z")) # Requires 1-dim `eps` and `z`. Grad tracked. shape_batch + (dim_z,)
            elif query == "x_logq":
                res = self.get_grad("x_logqt") - smart_grad(
                        self._out_z, self._x, grad_outputs=self.get_grad("z_logq"), retain_graph=True ) # It is `(grad_x z^T) grad_outputs`, but not `grad_x (z^T grad_outputs)`
            else: pass
            self._dc_grad[query] = res
        return res

class CyGen_FlowqNoInv(CyGen):
    def __init__(self, cmtype, pxtype, eval_logp, draw_q0, eval_z1eps_logqt, eval_z1eps = None,
            w_cm = 1., w_px = 1.,
            n_mc_cm = 1, n_mc_px = 1
        ):
        CyGen.__init__(self,
                **{k:v for k,v in locals().items() if k not in {"self", "__class__", "draw_q0", "eval_z1eps_logqt", "eval_z1eps"}},
                eval_logq=None, draw_q=None, draw_rho=None )
        self.draw_q0 = draw_q0
        self.eval_z1eps_logqt = eval_z1eps_logqt
        self.eval_z1eps = eval_z1eps
        self._sgc_rho = self._sgc_q = SampleGradContainer_FlowqNoInv(eval_logp, draw_q0, eval_z1eps_logqt)
        if eval_z1eps is None:
            self.draw_q = lambda x, n_mc: eval_z1eps_logqt(x, draw_q0(x, n_mc))[0]
        else:
            self.draw_q = lambda x, n_mc: eval_z1eps(x, draw_q0(x, n_mc))

    def compatibility(self, x):
        if self.cmtype in NoneVars: return tc.zeros([])
        elif self.cmtype in {"jacnorm_x", "jacnorm_z"}:
            sgc = self._sgc_q
            sgc.eval_if_hasnot(x)
            var1st = self.cmtype[-1]
            var2nd = sgc._z if var1st == "x" else sgc._x
            with tc.enable_grad(): grad1_logp = sgc.get_grad(var1st+"_logp")
            ls_eta = [tc.randn_like(grad1_logp) for _ in range(self.n_mc_cm)] # `eta` for logp and logq should be the same
            ls_grad2__grad1_logp_proj = [smart_grad(
                    grad1_logp, var2nd, grad_outputs=eta, retain_graph=True
                ) for eta in ls_eta]
            if var1st == "x":
                with tc.enable_grad(): gradx_logq = sgc.get_grad("x_logq")
                ls_grad2__grad1_logq_proj = [solve_vec(
                        smart_grad(gradx_logq, sgc._eps, grad_outputs=eta, retain_graph=True),
                        sgc.get_grad("eps_z")
                    ) for eta in ls_eta]
            else:
                with tc.enable_grad(): gradz_logq = sgc.get_grad("z_logq")
                ls_gradx_and_gradeps__gradz_logq_proj_t = [smart_grad(
                        gradz_logq, [sgc._x, sgc._eps], grad_outputs=eta, retain_graph=True
                    ) for eta in ls_eta]
                ls_grad2__grad1_logq_proj = [ gradx__gradz_logq_proj_t - smart_grad(
                        sgc._out_z, sgc._x, grad_outputs=solve_vec(
                            gradeps__gradz_logq_proj_t, sgc.get_grad("eps_z") ), retain_graph=True
                    ) for gradx__gradz_logq_proj_t, gradeps__gradz_logq_proj_t in ls_gradx_and_gradeps__gradz_logq_proj_t]
            ls_quad = [((grad2__grad1_logp_proj - grad2__grad1_logq_proj) ** 2).sum()
                    for grad2__grad1_logp_proj, grad2__grad1_logq_proj in zip(ls_grad2__grad1_logp_proj, ls_grad2__grad1_logq_proj)]
            return tc.stack(ls_quad).mean() / sgc._n_batch
        elif self.cmtype == "jacnorm_lite_x":
            sgc = self._sgc_q
            sgc.eval_if_hasnot(x)
            with tc.enable_grad(): gradx_logp = sgc.get_grad("x_logp")
            ls_eta = [tc.randn_like(gradx_logp) for _ in range(self.n_mc_cm)] # `eta` for logp and logq should be the same
            with tc.no_grad(): jaceps_z_stopgrad = sgc.get_grad("eps_z").detach() # `detach` here even `no_grad` since the output of `jacobian` may still be in a graph
            ls_J_gradz__gradx_logp_proj = [( jaceps_z_stopgrad @ smart_grad( # `@` here requires 1-dim `eps` and `z`.
                    gradx_logp, sgc._z, grad_outputs=eta, retain_graph=True).unsqueeze(-1)
                ).squeeze(-1) for eta in ls_eta]
            with tc.enable_grad(): gradx_logq = sgc.get_grad("x_logq")
            ls_Jinv_gradz__gradx_logq_proj = [smart_grad(
                    gradx_logq, sgc._eps, grad_outputs=eta, retain_graph=True
                ) for eta in ls_eta]
            ls_quad = [((J_gradz__gradx_logp_proj - Jinv_gradz__gradx_logq_proj) ** 2).sum()
                    for J_gradz__gradx_logp_proj, Jinv_gradz__gradx_logq_proj in zip(ls_J_gradz__gradx_logp_proj, ls_Jinv_gradz__gradx_logq_proj)]
            return tc.stack(ls_quad).mean() / sgc._n_batch
        else: raise KeyError(f"unknown `cmtype` '{self.cmtype}'")

    # `pxmatch`, `getlosses`, `getloss` and `llhmarg` are the same.

    def generate(self, gentype, draw_p, n_iter = None, z0 = None, x0 = None,
            stepsize = 1e-3, anneal = None, freeze = None, eps0 = None, *, x_range = None, no_grad = True
        ):
        ''' Number of samples is determined by `z0` or `x0`.
            `stepsize`, `anneal`, `freeze` only effective for 'langv-z' and 'langv-x'.
            `x_range` only effective for 'langv-x'. For other `gentype`, `draw_p` is responsible for the range.
            `eps0` only effective for 'langv-z'.
        '''
        if gentype in {"gibbs", "xhat", "langv-x"}:
            x, z = CyGen.generate(self, gentype, draw_p, n_iter, z0, x0, stepsize, anneal, freeze, x_range=x_range, no_grad=no_grad)
        elif gentype == "langv-z":
            """
            Requires a fixed `x0` to run in the `eps` space.
            """
            with tc.no_grad() if no_grad else suppress():
                if tc.is_grad_enabled(): warnings.warn("behavior undefined when `is_grad_enabled`")
                with tc.no_grad(): # `no_grad` even when `is_grad_enabled`
                    x0 = draw_p(z0, 1).squeeze(0) if x0 is None else track_var(x0, False)
                eps = self.draw_q0(x0, 1).squeeze(0) if eps0 is None else eps0.clone() # If `no_grad`, then no need of `detach`
                sgc = self._sgc_q
                for itr in range(n_iter):
                    sgc.reset(); sgc.eval_if_hasnot(**{"x": x0, "eps": eps, "track_x": False})
                    gradvar_logpvar = sgc.get_grad("z_logq") - sgc.get_grad("z_logp") # Won't create graph (no such need)
                    anneal_fac = 1. - math.exp(-(anneal/n_iter) * (itr+1)) if bool(anneal) and anneal > 0 else 1.
                    freeze_fac = math.exp(-(freeze/n_iter) * (itr+1)) if bool(freeze) and freeze > 0 else 1.
                    delta_z = (anneal_fac * stepsize) * gradvar_logpvar + math.sqrt(freeze_fac * 2*stepsize) * tc.randn_like(sgc._z)
                    delta_eps = solve_vec(delta_z, sgc.get_grad("eps_z"))
                    if tc.is_grad_enabled(): eps = eps + delta_eps
                    else: eps += delta_eps
                z = self.eval_z1eps_logqt(x0, eps)[0] if self.eval_z1eps is None else self.eval_z1eps(x0, eps)
                x = draw_p(z, 1).squeeze(0)
        return x, z

class DAE_FlowqNoInv(CyGen_FlowqNoInv):
    def __init__(self, eval_logp, draw_p, draw_q0, eval_z1eps_logqt = None, eval_z1eps = None, n_gibbs = 0):
        CyGen_FlowqNoInv.__init__(self, "None", "None", eval_logp, draw_q0, eval_z1eps_logqt, eval_z1eps)
        self.draw_p, self.n_gibbs = draw_p, n_gibbs

    def getlosses(self, x):
        x, z = self.generate("gibbs", self.draw_p, n_iter=self.n_gibbs, z0=None, x0=x, no_grad=False)
        return -self.eval_logp(x, z).mean(),

    def getloss(self, x): return self.getlosses(x)[0]

