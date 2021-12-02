# For `ELBO_FlowqNoInv` and `CyGen_FlowqNoInv`:
# eval_logp(x, z) -> logp
# draw_q0(x, n_mc) -> eps
# eval_z1eps_logqt(x, eps) -> z, logqt, (jaceps_z)
# eval_z1eps(x, eps) -> z [[Only `CyGen_FlowqNoInv`]]
#
# generate:
# draw_p(z, n_mc) -> x

