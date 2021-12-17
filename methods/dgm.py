#!/usr/bin/env python3.6
import math
from functools import reduce
from operator import __mul__
from contextlib import suppress
import torch as tc
import torch.nn as nn
from arch.layers import GatedConv2d, GatedConvTranspose2d
from utils.distributions import log_normal_diag, log_normal_standard

__author__ = "Chang Liu, Haoyue Tang"
__email__ = "changliu@microsoft.com"

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.arch_type = args.arch_type
        if args.arch_type == "mlp":
            self.x_nn = nn.Sequential(
                    nn.Linear(reduce(__mul__, args.input_size), 512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, 256))
            self.z_nn = nn.Sequential(
                    nn.Linear(args.dim_z, 512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, 256))
            self.xz_nn = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(256, 1))
            # self.x_nn = nn.Sequential(
            #         nn.Linear(reduce(__mul__, args.input_size), 20),
            #         nn.ReLU(),
            #         nn.Linear(20, 100),
            #         nn.ReLU(),
            #         nn.Linear(100, 100),
            #         nn.ReLU())
            # self.z_nn = nn.Sequential(
            #         nn.Linear(args.dim_z, 20),
            #         nn.ReLU(),
            #         nn.Linear(20, 100),
            #         nn.ReLU(),
            #         nn.Linear(100, 100),
            #         nn.ReLU())
            # self.xz_nn = nn.Sequential(
            #         nn.Linear(200, 100),
            #         nn.BatchNorm1d(100),
            #         nn.ReLU(),
            #         nn.Linear(100, 100),
            #         nn.BatchNorm1d(100),
            #         nn.ReLU(),
            #         nn.Linear(100, 10),
            #         nn.ReLU(),
            #         nn.Linear(10, 1))
        elif args.arch_type == "cnn":
            if args.input_size == [3, 32, 32]:
                self.x_nn = nn.Sequential(*[
                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1, 1)),
                    nn.Dropout(p=0.2),
                    nn.LeakyReLU(),

                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
                    nn.Dropout(p=0.2),
                    nn.LeakyReLU(),

                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(1, 1)),
                    nn.Dropout(p=0.2),
                    nn.LeakyReLU(),

                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2)),
                    nn.Dropout(p=0.2),
                    nn.LeakyReLU(),

                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(1, 1)),
                    nn.Dropout(p=0.2),
                    nn.LeakyReLU()])

                self.z_nn = nn.Sequential(*[
                    nn.Linear(args.dim_z, 512),
                    nn.Dropout(p=0.2),
                    nn.LeakyReLU(),

                    nn.Linear(512, 512),
                    nn.Dropout(p=0.2),
                    nn.LeakyReLU()])

                self.xz_nn = nn.Sequential(*[
                    nn.Linear(1024, 1024),
                    nn.Dropout(p=0.2),
                    nn.LeakyReLU(),

                    nn.Linear(1024, 1024),
                    nn.Dropout(p=0.2),
                    nn.LeakyReLU(),

                    nn.Linear(1024, 1)
                    ])
            else:
                if args.input_size == [1, 28, 28] or args.input_size == [3, 28, 28]:
                    self.last_kernel_size = 7
                elif self.input_size == [1, 28, 20]:
                    self.last_kernel_size = (7, 5)
                elif self.input_size == [3, 32, 32]:
                    self.last_kernel_size = 8

                self.x_nn = nn.Sequential(
                    GatedConv2d(args.input_size[0], 32, 5, 1, 2),
                    GatedConv2d(32, 32, 5, 2, 2),
                    GatedConv2d(32, 64, 5, 1, 2),
                    GatedConv2d(64, 64, 5, 2, 2),
                    GatedConv2d(64, 64, 5, 1, 2),
                    GatedConv2d(64, 256, self.last_kernel_size, 1, 0))

                self.z_nn = nn.Sequential(
                    nn.Linear(args.dim_z, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU())

                self.xz_nn = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 100),
                    nn.ReLU(),
                    nn.Linear(100, 1))
        else:
            raise NotImplementedError

    def forward(self, x, z):
        h_x = self.x_nn(x)
        if self.arch_type == "cnn": h_x = h_x.squeeze()
        h_z = self.z_nn(z)
        return self.xz_nn(tc.cat((h_x, h_z), dim=-1))

def set_requires_grad_(mod: nn.Module, req: bool=True):
    for w in mod.parameters():
        w.requires_grad_(req)

def eval_grad_penalty(model, x_real, z_real, x_fake, z_fake, shape_bat):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = tc.rand(shape_bat, device=x_real.device)
    alpha_x = alpha.view(shape_bat + (1,) * (x_real.ndim - len(shape_bat)))
    alpha_z = alpha.view(shape_bat + (1,) * (z_real.ndim - len(shape_bat)))
    # Get random interpolation between real and fake samples
    x_interp = (alpha_x * x_real + (1 - alpha_x) * x_fake).requires_grad_(True)
    z_interp = (alpha_z * z_real + (1 - alpha_z) * z_fake).requires_grad_(True)
    out_interp = model(x_interp, z_interp)
    # Get gradient w.r.t. interpolates
    grad_x, grad_z = tc.autograd.grad(
            outputs=out_interp,
            inputs=[x_interp, z_interp],
            grad_outputs=tc.ones_like(out_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
    )
    grad_sqnorms_x = (grad_x.view(shape_bat + (-1,)) ** 2).sum(dim=-1)
    grad_sqnorms_z = (grad_z.view(shape_bat + (-1,)) ** 2).sum(dim=-1)
    grad_norms = (grad_sqnorms_x + grad_sqnorms_z).sqrt()
    gp = ((grad_norms - 1) ** 2).mean()
    return gp

def acc_with_logits(model: tc.nn.Module, x: tc.Tensor, y: tc.LongTensor, is_binary: bool, u = None, use_u = False) -> float:
    with tc.no_grad(): logits = model(x) if not use_u else model(x,u)
    ypred = (logits > 0).long() if is_binary else logits.argmax(dim=-1)
    return (ypred == y).float().mean().item()

################ BASE CLASS ################

class GenBase:
    def __init__(self, args, n_discr_upd, eval_logprior, draw_prior, eval_logp, draw_p, *,
            eval_logq = None, draw_q = None,
            draw_q0 = None, eval_z1eps_logqt = None, eval_z1eps = None
        ):
        for k,v in locals().items():
            if k != "self": setattr(self, k, v)
        self.use_discr = bool(n_discr_upd) and n_discr_upd > 0
        if self.use_discr:
            self.pxtype = args.pxtype
            self.discr = [Discriminator(args).to(args.device)] # Make it a list to escape from including the parameters of the discriminator into `self`
            if self.pxtype == "js":
                self.getloss_adv = nn.BCEWithLogitsLoss()
            elif self.pxtype.startswith("ws"):
                self.getloss_adv = lambda out, y: ((1 - 2*y) * out).mean()
            self.opt_discr = self.get_opt_discr()
        if self.draw_q is None:
            if self.eval_z1eps is None:
                self.draw_q = lambda x, n_mc: self.eval_z1eps_logqt(x, self.draw_q0(x, n_mc))[0]
            else:
                self.draw_q = lambda x, n_mc: self.eval_z1eps(x, self.draw_q0(x, n_mc))
        if eval_logq is None:
            self.eval_z_logq = lambda x, n_mc: self.eval_z1eps_logqt(x, self.draw_q0(x, n_mc))[:2]
        else:
            self.eval_z_logq = lambda x, n_mc: (self.draw_q(x, n_mc), self.eval_logq(x, n_mc))

    def get_opt_discr(self):
        if self.pxtype == "js":
            return tc.optim.Adam(self.discr[0].parameters(), lr=self.args.lr_d, betas=(0.5, 0.999), weight_decay=self.args.weight_decay_d)
        elif self.pxtype.startswith("ws"):
            return tc.optim.RMSprop(self.discr[0].parameters(), lr=self.args.lr_d, weight_decay=self.args.weight_decay_d)

    def update_discr(self, x_real, z_real, x_fake, z_fake):
        assert self.use_discr
        x_real, z_real, x_fake, z_fake = [val.detach() for val in [x_real, z_real, x_fake, z_fake]]
        if self.args.reset_opt_discr: self.opt_discr = self.get_opt_discr()
        with tc.enable_grad():
            accs = []; ls_loss_d = []
            for itr in range(self.n_discr_upd):
                out_real = self.discr[0](x_real, z_real)
                out_fake = self.discr[0](x_fake, z_fake)
                y_real = tc.ones_like(out_real)
                y_fake = tc.zeros_like(out_fake)
                loss_d = self.getloss_adv(out_real, y_real) + self.getloss_adv(out_fake, y_fake)
                if loss_d < self.args.gan_lossd_thres: # -5:
                    return
                if self.pxtype == "wsgp":
                    loss_d += self.args.w_gp * eval_grad_penalty(self.discr[0], x_real, z_real, x_fake, z_fake, out_real.shape)

                self.opt_discr.zero_grad()
                loss_d.backward()
                self.opt_discr.step()
                # print(itr, 'Dloss', loss_d.cpu().data)
                # if tc.abs(loss_d) > 100.:
                #     print('irregular track', out_real.mean(), out_fake.mean())

                if self.pxtype == "ws":
                    for w in self.discr[0].parameters():
                        w.data.clamp_(-self.args.ws_clip_val, self.args.ws_clip_val) # w.data.clamp_(-0.1, 0.1)
                if self.args.print_acc_discr:
                    ls_loss_d.append(loss_d)
                    # print(f"iter {itr} lossd: {loss_d}, {self.getloss_adv(out_real, y_real)}, {out_real.mean()}, {self.getloss_adv(out_fake, y_fake)}")
                    accs.append( acc_with_logits(
                            self.discr[0], tc.cat([x_real, x_fake]), tc.cat([y_real, y_fake]),
                            is_binary=True, u=tc.cat([z_real, z_fake]), use_u=True ) )
            if self.args.print_acc_discr:
                print(tc.tensor(accs))
                print(tc.tensor(ls_loss_d))
        return loss_d

    def getloss_adv_gen(self, x_real, z_real, x_fake, z_fake):
        assert self.use_discr
        set_requires_grad_(self.discr[0], False)
        out_real = self.discr[0](x_real, z_real)
        out_fake = self.discr[0](x_fake, z_fake)
        set_requires_grad_(self.discr[0], True)
        y_real = tc.ones_like(out_fake)
        y_fake = tc.zeros_like(out_real)
        loss_g = self.getloss_adv(out_fake, y_real) + self.getloss_adv(out_real, y_fake)
        return loss_g

    def llhmarg(self, x, n_mc): # x: (shape_batch, shape_x)
        z, logq = self.eval_z_logq(x, n_mc) # z: (n_mc, shape_batch, shape_z). logq: (n_mc, shape_batch)
        logpz = self.eval_logprior(z) # (n_mc, shape_batch)
        logp = self.eval_logp(x, z) # (n_mc, shape_batch)
        return (logpz + logp - logq).logsumexp(dim=0) - math.log(n_mc) # (shape_batch): avg only over the mc batch. data batch remains

    def generate(self, n_mc: int, *, no_grad: bool=True, mean_only: bool=False):
        with tc.no_grad() if no_grad else suppress():
            z = self.draw_prior(n_mc)
            x = self.draw_p(z, 1, mean_only=mean_only).squeeze(0)
            return x, z

################ INSTANCES ################

class ELBO(GenBase):
    def __init__(self, n_mc_px, eval_logprior, draw_prior, eval_logp, draw_p, *,
            eval_logq = None, draw_q = None,
            draw_q0 = None, eval_z1eps_logqt = None, eval_z1eps = None
        ):
        GenBase.__init__(self, None, False, eval_logprior, draw_prior, eval_logp, draw_p,
                eval_logq=eval_logq, draw_q=draw_q,
                draw_q0=draw_q0, eval_z1eps_logqt=eval_z1eps_logqt, eval_z1eps=eval_z1eps)
        self.n_mc_px = n_mc_px

    def getlosses(self, x, kl_beta = 1.): # x: (shape_batch, shape_x)
        z, logq = self.eval_z_logq(x, self.n_mc_px) # z: (n_mc_px, shape_batch, shape_z). logq: (n_mc_px, shape_batch)
        logpz = self.eval_logprior(z) # (n_mc_px, shape_batch)
        logp = self.eval_logp(x, z) # (n_mc_px, shape_batch)
        recon = -logp.mean() # avg over both mc batch and data batch
        kl = (logq - logpz).mean() # avg over both mc batch and data batch
        # print(recon.data.item(), kl.data.item())
        return recon + kl_beta * kl, recon, kl

    def getloss(self, x, kl_beta = 1.):
        return self.getlosses(x, kl_beta)[0]

class GibbsNet(GenBase):
    def __init__(self, args, n_gibbs, eval_logprior, draw_prior, eval_logp, draw_p, *,
            eval_logq = None, draw_q = None,
            draw_q0 = None, eval_z1eps_logqt = None, eval_z1eps = None
        ): # `eval_logp`, `eval_z_logq` not used
        GenBase.__init__(self, args, args.n_discr_upd, eval_logprior, draw_prior, eval_logp, draw_p,
                eval_logq=eval_logq, draw_q=draw_q,
                draw_q0=draw_q0, eval_z1eps_logqt=eval_z1eps_logqt, eval_z1eps=eval_z1eps)
        self.n_gibbs = n_gibbs

    def getlosses(self, x, upd_discr: bool=None, *, mean_only: bool=False): # x: (shape_batch, shape_x)
        if upd_discr is None: upd_discr = tc.is_grad_enabled()
        z_real = self.draw_q(x, 1).squeeze(0)
        x_fake, z_fake = self.generate(x.shape[0], n_gibbs=self.n_gibbs, no_grad=False, mean_only=mean_only)
        if upd_discr:
            loss_d = self.update_discr(x, z_real, x_fake, z_fake)
        else: loss_d = None
        return self.getloss_adv_gen(x, z_real, x_fake, z_fake), loss_d

    def getloss(self, x, upd_discr: bool=None, *, mean_only: bool=False):
        return self.getlosses(x, upd_discr, mean_only=mean_only)[0]

    def generate(self, n_mc: int, n_gibbs: int=None, *, no_grad: bool=True, mean_only: bool=False):
        if n_gibbs is None: n_gibbs = self.n_gibbs
        with tc.no_grad() if no_grad else suppress():
            z = self.draw_prior(n_mc)
            x = self.draw_p(z, 1, mean_only=mean_only).squeeze(0)
            for _ in range(n_gibbs):
                z = self.draw_q(x, 1).squeeze(0)
                x = self.draw_p(z, 1, mean_only=mean_only).squeeze(0)
            return x, z

class BiGAN(GibbsNet):
    def __init__(self, args, eval_logprior, draw_prior, eval_logp, draw_p, *,
            eval_logq = None, draw_q = None,
            draw_q0 = None, eval_z1eps_logqt = None, eval_z1eps = None
        ): # `eval_logp`, `eval_z_logq` not used
        GibbsNet.__init__(self, args, 0, eval_logprior, draw_prior, eval_logp, draw_p,
                eval_logq=eval_logq, draw_q=draw_q,
                draw_q0=draw_q0, eval_z1eps_logqt=eval_z1eps_logqt, eval_z1eps=eval_z1eps)

