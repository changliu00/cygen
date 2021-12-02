""" Modified by Chang Liu <changliu@microsoft.com> and Haoyue Tang
    based on <https://github.com/riannevdberg/sylvester-flows/blob/master/models/VAE.py>.
"""
from __future__ import print_function

import math
import sys
from functools import reduce
from operator import __mul__
import torch
import torch.nn as nn
from torch.autograd import Variable
#import models.flows as flows
from . import flows
from arch.layers import GatedConv2d, GatedConvTranspose2d
sys.path.append("..")
from arch.mlp import MLP, iter_modules_params, ViewLayer

def tensorify(device=None, *args) -> tuple:
    return tuple(arg.to(device) if isinstance(arg, torch.Tensor) else torch.tensor(arg, device=device) for arg in args)

def eval_logp_normal(x, mean = 0., var = 1., ndim = 1):
    mean = tensorify(x.device, mean)[0]
    var = tensorify(x.device, var)[0].expand(x.shape)
    if ndim == 0:
        x = x.unsqueeze(-1); mean = mean.unsqueeze(-1); var = var.unsqueeze(-1)
        ndim = 1
    reduce_dims = tuple(range(-1, -ndim-1, -1))
    quads = ((x-mean)**2 / var).sum(dim=reduce_dims)
    log_det = var.log().sum(dim=reduce_dims)
    numel = reduce(__mul__, x.shape[-ndim:])
    return -.5 * (quads + log_det + numel * math.log(2*math.pi))

class VAE(nn.Module):
    """
    The base VAE class containing gated convolutional encoder and decoder architecture.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, args):
        super(VAE, self).__init__()

        # extract model settings from args
        self.arch_type = args.arch_type
        self.dim_z = args.dim_z
        self.prior_mean = args.prior_mean if hasattr(args, "prior_mean") else 0.
        self.prior_std = args.prior_std if hasattr(args, "prior_std") else 1.
        self.input_size = args.input_size
        self.set_p_var = args.set_p_var
        self.dim_x = reduce(__mul__, args.input_size)
        if self.arch_type == "cnn":
            self.input_type = args.input_type
            if self.input_size == [1, 28, 28] or self.input_size == [3, 28, 28]:
                self.last_kernel_size = 7
            elif self.input_size == [1, 28, 20]:
                self.last_kernel_size = (7, 5)
            elif self.input_size == [3, 32, 32]:
                self.last_kernel_size = 8
            else:
                raise ValueError('invalid input size!!')

            self.q_nn, self.q_mean, self.q_var = self.create_encoder()
            self.p_nn, self.p_mean, self.p_var = self.create_decoder()
            self.q_nn_output_dim = 256

        elif self.arch_type == "mlp":
            self.q_nn_output_dim = args.dims_x2h[-1] # dim of h
            self.dims_x2h = args.dims_x2h
            self.dims_z2h = args.dims_z2h
            self.actv_mlp = args.actv_mlp

            self.q_nn, self.q_mean, self.q_var = self.create_encoder()
            self.p_nn, self.p_mean, self.p_var = self.create_decoder()
        else: raise ValueError(f"unknown `arch_type` '{self.arch_type}'")
        # self.h_size = 256
        self.h_size = self.dim_z

        # auxiliary
        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        # log-det-jacobian = 0 without flows
        self.log_det_j = Variable(self.FloatTensor(1).zero_())

        if hasattr(args, 'vae_clamp'): self.vae_clamp = args.vae_clamp

    def create_encoder(self):
        """
        Helper function to create the elemental blocks for the encoder. Creates a gated convnet encoder.
        the encoder expects data as input of shape (batch_size, num_channels, width, height).
        """

        if self.arch_type == "cnn":
            if self.input_size == [3, 32, 32]:
                q_nn = nn.Sequential(
                    nn.Conv2d(3, 32, 5, 1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),

                    nn.Conv2d(32, 64, 4, 2),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),

                    nn.Conv2d(64, 128, 4, 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(),

                    nn.Conv2d(128, 256, 4, 2),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(),

                    nn.Conv2d(256, 512, 4, 1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(),

                    nn.Conv2d(512, 256, 1, 1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU()
                    )
                q_mean = nn.Linear(256, self.dim_z)
                q_var = nn.Sequential(
                    nn.Linear(256, self.dim_z),
                    nn.Softplus(),
                )

                return q_nn, q_mean, q_var

            #         nn.Conv2d)
            if self.input_type == 'binary':
                q_nn = nn.Sequential(
                    GatedConv2d(self.input_size[0], 32, 5, 1, 2),
                    GatedConv2d(32, 32, 5, 2, 2),
                    GatedConv2d(32, 64, 5, 1, 2),
                    GatedConv2d(64, 64, 5, 2, 2),
                    GatedConv2d(64, 64, 5, 1, 2),
                    GatedConv2d(64, 256, self.last_kernel_size, 1, 0),

                )
                q_mean = nn.Linear(256, self.dim_z)
                q_var = nn.Sequential(
                    nn.Linear(256, self.dim_z),
                    nn.Softplus(),
                )
                return q_nn, q_mean, q_var

            elif self.input_type == 'multinomial':
                act = None

                q_nn = nn.Sequential(
                    GatedConv2d(self.input_size[0], 32, 5, 1, 2, activation=act),
                    GatedConv2d(32, 32, 5, 2, 2, activation=act),
                    GatedConv2d(32, 64, 5, 1, 2, activation=act),
                    GatedConv2d(64, 64, 5, 2, 2, activation=act),
                    GatedConv2d(64, 64, 5, 1, 2, activation=act),
                    GatedConv2d(64, 256, self.last_kernel_size, 1, 0, activation=act)
                )
                q_mean = nn.Linear(256, self.dim_z)
                q_var = nn.Sequential(
                    nn.Linear(256, self.dim_z),
                    nn.Softplus(),
                    nn.Hardtanh(min_val=0.01, max_val=7.)

                )
                return q_nn, q_mean, q_var
            else: pass

        elif self.arch_type == "mlp":
            q_nn = MLP([self.dim_x] + self.dims_x2h, self.actv_mlp)
            q_mean = nn.Linear(self.dims_x2h[-1], self.dim_z)
            q_var = nn.Sequential(
                nn.Linear(self.dims_x2h[-1], self.dim_z),
                nn.Softplus(),
            )
            return q_nn, q_mean, q_var
        else: pass

    def create_decoder(self):
        """
        Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
        """

        if self.arch_type == "cnn":
            num_classes = 256

            if self.input_size == [3, 32, 32]:
                ngf = 32
                p_nn = nn.Sequential(
                    nn.ConvTranspose2d(self.dim_z, 256, 4, 1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(256, 128, 4, 2),
                    nn.BatchNorm2d(ngf * 4),
                    nn.LeakyReLU(),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(128, 64, 4, 1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(64, 32, 4, 2),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),

                    nn.ConvTranspose2d(32, 32, 5, 1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),

                    nn.Conv2d(32, 32, 1, 1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU())
                p_mean = nn.Sequential(nn.Conv2d(32, self.input_size[0], 1, 1), nn.Sigmoid())
                p_var = self.get_p_var()
                return p_nn, p_mean, p_var

            if self.input_type == 'binary':
                p_nn = nn.Sequential(
                    GatedConvTranspose2d(self.dim_z, 64, self.last_kernel_size, 1, 0),
                    GatedConvTranspose2d(64, 64, 5, 1, 2),
                    GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                    GatedConvTranspose2d(32, 32, 5, 1, 2),
                    GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                    GatedConvTranspose2d(32, 32, 5, 1, 2)
                )

                p_mean = nn.Sequential(
                    nn.Conv2d(32, self.input_size[0], 1, 1, 0),
                    nn.Sigmoid()
                )

                p_var = self.get_p_var()
                return p_nn, p_mean, p_var

            elif self.input_type == 'multinomial':
                act = None
                p_nn = nn.Sequential(
                    GatedConvTranspose2d(self.dim_z, 64, self.last_kernel_size, 1, 0, activation=act),
                    GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act),
                    GatedConvTranspose2d(64, 32, 5, 2, 2, 1, activation=act),
                    GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
                    GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act),
                    GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act)
                )

                p_mean = nn.Sequential(
                    nn.Conv2d(32, 256, 5, 1, 2),
                    nn.Conv2d(256, self.input_size[0] * num_classes, 1, 1, 0),
                    # output shape: batch_size, num_channels * num_classes, pixel_width, pixel_height
                )

                return p_nn, p_mean

            else:
                raise ValueError('invalid input type!!')

        elif self.arch_type == "mlp":
            p_nn = MLP([self.dim_z] + self.dims_z2h, self.actv_mlp)
            p_mean = nn.Linear(self.dims_z2h[-1], self.dim_x)
            p_var = self.get_p_var()
            return p_nn, p_mean, p_var
        else: pass

    # BEGIN: New interfaces
    def get_p_var(self):
        set_p_var = self.set_p_var
        if bool(set_p_var): set_p_var = torch.tensor(set_p_var)
        if not bool(set_p_var) or (set_p_var <= 0).any().item():
            p_var = nn.Sequential(
                nn.Linear(self.dims_z2h[-1], self.dim_x),
                nn.Softplus(),
                ViewLayer(self.input_size),
            )
        else:
            p_var_tensor = set_p_var.expand(self.input_size)
            p_var = lambda x: p_var_tensor.to(x)
        return p_var

    def eval_logprior(self, z):
        return eval_logp_normal(z, self.prior_mean, self.prior_std**2, ndim=1)

    def draw_prior(self, n_mc):
        return self.prior_mean + self.prior_std * torch.randn(
                n_mc, self.dim_z, device=list(self.parameters())[0].device)

    def eval_logp(self, x, z):
        if self.arch_type == "cnn":
            z = z.view(-1, self.dim_z, 1, 1)
        h = self.p_nn(z)
        x_mean = self.p_mean(h)
        x_var = self.p_var(h)
        return eval_logp_normal(x, x_mean, x_var, ndim=len(self.input_size))

    def draw_p(self, z, n_mc = 1, mean_only = False):
        if self.arch_type == "cnn":
            z = z.view(-1, self.dim_z, 1, 1)
        h = self.p_nn(z)
        x_mean = self.p_mean(h)
        x_mean = x_mean.expand((n_mc,) + x_mean.shape)
        if not mean_only:
            x_var = self.p_var(h)
            return x_mean + x_var.sqrt() * torch.randn_like(x_mean)
        else: return x_mean

    def eval_logq(self, z, x):
        h = self.q_nn(x)
        z_mean = self.q_mean(h)
        z_var = self.q_var(h)
        return eval_logp_normal(z, z_mean, z_var, ndim=1)

    def draw_q(self, x, n_mc = 1, mean_only = False):
        h = self.q_nn(x)
        if self.arch_type == "cnn": h = h.view(-1, 512)
        z_mean = self.q_mean(h)
        z_mean = z_mean.expand((n_mc,) + z_mean.shape)
        if not mean_only:
            z_var = self.q_var(h)
            return z_mean + z_var.sqrt() * torch.randn_like(z_mean)
        else: return z_mean

    def parameters_dec(self):
        if self.arch_type == "cnn":
            return iter_modules_params(self.p_nn, self.p_mean)
        elif self.arch_type == "mlp":
            return iter_modules_params(self.p_nn, self.p_mean, self.p_var)
        else: pass

    def parameters_enc(self):
        return iter_modules_params(self.q_nn, self.q_mean, self.q_var)
    # END: New interfaces

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """

        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_().to(mu)
        eps = Variable(eps)
        if hasattr(self, 'vae_clamp'): eps.data.clamp_(-self.vae_clamp, self.vae_clamp)
        z = eps.mul(std).add_(mu)

        return z

    def encode(self, x):
        """
        Encoder expects following data shapes as input: shape = (batch_size, num_channels, width, height)
        """

        h = self.q_nn(x)
        h = h.view(h.size(0), -1)
        mean = self.q_mean(h)
        var = self.q_var(h)

        return mean, var

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """

        if self.arch_type == "cnn":
            z = z.view(-1, self.dim_z, 1, 1)
        else:
            z = z.view(-1, self.dim_z)
        h = self.p_nn(z)
        x_mean = self.p_mean(h)

        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """

        # mean and variance of z
        z_mu, z_var = self.encode(x)
        # sample z
        z = self.reparameterize(z_mu, z_var)
        if self.arch_type == "cnn":
            z = z.view(-1, self.dim_z, 1, 1)
        x_mean = self.decode(z)

        return x_mean, z_mu, z_var, self.log_det_j, z, z


class PlanarVAE(VAE):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, args):
        super(PlanarVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Planar
        self.num_flows = args.num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z)
        self.amor_w = nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z)
        self.amor_b = nn.Linear(self.q_nn_output_dim, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_nn(x)
        h = h.view(-1, self.q_nn_output_dim)
        mean_z = self.q_mean(h)
        var_z = self.q_var(h)

        # return amortized u an w for all flows
        u = self.amor_u(h).view(batch_size, self.num_flows, self.dim_z, 1)
        w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.dim_z)
        b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)

        return mean_z, var_z, u, w, b

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var, u, w, b = self.encode(x)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class OrthogonalSylvesterVAE(VAE):
    """
    Variational auto-encoder with orthogonal flows in the encoder.
    """

    def __init__(self, args):
        super(OrthogonalSylvesterVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Sylvester
        self.num_flows = args.num_flows
        self.num_ortho_vecs = args.num_ortho_vecs

        assert (self.num_ortho_vecs <= self.dim_z) and (self.num_ortho_vecs > 0)

        # Orthogonalization parameters
        if self.num_ortho_vecs == self.dim_z:
            self.cond = 1.e-5
        else:
            self.cond = 1.e-6

        self.steps = 100
        identity = torch.eye(self.num_ortho_vecs, self.num_ortho_vecs)
        # Add batch dimension
        identity = identity.unsqueeze(0)
        # Put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer('_eye', Variable(identity))
        self._eye.requires_grad = False

        # Masks needed for triangular R1 and R2.
        triu_mask = torch.triu(torch.ones(self.num_ortho_vecs, self.num_ortho_vecs), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.num_ortho_vecs).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of R1 * R2 have to satisfy -1 < R1 * R2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_nn_output_dim, self.num_flows * self.num_ortho_vecs * self.num_ortho_vecs)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_nn_output_dim, self.num_flows * self.num_ortho_vecs),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_nn_output_dim, self.num_flows * self.num_ortho_vecs),
            self.diag_activation
        )

        self.amor_q = nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z * self.num_ortho_vecs)
        self.amor_b = nn.Linear(self.q_nn_output_dim, self.num_flows * self.num_ortho_vecs)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.num_ortho_vecs)
            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size * num_flows, dim_z * num_ortho_vecs)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, dim_z, num_ortho_vecs)
        """

        # Reshape to shape (num_flows * batch_size, dim_z * num_ortho_vecs)
        q = q.view(-1, self.dim_z * self.num_ortho_vecs)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        amat = torch.div(q, norm)
        dim0 = amat.size(0)
        amat = amat.resize(dim0, self.dim_z, self.num_ortho_vecs)

        max_norm = 0.

        # Iterative orthogonalization
        for s in range(self.steps):
            tmp = torch.bmm(amat.transpose(2, 1), amat)
            tmp = self._eye - tmp
            tmp = self._eye + 0.5 * tmp
            amat = torch.bmm(amat, tmp)

            # Testing for convergence
            test = torch.bmm(amat.transpose(2, 1), amat) - self._eye
            norms2 = torch.sum(torch.norm(test, p=2, dim=2) ** 2, dim=1)
            norms = torch.sqrt(norms2)
            max_norm = torch.max(norms).data[0]
            if max_norm <= self.cond:
                break

        if max_norm > self.cond:
            print('\nWARNING WARNING WARNING: orthogonalization not complete')
            print('\t Final max norm =', max_norm)

            print()

        # Reshaping: first dimension is batch_size
        amat = amat.view(-1, self.num_flows, self.dim_z, self.num_ortho_vecs)
        amat = amat.transpose(0, 1)

        return amat

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_nn(x)
        h = h.view(-1, self.q_nn_output_dim)
        mean_z = self.q_mean(h)
        var_z = self.q_var(h)

        # Amortized r1, r2, q, b for all flows

        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.resize(batch_size, self.num_ortho_vecs, self.num_ortho_vecs, self.num_flows)
        diag1 = diag1.resize(batch_size, self.num_ortho_vecs, self.num_flows)
        diag2 = diag2.resize(batch_size, self.num_ortho_vecs, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        q = self.amor_q(h)
        b = self.amor_b(h)

        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.num_ortho_vecs, self.num_flows)

        return mean_z, var_z, r1, r2, q, b

    def forward(self, x):
        """
        Forward pass with orthogonal sylvester flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var, r1, r2, q, b = self.encode(x)

        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], q_ortho[k, :, :, :], b[:, :, :, k])

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class HouseholderSylvesterVAE(VAE):
    """
    Variational auto-encoder with householder sylvester flows in the encoder.
    """

    def __init__(self, args):
        super(HouseholderSylvesterVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Sylvester
        self.num_flows = args.num_flows
        self.num_householder = args.num_householder
        assert self.num_householder > 0

        identity = torch.eye(self.dim_z, self.dim_z)
        # Add batch dimension
        identity = identity.unsqueeze(0)
        # Put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer('_eye', Variable(identity))
        self._eye.requires_grad = False

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.dim_z, self.dim_z), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.dim_z).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z * self.dim_z)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z),
            self.diag_activation
        )

        self.amor_q = nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z * self.num_householder)

        self.amor_b = nn.Linear(self.q_nn_output_dim, self.num_flows * (self.dim_z if self.arch_type == "cnn" else self.num_householder))

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.dim_z)

            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size, num_flows * dim_z * num_householder)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, dim_z, dim_z)
        """

        # Reshape to shape (num_flows * batch_size * num_householder, dim_z)
        batch_size = q.shape[:-1]
        q = q.view(-1, self.dim_z)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)   # ||v||_2
        v = torch.div(q, norm)  # v / ||v||_2

        # Calculate Householder Matrices
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L

        amat = self._eye - 2 * vvT  # NOTICE: v is already normalized! so there is no need to calculate vvT/vTv

        ## Reshaping: first dimension is batch_size * num_flows
        #amat = amat.view(-1, self.num_householder, self.dim_z, self.dim_z)

        #tmp = amat[:, 0]
        #for k in range(1, self.num_householder):
        #    tmp = torch.bmm(amat[:, k], tmp)

        #amat = tmp.view(*batch_size, self.num_flows, self.dim_z, self.dim_z)
        ##amat = amat.transpose(0, 1)
        #ndim_batch = len(batch_size)
        #amat = amat.permute([ndim_batch] + list(range(ndim_batch)) + [-2,-1])

        #return amat

        amat = amat.view(self.num_householder, self.num_flows, *batch_size, self.dim_z, self.dim_z)
        return reduce(torch.matmul, amat.unbind(0))

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        # batch_size = x.size(0)
        batch_size = x.shape[:-len(self.input_size)]

        h = self.q_nn(x)
        h = h.view(*batch_size, self.q_nn_output_dim)
        mean_z = self.q_mean(h)
        var_z = self.q_var(h)

        # Amortized r1, r2, q, b for all flows
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.view(*batch_size, self.dim_z, self.dim_z, self.num_flows)
        diag1 = diag1.view(*batch_size, self.dim_z, self.num_flows)
        diag2 = diag2.view(*batch_size, self.dim_z, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(-3, -2) * self.triu_mask

        r1[..., self.diag_idx, self.diag_idx, :] = diag1
        r2[..., self.diag_idx, self.diag_idx, :] = diag2

        q = self.amor_q(h)

        b = self.amor_b(h)

        # Resize flow parameters to divide over K flows
        b = b.view(*batch_size, 1, self.dim_z if self.arch_type == "cnn" else self.num_householder, self.num_flows) # original: `self.dim_z, self.num_flows)`

        return mean_z, var_z, r1, r2, q, b

    def forward(self, x):
        """
        Forward pass with orthogonal flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.
        batch_size = x.size(0)

        z_mu, z_var, r1, r2, q, b = self.encode(x)

        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]

            z_k, log_det_jacobian, _ = flow_k(z[k], r1[..., k], r2[..., k], q_k, b[..., k],
                    sum_ldj=True, eval_jac=False)
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]

    # BEGIN: New interfaces
    def draw_q0(self, x, n_mc = 1):
        # print(x.size())
        batch_size = x.shape[:-len(self.input_size)]
        eps = torch.randn((n_mc,) + batch_size + (self.dim_z,), device=x.device)
        if hasattr(self, 'vae_clamp'): eps.data.clamp_(-self.vae_clamp, self.vae_clamp)
        return eps

    def eval_z1eps_logqt(self, x, eps, eval_jac = False):
        z_mu, z_var, r1, r2, q, b = self.encode(x)
        z_std = z_var.sqrt()
        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)
        # Sample z_0
        z = [z_mu + z_std * eps]
        self.log_det_j = z_std.log().sum(dim=-1)
        jaceps_z = z_std.diag_embed() if eval_jac else None
        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]
            # print(z[k].size(), r1[..., k].size(), q_k.size(), b[..., k].size())
            z_k, log_det_jacobian, jaczk_zkp1 = flow_k(z[k], r1[..., k], r2[..., k], q_k, b[..., k],
                    sum_ldj=True, eval_jac=eval_jac)
            z.append(z_k)
            self.log_det_j = self.log_det_j + log_det_jacobian
            if eval_jac: jaceps_z = jaceps_z @ jaczk_zkp1
        # Evaluate log-density
        logpeps = eval_logp_normal(eps, 0., 1., ndim=1)
        logp = logpeps - self.log_det_j
        return z[-1], logp, jaceps_z

    def eval_z1eps(self, x, eps):
        z_mu, z_var, r1, r2, q, b = self.encode(x)
        z_std = z_var.sqrt()
        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)
        # Sample z_0
        z = [z_mu + z_std * eps]
        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]
            z_k = flow_k.draw(z[k], r1[..., k], r2[..., k], q_k, b[..., k])
            z.append(z_k)
        return z[-1]

    def draw_q(self, x, n_mc = 1):
        return self.eval_z1eps(x, self.draw_q0(x, n_mc))

    def parameters_enc(self):
        # `self.diag_activation` and `self.flow_k`s have no parameters.
        return iter_modules_params(self.q_nn, self.q_mean, self.q_var,
                self.amor_d, self.amor_diag1, self.amor_diag2, self.amor_q, self.amor_b,
            )
    # END: New interfaces

class TriangularSylvesterVAE(VAE):
    """
    Variational auto-encoder with triangular Sylvester flows in the encoder. Alternates between setting
    the orthogonal matrix equal to permutation and identity matrix for each flow.
    """

    def __init__(self, args):
        super(TriangularSylvesterVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.TriangularSylvester
        self.num_flows = args.num_flows

        # permuting indices corresponding to Q=P (permutation matrix) for every other flow
        flip_idx = torch.arange(self.dim_z - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.dim_z, self.dim_z), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.dim_z).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z * self.dim_z)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z),
            self.diag_activation
        )

        self.amor_b = nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.dim_z)

            self.add_module('flow_' + str(k), flow_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_nn(x)
        h = h.view(-1, self.q_nn_output_dim)
        mean_z = self.q_mean(h)
        var_z = self.q_var(h)

        # Amortized r1, r2, b for all flows
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.resize(batch_size, self.dim_z, self.dim_z, self.num_flows)
        diag1 = diag1.resize(batch_size, self.dim_z, self.num_flows)
        diag2 = diag2.resize(batch_size, self.dim_z, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        b = self.amor_b(h)

        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.dim_z, self.num_flows)

        return mean_z, var_z, r1, r2, b

    def forward(self, x):
        """
        Forward pass with orthogonal flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var, r1, r2, b = self.encode(x)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            if k % 2 == 1:
                # Alternate with reorderering z for triangular flow
                permute_z = self.flip_idx
            else:
                permute_z = None

            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], b[:, :, :, k], permute_z, sum_ldj=True)

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1].view(-1, self.dim_z, 1, 1) if self.arch_type == "cnn" else z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class IAFVAE(VAE):
    """
    Variational auto-encoder with inverse autoregressive flows in the encoder.
    """

    def __init__(self, args):
        super(IAFVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        # self.h_size = args.made_h_size

        self.h_context = nn.Linear(self.q_nn_output_dim, self.h_size)

        # Flow parameters
        self.num_flows = args.num_flows
        self.flow = flows.IAF(self.dim_z, num_flows=self.num_flows,
                              num_hidden=1, h_size=self.h_size, conv2d=False)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and context h for flows.
        """

        h = self.q_nn(x)
        h = h.view(-1, self.q_nn_output_dim)
        mean_z = self.q_mean(h)
        var_z = self.q_var(h)
        h_context = self.h_context(h)

        return mean_z, var_z, h_context

    def forward(self, x):
        """
        Forward pass with inverse autoregressive flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        # mean and variance of z
        z_mu, z_var, h_context = self.encode(x)
        # sample z
        z_0 = self.reparameterize(z_mu, z_var)

        # iaf flows
        z_k, self.log_det_j, _ = self.flow(z_0, h_context)

        if self.arch_type == "cnn":
            z_k = z_k.view(-1, self.dim_z, 1, 1)

        # decode
        x_mean = self.decode(z_k)

        return x_mean, z_mu, z_var, self.log_det_j, z_0, z_k

    def parameters_enc(self):
        # `self.diag_activation` and `self.flow_k`s have no parameters.
        return iter_modules_params(self.q_nn, self.q_mean, self.q_var, self.h_context, self.flow)

    def eval_z1eps(self, x, eps):
        z_mu, z_var, h_context = self.encode(x)
        z_std = z_var.sqrt()
        z = [z_mu + z_std * eps.squeeze()]
        # Normalizing flows
        z_k, self.log_det_j, _ = self.flow(z[0], h_context)

        if self.arch_type == "cnn":
            z_k = z_k.view(-1, self.dim_z, 1, 1)
        return z_k

    def eval_z1eps_logqt(self, x, eps, eval_jac = False):
        z_mu, z_var, h_context = self.encode(x)
        z_std = z_var.sqrt()
        z = [z_mu + z_std * eps.squeeze()]
        self.log_det_j = z_std.log().sum(dim=-1)
        jaceps_z = z_std.diag_embed() if eval_jac else None

        z_k, self.log_det_j, jaceps_p = self.flow(z[0], h_context, eval_jac=True)
        # print(jaceps_z.size(), jaceps_p.size())
        jaceps_z = jaceps_z @ jaceps_p

        # Evaluate log-density
        logpeps = eval_logp_normal(eps, 0., 1., ndim=1)
        logp = logpeps - self.log_det_j
        return z_k, logp, jaceps_z

