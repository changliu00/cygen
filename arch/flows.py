""" Modified by Chang Liu <changliu@microsoft.com>
    based on <https://github.com/riannevdberg/sylvester-flows/blob/master/models/flows.py>.

Collection of flow strategies
"""
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#from models.layers import MaskedConv2d, MaskedLinear
from .layers import MaskedConv2d, MaskedLinear
import numpy as np


class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):

        super(Planar, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """

        return 1 - self.h(x) ** 2

    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """

        zk = zk.unsqueeze(2)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return z, log_det_jacobian


class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):

        super(Sylvester, self).__init__()

        self.num_ortho_vecs = num_ortho_vecs

        self.h = nn.Tanh()

        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True, eval_jac=False):
        """
        All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param q_ortho: shape (batch_size, z_size, num_ortho_vecs)
        :param b: shape: (batch_size, 1, num_ortho_vecs)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(-2) # (batch_size, 1, z_size)

        # Save diagonals for log_det_j
        diag_r1 = r1[..., self.diag_idx, self.diag_idx] # (batch_size, num_ortho_vecs)
        diag_r2 = r2[..., self.diag_idx, self.diag_idx] # (batch_size, num_ortho_vecs)

        r1_hat = r1 # (batch_size, num_ortho_vecs, num_ortho_vecs)
        r2_hat = r2 # (batch_size, num_ortho_vecs, num_ortho_vecs)

        qr2 = q_ortho @ r2_hat.transpose(-2, -1) # (batch_size, z_size, num_ortho_vecs)
        qr1 = q_ortho @ r1_hat # (batch_size, z_size, num_ortho_vecs)

        # print(zk.size(), qr2.size(), b.size())
        r2qzb = zk @ qr2 + b # (batch_size, 1, num_ortho_vecs)
        z = self.h(r2qzb) @ qr1.transpose(-2, -1) + zk # (batch_size, 1, z_size)
        z = z.squeeze(-2) # (batch_size, z_size)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        h_deriv = self.der_h(r2qzb) # (batch_size, 1, num_ortho_vecs)
        diag_j = diag_r1 * diag_r2 # (batch_size, num_ortho_vecs)
        diag_j = h_deriv.squeeze(-2) * diag_j # (batch_size, num_ortho_vecs)
        diag_j += 1.
        log_diag_j = diag_j.abs().log() # (batch_size, num_ortho_vecs)

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1) # (batch_size,)
        else:
            log_det_j = log_diag_j # (batch_size,)

        if eval_jac:
            jac_zk_z = torch.eye(zk.shape[-1], device=zk.device) + (qr2 * h_deriv) @ qr1.transpose(-2, -1) # (batch_size, z_size, z_size)
        else:
            jac_zk_z = None

        return z, log_det_j, jac_zk_z

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True, eval_jac=False):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj, eval_jac)

    def draw(self, zk, r1, r2, q_ortho, b):
        # Amortized flow parameters
        zk = zk.unsqueeze(-2) # (batch_size, 1, z_size)

        # Save diagonals for log_det_j
        diag_r1 = r1[..., self.diag_idx, self.diag_idx] # (batch_size, num_ortho_vecs)
        diag_r2 = r2[..., self.diag_idx, self.diag_idx] # (batch_size, num_ortho_vecs)

        r1_hat = r1
        r2_hat = r2

        qr2 = q_ortho @ r2_hat.transpose(-2, -1) # (batch_size, z_size, num_ortho_vecs)
        qr1 = q_ortho @ r1_hat # (batch_size, z_size, num_ortho_vecs)

        r2qzb = zk @ qr2 + b # (batch_size, 1, num_ortho_vecs)
        z = self.h(r2qzb) @ qr1.transpose(-2, -1) + zk # (batch_size, 1, z_size)
        z = z.squeeze(-2) # (batch_size, z_size)
        return z


class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):

        super(TriangularSylvester, self).__init__()

        self.z_size = z_size
        self.h = nn.Tanh()

        diag_idx = torch.arange(0, z_size).long()
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        if permute_z is not None:
            # permute order of z
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk

        r2qzb = torch.bmm(z_per, r2.transpose(2, 1)) + b
        z = torch.bmm(self.h(r2qzb), r1.transpose(2, 1))

        if permute_z is not None:
            # permute order of z again back again
            z = z[:, :, permute_z]

        z += zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class IAF(nn.Module):
    """
    PyTorch implementation of inverse autoregressive flows as presented in
    "Improving Variational Inference with Inverse Autoregressive Flow" by Diederik P. Kingma, Tim Salimans,
    Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling.
    Inverse Autoregressive Flow with either MADE MLPs or Pixel CNNs. Contains several flows. Each transformation
     takes as an input the previous stochastic z, and a context h. The structure of each flow is then as follows:
     z <- autoregressive_layer(z) + h, allow for diagonal connections
     z <- autoregressive_layer(z), allow for diagonal connections
     :
     z <- autoregressive_layer(z), do not allow for diagonal connections.

     Note that the size of h needs to be the same as h_size, which is the width of the MADE layers.
     """

    def __init__(self, z_size, num_flows=2, num_hidden=0, h_size=50, forget_bias=1., conv2d=False):
        super(IAF, self).__init__()
        self.z_size = z_size
        self.num_flows = num_flows
        self.num_hidden = num_hidden
        self.h_size = h_size
        self.conv2d = conv2d
        if not conv2d:
            ar_layer = MaskedLinear
        else:
            ar_layer = MaskedConv2d
        self.activation = torch.nn.ELU
        # self.activation = torch.nn.ReLU

        self.forget_bias = forget_bias
        self.flows = []
        self.param_list = []

        # For reordering z after each flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        for k in range(num_flows):
            arch_z = [ar_layer(z_size, h_size), self.activation()]
            self.param_list += list(arch_z[0].parameters())
            z_feats = torch.nn.Sequential(*arch_z)
            arch_zh = []
            for j in range(num_hidden):
                arch_zh += [ar_layer(h_size, h_size), self.activation()]
                self.param_list += list(arch_zh[-2].parameters())
            zh_feats = torch.nn.Sequential(*arch_zh)
            linear_mean = ar_layer(h_size, z_size, diagonal_zeros=True)
            linear_std = ar_layer(h_size, z_size, diagonal_zeros=True)
            self.param_list += list(linear_mean.parameters())
            self.param_list += list(linear_std.parameters())

            if torch.cuda.is_available():
                z_feats = z_feats.cuda()
                zh_feats = zh_feats.cuda()
                linear_mean = linear_mean.cuda()
                linear_std = linear_std.cuda()
            self.flows.append((z_feats, zh_feats, linear_mean, linear_std))

        self.param_list = torch.nn.ParameterList(self.param_list)

    def build_mask(self, in_features, out_features, diagonal_zeros=False):
        n_in, n_out = in_features, out_features
        assert n_in % n_out == 0 or n_out % n_in == 0

        mask = np.ones((n_in, n_out), dtype=np.float32)
        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i + 1:, i * k:(i + 1) * k] = 0
                if diagonal_zeros:
                    mask[i:i + 1, i * k:(i + 1) * k] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[(i + 1) * k:, i:i + 1] = 0
                if diagonal_zeros:
                    mask[i * k:(i + 1) * k:, i:i + 1] = 0
        return mask

    def forward(self, z, h_context, eval_jac=False):

        logdets = 0.
        jaceps_z = torch.ones_like(z).to(z.device)
        jaceps_z = jaceps_z.diag_embed()
        for i, flow in enumerate(self.flows):
            if (i + 1) % 2 == 0 and not self.conv2d:
                # reverse ordering to help mixing
                z = z[:, self.flip_idx]
                jaceps_z = jaceps_z[:, :, self.flip_idx]

            # print(flow[0])
            h = flow[0](z)
            # for w in flow[0].state_dict(): print(w)
            grad_h = (h > 0) * 1. + (h < 0) * (h + 1) # ELU gradient
            jac_h1_pre = torch.tensor(self.build_mask(z.size()[-1], z.size()[-1]), device=h.device) * torch.tensor(flow[0][0].weight)
            jac_h1 = jaceps_z @ jac_h1_pre @ grad_h.diag_embed()

            h = h + h_context
            # for w in flow[1].state_dict(): print(w)
            h = flow[1](h)
            grad_h = (h > 0) * 1. + (h < 0) * (h + 1)
            jac_h2_pre = torch.tensor(self.build_mask(z.size()[-1], z.size()[-1]), device=h.device) * torch.tensor(flow[1][0].weight)
            jac_h2 = jac_h1 @ jac_h2_pre @ grad_h.diag_embed()
            mean, jac_h3 = flow[2](h, eval_jac=True)
            jac_h3 = jac_h2 @ jac_h3
            gate = F.sigmoid(flow[3](h) + self.forget_bias)
            l_gate = 1 - gate
            gate_grad = gate * (1 - gate)
            gate_grad = gate_grad.diag_embed()
            gate_jac = jac_h3 @ gate_grad
            z = gate * z + (1 - gate) * mean
            jac_zkp1 = gate_jac * z.diag_embed() + gate.diag_embed() + jac_h3 * l_gate.diag_embed() - gate_jac @ mean.diag_embed()
            logdets += torch.sum(gate.log().view(gate.size(0), -1), 1)

            if eval_jac: jaceps_z = jaceps_z @ jac_zkp1
            # print(jaceps_z.size())
        return z, logdets, jaceps_z

