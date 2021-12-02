""" Modified by Haoyue Tang
    based on <https://github.com/rtqichen/ffjord/blob/master/train_toy.py>.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time
import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch.nn.functional as F

#import utils.toy_data as toy_data
import utils.utils as utils
from utils.load_data import load_dataset
#import lib.layers.odefunc as odefunc

import numpy as np
from arch import VAE #, CNFVAE
from methods import cygen, dgm

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', choices=['mnist', 'cifar10', 'svhn'],
    type=str, default='mnist'
)
parser.add_argument(
    "--layer_type", type=str, default="concat",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='100-100')
parser.add_argument("--num_blocks", type=int, default=2, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
#parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
#parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--method', type=str, default="cygen", choices=["elbo", "cygen", "gan", "bigan", "gibbs", "dae"])
parser.add_argument('--flow', type=str, default="sylvester", choices=['iaf', 'sylvester', 'cnf_rank', 'cnf_bias', 'cnf_hyper', 'cnf_lyper'])
parser.add_argument('--dim_z', type=int, default=100)
parser.add_argument('--dims_x2h', type=int, nargs='+', default=[4,2]) # Not used. Just for initializing VAE.
parser.add_argument('--dims_z2h', type=int, nargs='+', default=[16,32,64,32,16,16])
parser.add_argument('--actv_mlp', type=str, default="ReLU")
parser.add_argument('--num_flows', type=int, default=4)
parser.add_argument('--num_householder', type=int, default=8)
parser.add_argument('--set_p_var', type=float, default=1e-2)
parser.add_argument('--rank', type=int, default=20)

parser.add_argument('--prior_mean', type=float, default=0.)
parser.add_argument('--prior_std', type=float, default=1.)

parser.add_argument('--cmtype', type=str, default="jacnorm_x")
parser.add_argument('--pxtype', type=str, default="nllhmarg")
parser.add_argument('--w_cm', type=float, default=1e-3)
parser.add_argument('--w_px', type=float, default=1.)
parser.add_argument('--n_mc_cm', type=int, default=1)
parser.add_argument('--n_mc_px', type=int, default=1)
parser.add_argument('--n_mc_eval', type=int, default=1)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")
parser.add_argument("--kl_beta", type=float, default=1.)

#parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
#parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
#parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
#parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
#parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--n_fid_samples', type=int, default=30000)
parser.add_argument('--warm_lr', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_d',type=float, default=1e-4, help="for method=bigan,gibbs")
parser.add_argument('--lr_shrink', type=float, default=0.99)
parser.add_argument('--pretr_lrratio_dec', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--weight_decay_d', type=float, default=0., help="for method=bigan,gibbs")
parser.add_argument('--optim', type=str, default="Adamax")

parser.add_argument('--gen_batch_size', type=int, default=200)
parser.add_argument('--gibbs_iter', type=int, default=18)
parser.add_argument('--langv_iter', type=int, default=200)
parser.add_argument('--gen_iter', type=int, default=100)
parser.add_argument('--gen_stepsize', type=float, default=3e-4)
parser.add_argument('--x_gen_stepsize', type=float, default=1e-3)
parser.add_argument('--z_gen_stepsize', type=float, default=3e-4)
parser.add_argument('--x_gen_anneal', type=float, default=10.)
parser.add_argument('--z_gen_anneal', type=float, default=10.)
parser.add_argument('--gen_mean_only', type=utils.boolstr, default=False)
parser.add_argument('--plt_npts', type=int, default=100)
parser.add_argument('--plt_memory', type=int, default=100)
parser.add_argument('--downstr_niters', type=int, default=100)
parser.add_argument('--downstr_lr', type=float, default=1e-3)

parser.add_argument('--epoch_verbose', type=utils.boolstr, default=False)
parser.add_argument('--warm_up', type=int, default=0)

# Track quantities
#parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
#parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
#parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
#parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
#parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
#parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

#parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--save', type=str, default='expm/cygen')
parser.add_argument('--viz_freq', type=int, default=1)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--downstr_eval_freq', type=int, default=1)
#parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--w_cm_anneal', type=utils.boolstr, default=False)

parser.add_argument('--n_discr_upd', type=int, default=1, help="for method=bigan,gibbs")
parser.add_argument('--gan_lossd_thres', default=-10., type=float, help="for method=bigan,gibbs")
parser.add_argument('--reset_opt_discr', type=utils.boolstr, default=True, help="for method=bigan,gibbs")
parser.add_argument('--print_acc_discr', type=utils.boolstr, default=False, help="for method=bigan,gibbs")
parser.add_argument('--ws_clip_val', type=float, default=0.1, help="for method=bigan,gibbs, pxtype=ws")
parser.add_argument('--w_gp', type=float, default=10., help="for method=bigan,gibbs, pxtype=wsgp")
# parser.add_argument('--kl_beta', type=float, default=1., help="for method=elbo")
parser.add_argument('--n_gibbs', type=int, default=10, help="for method=gibbs")
parser.add_argument('--lr_decay', type=bool, default=False)

parser.add_argument('--load_dict', type=str, default=None)
args = parser.parse_args()

args.arch_type = "cnn"
args.input_size = [2]
args.cuda = torch.cuda.is_available() and args.gpu >= 0

#if args.layer_type == "blend":
#    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
#    args.time_length = 1.0

device = torch.device('cuda:' + str(args.gpu) if args.gpu >= 0 and args.cuda else 'cpu')
args.device = device
print(device)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
#def get_transforms(model):
#
#    def sample_fn(z, logpz=None):
#        if logpz is not None:
#            return model(z, logpz, reverse=True)
#        else:
#            return model(z, reverse=True)
#
#    def density_fn(x, logpx=None):
#        if logpx is not None:
#            return model(x, logpx, reverse=False)
#        else:
#            return model(x, reverse=False)
#
#    return sample_fn, density_fn
def plot(p, x):
    return torchvision.utils.save_image(torch.clamp(x, 0., 1.), p, nrow=10)

#def compute_loss(args, model, batch_size=None):
def compute_loss(method, x, frame, nllh = False, frame_compt = None, train=True, beta=1):

    if method == "gan" or args.method == "bigan":
        loss = frame.getloss(x, upd_discr=train) if not nllh else -frame.llhmarg(x).mean()
    elif method == "cygen":
        loss = frame.getlosses(x)
    elif method == "elbo":
        loss = frame.getloss(x, kl_beta=beta)
    else:
        loss = frame.getloss(x) if not nllh else -frame.llhmarg(x).mean()
    if nllh:
        loss = -frame.llhmarg(x).mean()
    if frame_compt is None: return loss
    else: return loss, frame_compt.getloss(x)

def get_frame(method, model, args):
    if method == "cygen":
        frame = cygen.CyGen_FlowqNoInv(args.cmtype, args.pxtype,
                model.eval_logp,
                model.draw_q0,
                lambda x, eps: model.eval_z1eps_logqt(x, eps, eval_jac=True),
                model.eval_z1eps,
                args.w_cm, args.w_px,
                args.n_mc_cm, args.n_mc_px)
    elif method == "dae":
        frame = cygen.DAE_FlowqNoInv(model.eval_logp, model.draw_p, model.draw_q0,
                lambda x, eps: model.eval_z1eps_logqt(x, eps, eval_jac=True),
                model.eval_z1eps, n_gibbs=0)
    else:
        args_keys = ['eval_logprior', 'draw_prior', 'eval_logp', 'draw_p',
                'draw_q0', 'eval_z1eps_logqt', 'eval_z1eps']
        model_args = {key: getattr(model, key) for key in args_keys}
        if method == "elbo":
            # frame = dgm.ELBO(args.n_mc_px, **model_args)
            model_args['draw_q0'] = model.forward
            frame = dgm.ELBO_ffjord(args, **model_args)
        elif method == "bigan":
            frame = dgm.BiGAN(args, **model_args)
        elif method == "gibbs":
            frame = dgm.GibbsNet(args, args.n_gibbs, **model_args)
        else: raise ValueError("unsupported `method` '" + method + "'")
    return frame

def get_current_accuracy(model, train_loader, test_loader):
    model.eval()
    netC = nn.Sequential(nn.Linear(args.dim_z, 100), nn.LeakyReLU(), nn.Linear(100, 10))
    netC = netC.to(device)
    netC.train()
    optimizerC = optim.Adamax(netC.parameters(), lr=args.downstr_lr, eps=1e-7)
    lossfn = utils.get_ce_or_bce_loss(dim_y=10)[1]

    for itr in range(args.downstr_niters):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            with torch.no_grad(): z = model.draw_q(data, 1).squeeze(0)
            output = netC(z)
            loss = lossfn(output, target)
            optimizerC.zero_grad()
            loss.backward()
            optimizerC.step()

    netC.eval()
    test_loss = 0
    correct = 0
    total_sample = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            z = model.draw_q(data, 1).squeeze(0)
            output = netC(z)
            test_loss += lossfn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_sample += data.size()[0]
            if batch_idx > 10: break

    log_message = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss / (batch_idx + 1), correct, total_sample,
        100. * correct / total_sample)
    logger.info(log_message)

    model.train()

if __name__ == '__main__':
    ckpt = torch.load('expm/' + args.load_dict)
    opt = args

    args = ckpt['args']
    args.save = 'expmplot/' + opt.load_dict
    utils.makedirs(args.save)
    utils.makedirs(args.save+'/samplescolumn')

    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    train_loader, val_loader, test_loader, args = load_dataset(ckpt['args'], **kwargs)
    #if args.flow == 'cnf_rank':
    #    model = CNFVAE.AmortizedLowRankCNFVAE(args).to(device)
    #elif args.flow == 'cnf_bias':
    #    model = CNFVAE.AmortizedBiasCNFVAE(args).to(device)
    #elif args.flow == 'cnf_hyper':
    #    model = CNFVAE.HypernetCNFVAE(args).to(device)
    #elif args.flow == 'cnf_lyper':
    #    model = CNFVAE.LypernetCNFVAE(args).to(device)
    #elif args.flow == 'sylvester':
    model = VAE.HouseholderSylvesterVAE(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    print("Model Loaded")

    frame = get_frame(args.method, model, args)

    ## gibbs sampling
    x, y = next(iter(train_loader))
    x = x.to(device)

    loss, comptloss, pxloss = frame.getlosses(x)
    log_message = 'Model Test Loss {:.6f} | NLLH {:.6f} | Compt {:.6f}'.format(loss, pxloss, comptloss)
    logger.info(log_message)

    def create_session():
        import tensorflow.compat.v1 as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.0
        config.gpu_options.visible_device_list = ''
        return tf.Session(config=config)

    to_range_0_1 = lambda x: (x + 1.) / 2.
    ds_fid = np.array(torch.cat([to_range_0_1(next(iter(train_loader))[0]) for _ in range(opt.n_fid_samples // args.batch_size)]).cpu().numpy())
    ds_fid = ds_fid.squeeze()
    draw_p = lambda z, n_mc: model.draw_p(z, n_mc, mean_only=args.gen_mean_only)

    def sample_x():
        if opt.method == 'dae':
            draw_p_hat = lambda z, n_mc: model.draw_p(z, n_mc, mean_only=True)
            zp, xp, xp0 = torch.randn((args.batch_size, args.dim_z)).to(device), x, torch.rand_like(x).to(device)
            model_samples_ori_x = frame.generate("gibbs", draw_p, 100, x0=xp0, stepsize=opt.x_gen_stepsize)
            return draw_p_hat(model_samples_ori_x[1], 1).squeeze(0).detach().cpu()

        elif args.method == 'cygen':
            draw_p_hat = lambda z, n_mc: model.draw_p(z, n_mc, mean_only=True)
            zp, xp, xp0 = torch.randn((args.batch_size, args.dim_z)).to(device), x, torch.rand_like(x).to(device)
            model_samples_ori_x = frame.generate("langv-x", draw_p, 200, x0=xp0, stepsize=opt.x_gen_stepsize)
            return draw_p_hat(model_samples_ori_x[1], 1).squeeze(0).detach().cpu()
        elif args.method == 'elbo':
            draw_p_hat = lambda z, n_mc: model.draw_p(z, n_mc, mean_only=True)
            z0 = torch.randn((args.batch_size, args.dim_z)).to(device)
            model_samples = draw_p_hat(z0, 1).squeeze()
            return model_samples.detach().cpu()


    x_samples = torch.cat([to_range_0_1(sample_x()) for _ in range(int(opt.n_fid_samples // args.batch_size))]).numpy()

    # x_samples = np.array(torch.cat([to_range_0_1(next(iter(train_loader))[0]) for _ in range(opt.n_fid_samples // args.batch_size)]).cpu().numpy())
    print(x_samples.shape)
    x_samples = x_samples.squeeze()
    from utils.fid_v2_tf_cpu import fid_score
    to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))
    print(ds_fid.shape, x_samples.shape)

    x_data_nhwc = to_nhwc(255 * ds_fid)
    x_samples_nhwc = to_nhwc(255 * x_samples)
    fid = fid_score(create_session, x_data_nhwc, x_samples_nhwc, cpu_only=True)

    print(fid)

