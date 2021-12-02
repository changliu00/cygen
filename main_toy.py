""" Modified by Chang Liu <changliu@microsoft.com>
    based on <https://github.com/rtqichen/ffjord/blob/master/train_toy.py>.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time
from itertools import cycle

import torch
import torch.optim as optim

import utils.toy_data as toy_data
import utils.utils as utils
#import lib.toy_data as toy_data
#import lib.utils as utils
#from lib.visualize_flow import visualize_transform
#import lib.layers.odefunc as odefunc

import numpy as np
from arch import VAE #, CNFVAE
from methods import cygen, dgm

#from train_misc import standard_normal_logprob
#from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
#from train_misc import add_spectral_norm, spectral_norm_power_iteration
#from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
#from train_misc import build_model_tabular

#from diagnostics.viz_toy import save_trajectory, trajectory_to_video
# from utils.viz_toy import save_trajectory, trajectory_to_video

#SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
#parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='pinwheel'
)
#parser.add_argument(
#    "--layer_type", type=str, default="concat",
#    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
#)
#parser.add_argument('--dims', type=str, default='20-20-20')
#parser.add_argument("--num_blocks", type=int, default=2, help='Number of stacked CNFs.')
#parser.add_argument('--time_length', type=float, default=0.5)
#parser.add_argument('--train_T', type=eval, default=True)
#parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
#parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

ALL_METHODS = ["cygen", "dae", "elbo", "bigan", "gibbs"]
parser.add_argument('--method', type=str, default="cygen", choices=ALL_METHODS)
parser.add_argument('--pretr_method', type=str, default="None", choices=ALL_METHODS+["None"]) # "elbo"
parser.add_argument('--pretr_niter', type=int, default=0) # 1000
parser.add_argument('--pretr_lrratio_dec', type=float, default=0.1)
parser.add_argument('--dim_z', type=int, default=2)
parser.add_argument('--dims_x2h', type=int, nargs='+', default=[8,8]) # Not used. Just for initializing VAE.
parser.add_argument('--dims_z2h', type=int, nargs='+', default=[16,16])
parser.add_argument('--actv_mlp', type=str, default="ReLU")
parser.add_argument('--num_flows', type=int, default=32)
parser.add_argument('--num_householder', type=int, default=2)
parser.add_argument('--set_p_var', type=float, default=1e-2)
#parser.add_argument('--flow', type=str, default="cnf_bias", choices=['sylvester', 'cnf_rank', 'cnf_bias', 'cnf_hyper', 'cnf_lyper'])
#parser.add_argument('--rank', type=int, default=1)

parser.add_argument('--cmtype', type=str, default="jacnorm_lite_x",
        choices=["jacnorm_z", "jacnorm_x", "jacnorm_lite_x"], help="for method=cygen")
parser.add_argument('--pxtype', type=str, default="nllhmarg",
        help="{nllhmarg} for method=cygen, {js,ws,wsgp} for method=bigan,gibbs")
parser.add_argument('--w_cm', type=float, default=1e-5, help="for method=cygen") # {"jacnorm_z": 1e-3, "jacnorm_x": 1e-4, "jacnorm_lite_x": 1e-5}
parser.add_argument('--w_px', type=float, default=1., help="for method=cygen")
parser.add_argument('--n_mc_cm', type=int, default=1, help="for method=cygen (for Hutchinson estimator)")
parser.add_argument('--n_mc_px', type=int, default=16,
        help="for method=cygen,elbo (for MC under q for nllhmarg `pxtype`")
parser.add_argument('--n_mc_eval', type=int, default=None)
parser.add_argument('--n_discr_upd', type=int, default=128, help="for method=bigan,gibbs")
parser.add_argument('--gan_lossd_thres', type=float, default=-5, help="for method=bigan,gibbs")
parser.add_argument('--ws_clip_val', type=float, default=0.1, help="for method=bigan,gibbs, pxtype=ws")
parser.add_argument('--w_gp', type=float, default=10., help="for method=bigan,gibbs, pxtype=wsgp")
parser.add_argument('--reset_opt_discr', type=utils.boolstr, default=False, help="for method=bigan,gibbs")
parser.add_argument('--print_acc_discr', type=utils.boolstr, default=False, help="for method=bigan,gibbs")
# parser.add_argument('--kl_beta', type=float, default=1., help="for method=elbo")
parser.add_argument('--n_gibbs', type=int, default=10, help="for method=gibbs")

#parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
#parser.add_argument('--atol', type=float, default=1e-5)
#parser.add_argument('--rtol', type=float, default=1e-5)
#parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

#parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
#parser.add_argument('--test_atol', type=float, default=None)
#parser.add_argument('--test_rtol', type=float, default=None)

#parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
#parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
#parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
#parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
#parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--niters', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_d',type=float, default=1e-4, help="for method=bigan,gibbs")
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--weight_decay_d', type=float, default=0., help="for method=bigan,gibbs")
parser.add_argument('--optim', type=str, default="Adam")
parser.add_argument('--clip_grad', type=utils.boolstr, default=False)

# parser.add_argument('--gen_type', type=str, default="gibbs")
parser.add_argument('--gen_batch_size', type=int, default=2000)
parser.add_argument('--gen_mean_only', type=utils.boolstr, default=False)
parser.add_argument('--gen_niter', type=int, default=100, help="for gentype=langv,gibbs")
parser.add_argument('--gen_init_std', type=float, default=0., help="for gentype=langv,gibbs. <=0 means init with data samples x0")
parser.add_argument('--gen_stepsize', type=float, default=3e-4, help="for gentype=langv")
parser.add_argument('--gen_anneal', type=float, default=0., help="for gentype=langv. <=0 means do not anneal")
parser.add_argument('--plt_npts', type=int, default=100)
parser.add_argument('--plt_memory', type=int, default=100)
parser.add_argument('--plt_z_side_range', type=float, nargs=2, default=[-8.,8.])
parser.add_argument('--plt_scat_same_range', type=utils.boolstr, default=True)
parser.add_argument('--plt_scat_pointwise', type=utils.boolstr, default=True)

# Track quantities
#parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
#parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
#parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
#parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
#parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
#parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--save', type=str, default='')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
#parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--fig_ext', type=str, default='jpg')
args = parser.parse_args()

args.arch_type = "mlp"
args.input_size = [2]
args.cuda = torch.cuda.is_available() and args.gpu >= 0

if args.n_mc_eval is None: args.n_mc_eval = args.n_mc_px

# if args.method == 'bigan' or args.method == 'elbo': args.clip_grad = True

is_pretr = args.pretr_method != "None" and args.pretr_niter > 0

# logger
if len(args.save) == 0:
    args.save = "expm/" + "-".join([ args.data, args.method, f"pvar{args.set_p_var:.0e}" ])
else:
    args.save = "expm/" + args.save
if args.method == "cygen":
    subfolder = "-".join([ args.pxtype, args.cmtype,
            (("PT" + args.pretr_method + str(args.pretr_niter)) if is_pretr else ""),
            f"wcm{args.w_cm:.0e}",
            f"ginit{args.gen_init_std:1.0f}"
        ])
elif args.method == "dae":
    subfolder = "res" + (("-PT" + args.pretr_method + str(args.pretr_niter)) if is_pretr else "") + f"-ginit{args.gen_init_std:1.0f}"
elif args.method == "elbo":
    subfolder = "res" + (("-PT" + args.pretr_method + str(args.pretr_niter)) if is_pretr else "") # + f"-beta{args.kl_beta:.1f}"
elif args.method in ["bigan", "gibbs"]:
    subfolder = args.pxtype + (("-PT" + args.pretr_method + str(args.pretr_niter)) if is_pretr else "")

args.save += "/" + subfolder.replace("jacnorm_", "jn").replace("--", "-").replace("_", "")
args.save = utils.unique_filename(args.save)
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info("resdir: " + args.save)

#if args.layer_type == "blend":
#    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
#    args.time_length = 1.0

logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if args.cuda else 'cpu')
args.device = device
logger.info(device)

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


#def compute_loss(args, model, batch_size=None):
#    if batch_size is None: batch_size = args.batch_size
#
#    # load data
#    x = toy_data.inf_train_gen(args.data, batch_size=batch_size)
#    x = torch.from_numpy(x).type(torch.float32).to(device)
#    zero = torch.zeros(x.shape[0], 1).to(x)
#
#    # transform to z
#    z, delta_logp = model(x, zero)
#
#    # compute log q(z)
#    logpz = standard_normal_logprob(z).sum(1, keepdim=True)
#
#    logpx = logpz - delta_logp
#    loss = -torch.mean(logpx)
#    return loss

def draw_data(data, batch_size, device = None, tell_labels = False):
    # If `device is None`, then do not make `x` a torch tensor.
    res = toy_data.inf_train_gen(data, batch_size=batch_size, tell_labels=tell_labels)
    if device is None: return res
    else:
        if tell_labels:
            return torch.from_numpy(res[0]).type(torch.float32).to(device), res[1]
        else: return torch.from_numpy(res).type(torch.float32).to(device)

#LOW = -4; HIGH = 4
PLOT_SIDE_RANGE = [-4,4]

#def plot_samples(samples, ax, npts=100, title=r"$x \sim p(x)$"):
def plot_samples(samples, ax, npts=100, title=r"$x \sim p(x)$", side_range=PLOT_SIDE_RANGE):
    #ax.hist2d(samples[:, 0], samples[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
    ax.hist2d(samples[:, 0], samples[:, 1], bins=npts, range=[side_range, side_range], density=True)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

#def plot_density(eval_logp, ax, npts=100, memory=100, title=r"$q(x)$", device="cpu"):
def plot_density(eval_logp, ax, npts=100, memory=100, title=r"$p(x)$", device="cpu", side_range=PLOT_SIDE_RANGE):
    side = np.linspace(*side_range, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    x = torch.from_numpy(x).type(torch.float32).to(device)

    ls_logpx = []
    inds = torch.arange(0, x.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        with torch.no_grad():
            ls_logpx.append( eval_logp(x[ii]) )
    logpx = torch.cat(ls_logpx, 0)
    px = logpx.exp().cpu().numpy().reshape(npts, npts)

    ax.imshow(px, interpolation='nearest') # Or, use `ax.pcolormesh(xx, yy, px)`
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

def plot_scatter_classwise(x, y, ax, title, side_range=PLOT_SIDE_RANGE, pointwise=True):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 'v', 's', '^', 'D', '<', 'p', '>', 'h', 'd', 'H', '8', '*', 'P', 'X']
    ax.set_xlim(side_range); ax.set_ylim(side_range)
    if pointwise:
        for x_i, y_i in zip(x, y):
            ax.scatter(x_i[0], x_i[1], s=2, c=colors[y_i % len(colors)], marker=markers[y_i % len(markers)])
    else:
        for i, c, m in zip(range(y.max() + 1), cycle(colors), cycle(markers)):
            x_i = x[y==i]
            ax.scatter(x_i[:, 0], x_i[:, 1], s=2, c=c, marker=m) # , marker='.')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

def save_fig(fig_name):
    fig_fullname = os.path.join(args.save, fig_name + "." + args.fig_ext)
    plt.savefig(fig_fullname)
    plt.close()

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
            frame = dgm.ELBO(args.n_mc_px, **model_args)
        elif method == "bigan":
            frame = dgm.BiGAN(args, **model_args)
        elif method == "gibbs":
            frame = dgm.GibbsNet(args, args.n_gibbs, **model_args)
        else: raise ValueError("unsupported `method` '" + method + "'")
    return frame


if __name__ == '__main__':
    method = args.pretr_method if is_pretr else args.method

    #regularization_fns, regularization_coeffs = create_regularization_fns(args)
    #model = build_model_tabular(args, 2, regularization_fns).to(device)
    model = VAE.HouseholderSylvesterVAE(args).to(device)
    #if args.spectral_norm: add_spectral_norm(model)
    #set_cnf_options(args, model)
    frame = get_frame(method, model, args)
    if method != "cygen":
        tmp = args.pxtype
        args.pxtype = "None"
        frame_cygen = get_frame("cygen", model, args)
        args.pxtype = tmp

    logger.info(model)
    #logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    #nfef_meter = utils.RunningAverageMeter(0.93)
    #nfeb_meter = utils.RunningAverageMeter(0.93)
    #tt_meter = utils.RunningAverageMeter(0.93)

    plt.clf()
    ax = plt.subplot(aspect="equal")
    x_samples_np = draw_data(args.data, args.gen_batch_size, device=None, tell_labels=False)
    plot_samples(x_samples_np, ax, npts=args.plt_npts, title=r"$x \sim p^*(x)$")
    save_fig('data_samples')
    if method in ["elbo"]:
        plt.clf()
        ax = plt.subplot(aspect="equal")
        z_samples = model.draw_prior(args.gen_batch_size)
        plot_samples(z_samples.cpu().numpy(), ax, npts=args.plt_npts, title=r"$z \sim p(z)$", side_range=args.plt_z_side_range)
        save_fig('prior_samples')

    end = time.time()
    best_loss = float('inf')
    model.train()
    dont_update = False
    for itr in range(1, args.niters + 1):
        if is_pretr and itr > args.pretr_niter:
            itr -= 1 # re-test the model under the new method at the beginning of training
            method = args.method
            frame = get_frame(method, model, args)
            optimizer = getattr(optim, args.optim)(
                    [{'params': model.parameters_enc(), 'lr': args.lr}] + (
                    [{'params': model.parameters_dec(), 'lr': args.lr * args.pretr_lrratio_dec}] if args.pretr_lrratio_dec else []
                ), weight_decay=args.weight_decay)
            logger.info("Pretraining using '" + args.pretr_method + "' done. Start training with '" + args.method + "'")
            is_pretr = False
            dont_update = True

        optimizer.zero_grad()
        #if args.spectral_norm: spectral_norm_power_iteration(model, 1)

        #loss = compute_loss(args, model)
        x = draw_data(args.data, args.batch_size, device=device, tell_labels=False)
        if method == "cygen":
            loss, comptloss, pxloss = frame.getlosses(x)
        else:
            loss = frame.getloss(x)
        loss_meter.update(loss.item())

        #if len(regularization_coeffs) > 0:
        #    reg_states = get_regularization(model, regularization_coeffs)
        #    reg_loss = sum(
        #        reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
        #    )
        #    loss = loss + reg_loss

        #total_time = count_total_time(model)
        #nfe_forward = count_nfe(model)

        if not dont_update:
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            #nfe_total = count_nfe(model)
            #nfe_backward = nfe_total - nfe_forward
            #nfef_meter.update(nfe_forward)
            #nfeb_meter.update(nfe_backward)

            time_meter.update(time.time() - end)
            #tt_meter.update(total_time)

        log_message = (
            #'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
            #' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
            #    itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg,
            #    nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(
                itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg
            )
        )
        if method == "cygen":
            log_message += ' | Compt {:.6f} | pxloss {:.6f}'.format(comptloss, pxloss)
        #if len(regularization_coeffs) > 0:
        #    log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
        # if itr % args.val_freq == 0:
        logger.info(log_message)

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                model.eval()
                #test_loss = compute_loss(args, model, batch_size=args.test_batch_size)
                x = draw_data(args.data, args.test_batch_size, device=device, tell_labels=False)
                if method == "cygen":
                    test_loss, comptloss, nllh = frame.getlosses(x)
                    if args.pxtype != "nllhmarg":
                        nllh = -frame.llhmarg(x, args.n_mc_eval).mean()
                else:
                    test_loss = frame.getloss(x)
                    nllh = -frame.llhmarg(x, args.n_mc_eval).mean()
                    comptloss = frame_cygen.getlosses(x)[1]
                #test_nfe = count_nfe(model)
                #log_message = '[TEST] Iter {:04d} | Test Loss {:.6f} | NFE {:.0f}'.format(itr, test_loss, test_nfe)
                log_message = '[TEST] Iter {:04d} | Test Loss {:.6f} | NLLH {:.6f} | Compt {:.6f}'.format(
                        itr, test_loss, nllh, comptloss)
                logger.info(log_message)

                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    utils.makedirs(args.save)
                    torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                    }, os.path.join(args.save, 'ckpt.pth'))
                    #}, os.path.join(args.save, 'checkpt.pth'))
                model.train()

        if itr % args.viz_freq == 0:
            with torch.no_grad():
                model.eval()
                #sample_fn, density_fn = get_transforms(model)

                #visualize_transform(
                #    x_samples, torch.randn, standard_normal_logprob, transform=sample_fn, inverse_transform=density_fn,
                #    samples=True, npts=800, device=device
                #)

                x, y_np = draw_data(args.data, args.gen_batch_size, device=device, tell_labels=True)
                z_np = frame.draw_q(x, 1).squeeze(0).cpu().numpy()
                scat_side_range = args.plt_z_side_range if args.plt_scat_same_range else PLOT_SIDE_RANGE
                if method in ["cygen", "dae"]:
                    plt.figure(figsize=(9, 6))
                    plt.clf()
                    ax = plt.subplot(2, 3, 1, aspect="equal")
                    plot_density((lambda x: frame.llhmarg(x, args.n_mc_eval)), ax, args.plt_npts, args.plt_memory, device=device, title=r"$p(x)$")

                    if bool(args.gen_init_std) and args.gen_init_std > 0:
                        x0, z0 = None, args.gen_init_std * torch.randn(args.gen_batch_size, args.dim_z, device=device)
                    else: x0, z0 = x, None
                    gen_draw_p = lambda z, n_mc: model.draw_p(z, n_mc, mean_only=args.gen_mean_only)
                    x_samples_gibbs, z_samples_gibbs = frame.generate("gibbs", gen_draw_p, args.gen_niter, z0=z0, x0=x0)
                    x_samples_langvz, z_samples_langvz = frame.generate("langv-z", gen_draw_p, args.gen_niter, z0=z0, x0=x0, stepsize=args.gen_stepsize, anneal=args.gen_anneal)
                    ax = plt.subplot(2, 3, 2, aspect="equal")
                    plot_samples(x_samples_gibbs.cpu().numpy(), ax, npts=args.plt_npts, title=r"$x \sim p(x)$, gibbs")
                    ax = plt.subplot(2, 3, 3, aspect="equal")
                    plot_samples(x_samples_langvz.cpu().numpy(), ax, npts=args.plt_npts, title=r"$x \sim p(x)$, langv-z")

                    ax = plt.subplot(2, 3, 4, aspect="equal")
                    plot_scatter_classwise(z_np, y_np, ax, title="aggr. post.", side_range=scat_side_range, pointwise=args.plt_scat_pointwise)
                    ax = plt.subplot(2, 3, 5, aspect="equal")
                    plot_samples(z_samples_gibbs.cpu().numpy(), ax, npts=args.plt_npts, title=r"$z \sim p(z)$, gibbs", side_range=args.plt_z_side_range)
                    ax = plt.subplot(2, 3, 6, aspect="equal")
                    plot_samples(z_samples_langvz.cpu().numpy(), ax, npts=args.plt_npts, title=r"$z \sim p(z)$, langv-z", side_range=args.plt_z_side_range)

                else:
                    plt.figure(figsize=(9, 3))
                    plt.clf()
                    ax = plt.subplot(1, 3, 1, aspect="equal")
                    plot_density((lambda x: frame.llhmarg(x, args.n_mc_eval)), ax, args.plt_npts, args.plt_memory, device=device, title=r"$p(x)$")

                    x_samples = frame.generate(args.gen_batch_size, mean_only=args.gen_mean_only)[0]
                    ax = plt.subplot(1, 3, 2, aspect="equal")
                    plot_samples(x_samples.cpu().numpy(), ax, npts=args.plt_npts, title=r"$x \sim p(x)$")

                    ax = plt.subplot(1, 3, 3, aspect="equal")
                    plot_scatter_classwise(z_np, y_np, ax, title="aggr. post.", side_range=scat_side_range, pointwise=args.plt_scat_pointwise)

                save_fig(f"{itr:04d}" + ("_new" if dont_update else ""))
                #fig_filename = os.path.join(args.save, 'figs', '{:04d}.jpg'.format(itr))
                #utils.makedirs(os.path.dirname(fig_filename))
                #plt.savefig(fig_filename)
                #plt.close()
                model.train()

        if dont_update: dont_update = False

        end = time.time()

    logger.info('Training has finished.')

    #save_traj_dir = os.path.join(args.save, 'trajectory')
    #logger.info('Plotting trajectory to {}'.format(save_traj_dir))
    #data_samples = toy_data.inf_train_gen(args.data, batch_size=args.gen_batch_size)
    #save_trajectory(model, data_samples, save_traj_dir, device=device)
    #trajectory_to_video(save_traj_dir)
