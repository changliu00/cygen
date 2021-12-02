""" Modified by Haoyue Tang and Chang Liu <changliu@microsoft.com>
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

parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--warm_lr', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=1e-4)
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

parser.add_argument('--vae_clamp', type=float, default=1e9, help="clamping prior samples in generation")

args = parser.parse_args()

args.arch_type = "cnn"
args.input_size = [2]
args.cuda = torch.cuda.is_available() and args.gpu >= 0
args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_')
args.model_signature = args.model_signature.replace(':', '_')
args.save = 'expm/' + args.dataset + '_' + args.method + '_' + args.flow + '_' + str(args.w_cm) + '_' + str(args.w_cm_anneal) + '_optim_' + args.optim + '_' + args.model_signature
# logger
utils.makedirs(args.save)
utils.makedirs(args.save+'/samples')
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

#if args.layer_type == "blend":
#    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
#    args.time_length = 1.0

logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if args.gpu >= 0 and args.cuda else 'cpu')
args.device = device
torch.cuda.set_device('cuda:{}'.format(args.gpu))
print(device)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

def plot(p, x):
    return torchvision.utils.save_image(torch.clamp(x, 0., 1.), p, nrow=int(np.sqrt(args.batch_size)))

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
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    prior_clamp = args.vae_clamp
    args.vae_clamp = 100.

    model = VAE.HouseholderSylvesterVAE(args).to(device)
    frame = get_frame(args.method, model, args)

    logger.info(model)
    #logger.info("Number of trainable parameters: {}".format(count_parameters(model)))
    if args.optim == 'SGD':
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.warm_lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.warm_lr, eps=1.e-7)
    else:
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.warm_lr, weight_decay=args.weight_decay)

    if args.method != "cygen":
        frame_cygen = get_frame("cygen", model, args)

    lrsched = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_shrink)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    compt_meter = utils.RunningAverageMeter(0.93)
    kl_meter = utils.RunningAverageMeter(0.93)
    rec_meter = utils.RunningAverageMeter(0.93)
    #nfef_meter = utils.RunningAverageMeter(0.93)
    #nfeb_meter = utils.RunningAverageMeter(0.93)
    #tt_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    best_loss = float('inf')
    model.train()

    w_cm_prior = args.w_cm

    # warm up
    if args.warm_up > 0:
        frame_warm = get_frame("elbo", model, args)

        for itr in range(1, args.warm_up + 1):
            # beta = itr / args.warm_up * args.kl_beta
            beta = args.kl_beta
            if itr < args.warm_up / 2:
                beta = itr / args.warm_up * 2 * args.kl_beta
            else:
                beta = args.kl_beta
            for i, (x, y) in enumerate(train_loader):

                x = x.to(device)
                loss = compute_loss("elbo", x, frame_warm, beta=beta)
                loss_meter.update(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                time_meter.update(time.time() - end)

            log_message = (
            'Warmup Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(
                itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg,
                # , rec_meter.val, rec_meter.avg, kl_meter.val, kl_meter.avg,
                # compt_meter.val, compt_meter.avg
            ))
            logger.info(log_message)

        # z0 = torch.randn((args.batch_size, args.dim_z)).to(device)
        model_samples = frame_warm.generate(args.batch_size)
        plot('{}/samples/{:>06d}_warmup.png'.format(args.save, 0), model_samples[0])
        get_current_accuracy(model, train_loader, test_loader)

    VAE.vae_clamp = prior_clamp

    if args.optim == 'SGD':
        optimizer = getattr(optim, args.optim)([{'params': model.parameters_enc(), 'lr': args.lr},
            {'params': model.parameters_dec(), 'lr': args.lr * args.pretr_lrratio_dec}],
            weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == 'Adamax':
        optimizer = optim.Adamax([{'params': model.parameters_enc(), 'lr': args.lr},
            {'params': model.parameters_dec(), 'lr': args.lr * args.pretr_lrratio_dec},], eps=1.e-7)
    else:
        optimizer = getattr(optim, args.optim)([{'params': model.parameters_enc(), 'lr': args.lr},
            {'params': model.parameters_dec(), 'lr': args.lr * args.pretr_lrratio_dec},], lr=args.lr, weight_decay=args.weight_decay)

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    ## formal trsaining
    utils.makedirs(args.save)
    torch.save({'args': args,
                'state_dict': model.state_dict(),
                }, os.path.join(args.save, 'warm_up.pth'))
    for itr in range(1, args.niters + 1):
        # loss = float('inf')
        model.train()
        # draw_p = lambda z, n_mc: model.draw_p(z, n_mc, mean_only=args.gen_mean_only)
        # z0 = torch.randn((args.batch_size, args.dim_z)).to(device)
        # z0.requires_grad_(True)

        # model_samples_ori_z = frame.generate("langv-x", draw_p, 10, z0=z0, stepsize=args.x_gen_stepsize, anneal=args.x_gen_anneal, x_range=[0.,1.])
        # model_samples_ori_z = frame.generate("langv-z", draw_p, 10, z0=z0, stepsize=args.z_gen_stepsize, anneal=args.z_gen_anneal)
        # print('plot succeed')
        if args.w_cm_anneal:
            # args.w_cm = (1 - np.exp(-0.1 * float(itr))) / (1 + np.exp(-0.1 * float(itr))) * w_cm_prior
            if itr < 10:
                args.w_cm = w_cm_prior / 100 * itr
            else:
                args.w_cm = w_cm_prior

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            if args.method == "cygen":
                loss, comptloss, pxloss = frame.getlosses(x)
            else:
                loss = frame.getloss(x)
            loss_meter.update(loss.item())
            if args.epoch_verbose: print(loss.data.item())

            #if len(regularization_coeffs) > 0:
            #    reg_states = get_regularization(model, regularization_coeffs)
            #    reg_loss = sum(
            #        reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
            #    )
            #    loss = loss + reg_loss

            #total_time = count_total_time(model)
            #nfe_forward = count_nfe(model)

            optimizer.zero_grad()
            loss.backward()
            # if args.nonlinearity == 'tanh':
            # if args.method == 'bigan':
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
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
        if args.method == "cygen":
            log_message += ' | Compt {:.6f} | pxloss {:.6f}'.format(comptloss, pxloss)
        #if len(regularization_coeffs) > 0:
        #    log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
        logger.info(log_message)
        lrsched.step()

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                model.eval()
                #test_loss = compute_loss(args, model, batch_size=args.test_batch_size)
                x_test, _ = next(iter(test_loader))
                x_test = x_test.to(device)

                if args.method == "cygen":
                    test_loss, comptloss, nllh = frame.getlosses(x)
                    if args.pxtype != "nllhmarg":
                        nllh = -frame.llhmarg(x, args.n_mc_eval).mean()
                elif args.method == "dae":
                    test_loss = frame.getlosses(x)[0]
                    nllh = -frame.llhmarg(x, args.n_mc_eval).mean()
                    comptloss = frame_cygen.getlosses(x)[1]
                elif args.method == "bigan" or args.method == "gibbs":
                    test_loss = frame.getloss(x)
                    nllh = -frame.llhmarg(x, args.n_mc_eval).mean()
                    comptloss = frame_cygen.getlosses(x)[1]
                else:
                    test_loss, nllh, _ = frame.getlosses(x, kl_beta=args.kl_beta)
                    comptloss = frame_cygen.getlosses(x)[1]

                log_message = '[TEST] Iter {:04d} | Test Loss {:.6f} | NLLH {:.6f} | Compt {:.6f}'.format(
                        itr, test_loss, nllh, comptloss)
                logger.info(log_message)
                flag = True

                if test_loss.item() < best_loss or flag:
                    best_loss = test_loss.item()
                    utils.makedirs(args.save)
                    torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                        'x_test': x_test,
                    }, os.path.join(args.save, 'ckpt{:>06d}.pth'.format(itr)))
                model.train()

        if itr % args.downstr_eval_freq == 0 or itr == args.niters:
            get_current_accuracy(model, train_loader, test_loader)

        if itr % args.viz_freq == 0 or itr == args.niters:
            with torch.no_grad():
                model.eval()

                x_test, _ = next(iter(test_loader))
                x_test = x_test.to(device)
                p_samples_tc = torch.rand_like(x_test).to(device)

                plot('{}/samples/{:>06d}_x_fixed.png'.format(args.save, itr), x_test)

                draw_p = lambda z, n_mc: model.draw_p(z, n_mc, mean_only=args.gen_mean_only)
                z0 = torch.randn((args.batch_size, args.dim_z)).to(device)
                if args.method in {"cygen", "dae"}:
                # for gen_iter in range(args.gen_iter // 5):
                #   model_samples = frame.generate("gibbs", draw_p, gen_iter*5, z0=z0)
                #   plot('{}/samples/{:>06d}_{:>06d}_x_gibbs.png'.format(args.save, itr, gen_iter*5), model_samples)
                    draw_p_hat = lambda z, n_mc: model.draw_p(z, n_mc, mean_only=True)
                    # gibbs
                    zp, xp = z0, x_test
                    for i in range(1, 6):
                        model_samples_ori_z = frame.generate("gibbs", draw_p, 10, z0=zp)
                        model_samples_ori_x = frame.generate("gibbs", draw_p, 10, x0=xp)
                        zp, xp = model_samples_ori_z[1], model_samples_ori_x[0]
                        plot('{}/samples/{:>06d}_gibbs_oriz_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_z[1], 1).squeeze(0))
                        plot('{}/samples/{:>06d}_gibbs_orix_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_x[1], 1).squeeze(0))

                    # langevin
                    zp, xp, xp0 = z0, x_test, torch.rand_like(x_test).to(device)
                    for i in range(1, 11):
                        model_samples_ori_z = frame.generate("langv-x", draw_p, 10, z0=zp, stepsize=args.x_gen_stepsize, anneal=args.x_gen_anneal, x_range=[0.,1.])
                        model_samples_ori_x = frame.generate("langv-x", draw_p, 10, x0=xp, stepsize=args.x_gen_stepsize, anneal=args.x_gen_anneal, x_range=[0.,1.])
                        model_samples_ori_x0 = frame.generate("langv-x", draw_p, 10, x0=xp0, stepsize=args.x_gen_stepsize, anneal=args.x_gen_anneal, x_range=[0.,1.])
                        zp, xp, xp0 = model_samples_ori_z[1], model_samples_ori_x[0], model_samples_ori_x0[0]
                        plot('{}/samples/{:>06d}_x_langv_oriz_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_z[1], 1).squeeze(0))
                        plot('{}/samples/{:>06d}_x_langv_orix_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_x[1], 1).squeeze(0))
                        plot('{}/samples/{:>06d}_x_langv_orix0_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_x0[1], 1).squeeze(0))

                    zp, xp, xp0 = z0, x_test, torch.rand_like(x_test).to(device)
                    for i in range(1, 11):
                        model_samples_ori_z = frame.generate("langv-z", draw_p, 10, z0=zp, stepsize=args.z_gen_stepsize, anneal=args.z_gen_anneal)
                        model_samples_ori_x = frame.generate("langv-z", draw_p, 10, x0=xp, stepsize=args.z_gen_stepsize, anneal=args.z_gen_anneal)
                        model_samples_ori_x0 = frame.generate("langv-z", draw_p, 10, x0=xp0, stepsize=args.z_gen_stepsize, anneal=args.z_gen_anneal)
                        zp, xp, xp0 = model_samples_ori_z[1], model_samples_ori_x[0], model_samples_ori_x0[0]
                        plot('{}/samples/{:>06d}_z_langv_oriz_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_z[1], 1).squeeze(0))
                        plot('{}/samples/{:>06d}_z_langv_orix_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_x[1], 1).squeeze(0))
                        plot('{}/samples/{:>06d}_z_langv_orix0_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_x0[1], 1).squeeze(0))

                    # zp, xp, xp0 = z0, x_test, torch.rand_like(x_test).to(device)
                    # for i in range(1, 11):
                    #     model_samples_ori_z = frame.generate("langv-x", draw_p, 10, z0=zp, stepsize=args.x_gen_stepsize, anneal=args.x_gen_anneal, x_range=[0.,1.])
                    #     model_samples_ori_x = frame.generate("langv-x", draw_p, 10, x0=xp, stepsize=args.x_gen_stepsize, anneal=args.x_gen_anneal, x_range=[0.,1.])
                    #     model_samples_ori_x0 = frame.generate("langv-x", draw_p, 10, x0=xp0, stepsize=args.x_gen_stepsize, anneal=args.x_gen_anneal, x_range=[0.,1.])
                    #     zp, xp, xp0 = model_samples_ori_z[1], model_samples_ori_x[0], model_samples_ori_x0[0]
                    #     plot('{}/samples/{:>06d}_x_langv_oriz_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_z[1], 1).squeeze(0))
                    #     plot('{}/samples/{:>06d}_x_langv_orix_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_x[1], 1).squeeze(0))
                    #     plot('{}/samples/{:>06d}_x_langv_orix0_{:>06d}.png'.format(args.save, itr, i*10), draw_p_hat(model_samples_ori_x0[1], 1).squeeze(0))

                    model_samples = frame.generate("xhat", draw_p_hat, 0, x0=x_test)
                    plot('{}/samples/{:>06d}_x_hat.png'.format(args.save, itr), model_samples[0])
                elif args.method == "elbo":
                    draw_p_hat = lambda z, n_mc: model.draw_p(z, n_mc, mean_only=True)
                    z0 = torch.randn((args.batch_size, args.dim_z)).to(device)
                    model_samples = draw_p_hat(z0, 1).squeeze(0)
                    plot('{}/samples/{:>06d}_x_hat.png'.format(args.save, itr), model_samples)
                elif args.method in {"gan", "bigan"}:
                    model_samples = frame.generate(draw_p, args.gen_batch_size, device=device)
                    print(model_samples[0, :, :, :])
                    ax = plt.subplot(1, 4, 3, aspect="equal")
                    plot_samples(model_samples.cpu().numpy(), ax, npts=args.plt_npts)

                # fig_filename = os.path.join(args.save, 'figs', '{:04d}.pdf'.format(itr))
                # print(fig_filename)
                # utils.makedirs(os.path.dirname(fig_filename))
                # plt.savefig(fig_filename)
                # plt.close()
                model.train()

        end = time.time()

    logger.info('Training has finished.')
