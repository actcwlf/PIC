import time
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn


import torch.utils.data
from loguru import logger

import random

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

from configurations import ModelParams, PipelineParams, OptimizationParams, FinetunePipelineParams
from models.dcc import DCC, PICWrapper
from utils.common import wrap_str
from utils.data import ImageFolderDataset, LSDIRDataset
from utils.dataloader import build_dataloader
from utils.inspector import check_tensor
from utils.logging import log_valid_results
from utils.loss import rd_loss
from utils.metrics import msssim_fn, lpips_fn
from utils.prune import importance_prune


@lru_cache(1)
def cached_valid_loader(data_dir, crop=None):
    val_dataset = ImageFolderDataset(data_dir, crop=crop)
    valid_loader = build_dataloader(val_dataset, batch_size=1, shuffle=False, train=False)
    return valid_loader


@torch.no_grad()
def eval_model(model: DCC, pp: PipelineParams, epoch, step, total_iter, crop=None, vis=False):
    valid_loader = cached_valid_loader(pp.valid_data_dir, crop=crop)

    model.codec.reset_codec()
    model.eval()



    psnr_list = []
    psnr_list_ref = []
    bpp_list = []
    bpp2_list = []
    net_latent_list = []
    bit_len_list = []
    shared_net_bits = 32 * 16 * 17 * 3
    ms_ssim_list = []
    lpips_list = []


    mnif_weight_list = []

    for j, y in enumerate(valid_loader):

        # if j != 8:
        #     continue
        y = y.cuda().permute(0, 1, 2, 3)
        # print(y.shape)
        # _, c,  h, w = y.shape
        # w = w // 8 * 8
        # h = h // 8 * 8
        #
        # y = y[..., :h, :w ]

        # if j != 7:
        #     continue
        # model.synthesis.layer0.prune_mask = None
        # model.synthesis.layer1.prune_mask = None
        # for j in range(y.shape[0]):
        #     ins = y[j:j + 1, ...]
        #     model.train()
        #     for module in model.modules():
        #         if isinstance(module, nn.BatchNorm2d):
        #             module.track_running_stats = False  # 临时禁用统计追踪
        #
        #     ins_out = model(ins)
        #
        #     total_loss, distortion_loss, train_psnr, train_bpp, sparsity_reg = rd_loss(ins_out, y, lmbda=0.001)
        #     total_loss.backward()
        #     importance_prune(model.synthesis.layer0, model.synthesis.layer1, num=1)
        #     model.eval()


        n_pixels = y.shape[-1] * y.shape[-2]
        out = model(y)
        x_hat_ref = out['x_hat']

        start = time.time()
        x_hat, bs, gamma, beta = model.encode(y)
        # print('encode ', j, end=' ')

        # out = model(y)
        # x_hat_ref = out['x_hat']
        x_hat = model.codec.decode(bs)
        # print('decode ', j)
        bpp_list.append(len(bs) * 8 / n_pixels)
        bpp2_list.append((len(bs) * 8 - shared_net_bits) / n_pixels)
        x_hat = x_hat.permute(2, 0, 1).unsqueeze(0)
        if vis:
            net_latent, syn_mnif_weight = model.get_net_latent(y)
            net_latent_list.append(net_latent)
            mnif_weight_list.append(syn_mnif_weight)

        # print(x_hat_ref.shape, y.shape)
        psnr_val_ref = peak_signal_noise_ratio(
            x_hat_ref.cpu().detach().numpy(),
            y.cpu().detach().numpy(),
            data_range=1.
        )
        psnr_list_ref.append(psnr_val_ref)
        # print(x_hat.shape, y.squeeze(0).permute(1, 2, 0).shape)
        psnr_val = peak_signal_noise_ratio(
            x_hat.cpu().detach().numpy(),
            y.cpu().detach().numpy(),
            data_range=1.
        )
        msssim_val = msssim_fn(x_hat, y)
        lpips_val = lpips_fn(x_hat, y, normalize=True)
        # n_pixels = y.shape[-1] * y.shape[-2]
        # bpp = out['rate_y'] / n_pixels
        # _, psnr_val, bpp = rd_loss(out, y, op.lmbda)
        # print(x_hat.shape, 'bpp', len(bs) * 8 / n_pixels, 'psnr', psnr_val, psnr_val_ref)
        psnr_list.append(psnr_val)
        ms_ssim_list.append(msssim_val)
        lpips_list.append(lpips_val)
        # bpp_list.append(bpp)

        # if j == 0:
        #     weight_sim = model.collect_weight_sim()
        #     weight_sim = weight_sim.abs().mean()
        #     logger.info(wrap_str('Weight avg cosine similarity', weight_sim.item()))
        # break

    if vis:
        net_latent = torch.cat(net_latent_list, dim=0)
        step = net_latent.shape[1] // model.encoder.n_latent
        for l_i in range(model.encoder.n_latent):

            plt.imshow(net_latent[:, l_i * step: (l_i+1) * step].detach().cpu().numpy())
            plt.title(f'latent {l_i}')
            plt.colorbar(location='bottom')
            plt.show()

        keys = mnif_weight_list[0].keys()

        for k in keys:
            result_list = [pack[k] for pack in mnif_weight_list]

            net_mnif_weight = torch.cat(result_list, dim=0)
            plt.imshow(net_mnif_weight.detach().cpu().numpy())
            plt.colorbar(location='bottom')
            plt.title(k)
            plt.show()

        # net_mnif_weight = torch.cat([w[1] for w in mnif_weight_list], dim=0)
        # plt.imshow(net_mnif_weight.detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()
        #
        # net_mnif_weight = torch.cat([w[2] for w in mnif_weight_list], dim=0)
        # plt.imshow(net_mnif_weight.detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()
        #
        # net_mnif_weight = torch.cat([w[3] for w in mnif_weight_list], dim=0)
        # plt.imshow(net_mnif_weight.detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()

    psnr_val = torch.tensor(psnr_list).mean().item()
    psnr_val_ref = torch.tensor(psnr_list_ref).mean().item()
    bpp = torch.tensor(bpp_list).mean().item()
    bpp2 = torch.tensor(bpp2_list).mean().item()
    msssim_val =  torch.tensor(ms_ssim_list).mean().item()
    lpips_val = torch.tensor(lpips_list).mean().item()
    logger.info(wrap_str('Eval: epoch', epoch, 'step', step, 'iter', total_iter,
                         'PSNR', psnr_val, 'PSNRref', psnr_val_ref,
                         'MSSSIM', msssim_val,'LPIPS', lpips_val,
                         'BPP', bpp, '/', bpp2))


    log_valid_results(total_iter, psnr_val, psnr_val_ref, bpp)
        # for name, param in model.named_parameters():
        #     tb_writer.add_histogram(tag=name + '_grad', values=param.grad, global_step=total_iter)
        #     tb_writer.add_histogram(tag=name + '_data', values=param.data, global_step=total_iter)
    model.train()


def eval_finetune_model(y, model: PICWrapper, pp: FinetunePipelineParams,  step, total_iter, tb_writer, crop=None, vis=False):

    model.eval()


    n_pixels = y.shape[-1] * y.shape[-2]


    bpp, bitstream = model.encode()

    x_hat = model.decode(bitstream)

    bpp = len(bitstream) * 8 / n_pixels

    # print(bpp)

    # if vis:
    #         net_latent, syn_mnif_weight = model.get_net_latent(y)
    #         net_latent_list.append(net_latent)
    #         mnif_weight_list.append(syn_mnif_weight)


    psnr_val = peak_signal_noise_ratio(
            x_hat.permute(2, 0, 1).cpu().detach().numpy(),
            y.cpu().detach().numpy(),
            data_range=1.
    )

    logger.info(wrap_str('Eval: step', step, 'total_iter', total_iter,
                         'PSNR', psnr_val, 'BPP', bpp))

    if tb_writer:
        log_valid_results(tb_writer, total_iter, psnr_val, bpp)
        # for name, param in model.named_parameters():
        #     tb_writer.add_histogram(tag=name + '_grad', values=param.grad, global_step=total_iter)
        #     tb_writer.add_histogram(tag=name + '_data', values=param.data, global_step=total_iter)
    model.train()

    return bitstream
