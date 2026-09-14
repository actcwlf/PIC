from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np
import torch

import torch.utils.data
from loguru import logger

import random

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio
from torch.optim.lr_scheduler import CosineAnnealingLR
from compressai.optimizers import net_aux_optimizer

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


from configurations import ModelParams, PipelineParams, OptimizationParams, FinetunePipelineParams
from models.dcc import DCC, PICWrapper
from utils.common import wrap_str
from utils.data import ImageFolderDataset, LSDIRDataset
from utils.dataloader import build_dataloader
from utils.evaluation import eval_model, eval_finetune_model
from utils.logging import log_training_results
from utils.loss import rd_loss
from utils.scheduler import build_linear_scheduler, build_expon_scheduler


@dataclass
class TrainMeta:
    trained_iterations: int = 0
    total_iterations: int = 0


def configure_optimizers(net, lr, aux_lr):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": lr},
        "aux": {"type": "Adam", "lr": aux_lr},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]



def prepare_target_img(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.ToTensor()
    img = transform(img)

    if img.shape[-2] < img.shape[-1]:
        img = img.permute(0, 2, 1)
    return img


def finetune_model(
        mp: ModelParams,
        pp: FinetunePipelineParams,
        op: OptimizationParams,
        tb_writer: SummaryWriter
):
    # if op.tf32:
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True
    # torch.use_deterministic_algorithms(True)
    logger.info(wrap_str('tf32', torch.backends.cuda.matmul.allow_tf32))
    # torch.backends.cudnn.allow_tf32 = True
    # trained_iterations = 0
    base_model = DCC(n_latent=mp.n_latent, n_channel=mp.n_channel, n_nif=mp.n_nif, q_scale=mp.q_scale).cuda()






    logger.info(wrap_str(
        '[Total epochs]', op.finetune_interation,
    ))



    if pp.checkpoint:
        (ckpt, train_meta_ckpt) = torch.load(pp.checkpoint)
        base_model.restore(ckpt)
        dcc_train_meta = TrainMeta(**train_meta_ckpt)
        logger.info(f'load checkpoint: {pp.checkpoint}')
        logger.info(f'checkpoint meta: {dcc_train_meta}')

    finetune_meta = TrainMeta(total_iterations=op.finetune_interation)

    logger.info(wrap_str(
        '[Start iterations]', finetune_meta.trained_iterations
    ))

    # logger.info(wrap_str(
    #     '[MoE Temperature] expon 10 -> 1, max', total_iterations
    # ))
    x = prepare_target_img(pp.image_path).cuda()
    # x = x[..., :-1]
    # print(x.shape)
    # with torch
    base_model.eval()
    model = PICWrapper(base_model, x.unsqueeze(0)).cuda()
    model.train()
    # dataset = ImageFolderDataset(pp.train_data_dir)


    # optimizer = optim.Adam(model.parameters(), lr=op.lr)

    optimizer, aux_optimizer = configure_optimizers(model, op.lr, aux_lr=op.aux_lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=1e-5)
    bitstream = eval_finetune_model(x, model, pp, step=0, total_iter=op.finetune_interation, tb_writer=tb_writer)
    # 继续训练时重新设定对应的学习率
    # for i in range(train_meta.trained_iterations):
    #     scheduler.step()
    for iteration in range(op.finetune_interation):
        x = x.cuda()

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        finetune_meta.trained_iterations += 1

        # temperature = temperature_scheduler(train_meta.trained_iterations)
        temperature = 1
        # model.temperature = temperature
        out = model()
        # z_loss = model.collect_z_loss()
        # z_loss = z_loss.pow(2).mean()

        # lr = scheduler.get_last_lr()[0]

        # weight_sim = model.collect_weight_sim()
        # weight_sim = weight_sim.abs().mean()

        total_loss, distortion_loss, train_psnr, train_bpp, sparsity_reg = rd_loss(out, x, lmbda=op.lmbda)

        total_loss = total_loss
        total_loss.backward()  # noqa
        optimizer.step()
        # scheduler.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        log_training_results(
            tb_writer, finetune_meta.trained_iterations,
            train_psnr, train_bpp, distortion_loss.mean(), total_loss, 0, 0, temperature, lr=op.lr, aux_loss=aux_loss
        )

        if finetune_meta.trained_iterations % pp.eval_freq == 0:
            eval_finetune_model(x, model, pp, step=iteration, total_iter=op.finetune_interation, tb_writer=tb_writer)
            # # eval_model(model, pp, epoch=epoch, step=i, total_iter=finetune_meta.trained_iterations, tb_writer=tb_writer)
            # logger.info(wrap_str('Eval: iter', iteration, 'PSNR', train_psnr, 'BPP', train_bpp))

    if op.finetune_interation != 0:
        bitstream = eval_finetune_model(x, model, pp, step=iteration, total_iter=op.finetune_interation, tb_writer=tb_writer)

    if pp.output_path is not None:
        with open(pp.output_path, 'wb') as f:
            f.write(bitstream)
