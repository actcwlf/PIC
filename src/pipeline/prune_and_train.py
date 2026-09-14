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



from configurations import ModelParams, PipelineParams, OptimizationParams
from models.dcc import DCC
from utils.common import wrap_str
from utils.data import ImageFolderDataset, LSDIRDataset
from utils.dataloader import build_dataloader
from utils.evaluation import eval_model
from utils.logging import log_training_results
from utils.loss import rd_loss
from utils.prune import importance_prune
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



def prune_and_train_model(
        mp: ModelParams,
        pp: PipelineParams,
        op: OptimizationParams,
        tb_writer: SummaryWriter
):
    # trained_iterations = 0
    model = DCC(n_latent=mp.n_latent, n_channel=mp.n_channel, n_nif=mp.n_nif, q_scale=mp.q_scale).cuda()

    dataset = LSDIRDataset(pp.train_data_dir, crop=op.crop)
    train_loader = build_dataloader(dataset, batch_size=op.batch_size, shuffle=True, train=True, num_workers=pp.num_workers)


    batch_per_epoch = len(train_loader)
    total_iterations = batch_per_epoch * op.total_epoch

    logger.info(wrap_str(
        '[Total epochs]', op.total_epoch,
        '[Batch per epoch]', batch_per_epoch,
        '[Total iterations]', total_iterations
    ))


    train_meta = TrainMeta(total_iterations=total_iterations)
    if pp.checkpoint:
        (ckpt, train_meta_ckpt) = torch.load(pp.checkpoint)
        model.restore(ckpt)
        train_meta = TrainMeta(**train_meta_ckpt)
        logger.info(f'load checkpoint: {pp.checkpoint}')
        logger.info(f'checkpoint meta: {train_meta}')

        if total_iterations != train_meta.total_iterations:
            logger.warning(f'total iterations mismatch: current {total_iterations} ckpt {train_meta.total_iterations}, '
                           f'set max_iteration for scheduler to current {total_iterations}')
            train_meta.total_iterations = total_iterations

    start_epoch = train_meta.trained_iterations // batch_per_epoch

    # temperature_scheduler = build_linear_scheduler(10, 1, total_iterations)
    temperature_scheduler = build_expon_scheduler(10, 1, max_steps=total_iterations)

    logger.info(wrap_str(
        '[Start epochs]', start_epoch,
        '[Start iterations]', train_meta.trained_iterations
    ))

    # logger.info(wrap_str(
    #     '[MoE Temperature] expon 10 -> 1, max', total_iterations
    # ))

    model.train()
    # dataset = ImageFolderDataset(pp.train_data_dir)

    # optimizer = optim.Adam(model.parameters(), lr=op.lr)

    optimizer, aux_optimizer = configure_optimizers(model, op.lr, aux_lr=op.aux_lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=1e-5)

    # 继续训练时重新设定对应的学习率
    # for i in range(train_meta.trained_iterations):
    #     scheduler.step()
    for epoch in range(start_epoch, op.total_epoch):
        for i, x in enumerate(train_loader):
            x = x.cuda()

            for j in range(x.shape[0]):
                ins = x[j:j+1, ...]
                ins_out = model(ins, op.l2_penalty)
                total_loss, distortion_loss, train_psnr, train_bpp, sparsity_reg = rd_loss(ins_out, x, lmbda=op.lmbda)
                total_loss.backward()
                importance_prune(model.synthesis.layer0, model.synthesis.layer1, num=1)


            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            train_meta.trained_iterations += 1

            # temperature = temperature_scheduler(train_meta.trained_iterations)
            temperature = 1
            # model.temperature = temperature
            out = model(x, op.l2_penalty)
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
                tb_writer, train_meta.trained_iterations,
                train_psnr, train_bpp, distortion_loss.mean(), total_loss, 0, 0, temperature, lr=op.lr, aux_loss=aux_loss
            )

            if train_meta.trained_iterations % pp.eval_freq == 0:
                eval_model(model, pp, epoch=epoch, step=i, total_iter=train_meta.trained_iterations, tb_writer=tb_writer)

            if train_meta.trained_iterations in pp.checkpoint_iterations:
                torch.save(
                    (model.capture(), asdict(train_meta)),
                    pp.checkpoint_dir / f'ckpt_{train_meta.trained_iterations}.pth'
                )
                logger.info(
                    wrap_str('Save checkpoint', pp.checkpoint_dir / f'ckpt_{train_meta.trained_iterations}.pth'))

            # 对齐训练轮次
            if train_meta.trained_iterations > total_iterations:
                break

        torch.save(
            (model.capture(), asdict(train_meta)),
            pp.checkpoint_dir / f'ckpt_newest.pth'
        )

        if (epoch + 1) % 10 == 0:
            torch.save(
                (model.capture(), asdict(train_meta)),
                pp.checkpoint_dir / f'ckpt_{train_meta.trained_iterations}.pth'
            )
            logger.info(wrap_str('Save checkpoint', pp.checkpoint_dir / f'ckpt_{train_meta.trained_iterations}.pth'))

    torch.save(
        (model.capture(), asdict(train_meta)),
        pp.checkpoint_dir / f'ckpt_{train_meta.trained_iterations}.pth'
    )
