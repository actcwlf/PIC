from functools import lru_cache

import numpy as np
import torch

import torch.utils.data
from loguru import logger

import random

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio


from configurations import ModelParams, PipelineParams, OptimizationParams
from models.dcc import DCC
from utils.common import wrap_str

from utils.evaluation import eval_model



def eval_model_pipeline(
        mp: ModelParams,
        pp: PipelineParams,
        # tb_writer: SummaryWriter
):

    model = DCC(n_latent=mp.n_latent, n_channel=mp.n_channel, n_nif=mp.n_nif).cuda()


    assert pp.checkpoint
    (ckpt, trained_iterations) = torch.load(pp.checkpoint, weights_only=True)
    model.restore(ckpt)
    logger.info(f'load checkpoint: {pp.checkpoint}')


    logger.info(wrap_str(
        '[Start iterations]', trained_iterations
    ))
    eval_model(model, pp, epoch=-1, step=-1, total_iter=trained_iterations, crop=None, vis=False)