import numpy as np
import torch

import torch.utils.data
from loguru import logger

import random

def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return



def build_dataloader(dataset, batch_size, shuffle=False, distributed=False, train=True, num_workers=0):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    logger.info(f'Shuffled data {shuffle}')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(sampler is None) and train,
        num_workers=num_workers, pin_memory=True, sampler=sampler,
        drop_last=True, worker_init_fn=worker_init_fn
    )

    return dataloader