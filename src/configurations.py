from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class PipelineParams:

    """ pipeline config """

    train_data_dir: str          # 训练数据集路径
    valid_data_dir: str

    output_dir: str = './output'  #
    eval_freq: int = 100
    checkpoint: Any = None
    checkpoint_dir: Any = './checkpoints'
    checkpoint_iterations: List[int] = field(default_factory=lambda: [])
    entropy_checkpoint: Any = None
    debug: bool = False            #
    num_workers: int = 4
    env_path: str = None
    exp_name: str = 'dev'
    description: str = None
    tags: List[str] = field(default_factory=list)


@dataclass
class FinetunePipelineParams:
    """ pipeline config """

    image_path: str

    output_dir: str = './output'  #
    eval_freq: int = 100
    checkpoint: Any = None
    checkpoint_dir: Any = './checkpoints'
    checkpoint_iterations: List[int] = field(default_factory=lambda: [])
    debug: bool = False            #
    num_workers: int = 4
    env_path: str = None
    output_path: str = None


@dataclass
class ModelParams:
    n_latent: int = 8
    n_channel: int = 16
    n_nif: int = 256
    q_scale: int = 16


@dataclass
class OptimizationParams:
    batch_size: int = 16
    lr: float = 1e-4
    entropy_lr: float = 1e-4
    syn_lr: float = 1e-4
    aux_lr: float = 1e-3
    total_epoch: int = 20
    lmbda: float =  0.001
    l2_penalty: float = 0
    crop: int = 512
    beta_z_loss: float = 0
    beta_weight_sim: float = 0
    # tf32: bool = True
    finetune_interation: int = 0
    seed: int = 0

