import pathlib
import sys
import os

import swanlab
from loguru import logger

from simple_parsing import ArgumentParser

from configurations import ModelParams, PipelineParams, OptimizationParams
from pipeline.train import train_model
from utils.common import safe_state




def main():
    parser = ArgumentParser(add_config_path_arg=True)

    parser.add_arguments(ModelParams, dest="model")
    parser.add_arguments(PipelineParams, dest="pipeline")
    parser.add_arguments(OptimizationParams, dest="optimization")

    args = parser.parse_args(sys.argv[1:])

    mp: ModelParams = args.model
    pp: PipelineParams = args.pipeline
    op: OptimizationParams = args.optimization

    swanlab.init(
        # 设置将记录此次运行的项目信息
        project="PIC_CB",
        workspace="w0lf",
        experiment_name=pp.exp_name,
        description=pp.description,
        tags=pp.tags,
        # 跟踪超参数和运行元数据
        # config={
        #     "learning_rate": 0.02,
        #     "architecture": "CNN",
        #     "dataset": "CIFAR-100",
        #     "epochs": 10
        # }
    )


    # 创建输出目录
    pp.output_dir = pathlib.Path(pp.output_dir)
    pp.output_dir.mkdir(exist_ok=True)

    pp.checkpoint_dir = pathlib.Path(pp.output_dir / 'checkpoints')
    pp.checkpoint_dir.mkdir(exist_ok=True)

    # if pp.env_path is not None:
    #     os.environ["PATH"] = pp.env_path + ':' + os.environ["PATH"]

    # 初始化日志文件
    logger.add(pp.output_dir / 'output.log')

    logger.info(f'args: {args}')
    # logger.info(f'running tag: {args.tag}')

    dataset = pathlib.Path(pp.train_data_dir).name

    # tb_writer = setup_tensorboard(pp.output_dir)
    # tb_writer.prefix = dataset

    logger.info("Output dir" + str(pp.output_dir.absolute()))

    # 初始化随机种子
    safe_state(op.seed)

    train_model(
        mp=mp,
        pp=pp,
        op=op,
        # tb_writer=tb_writer
    )


if __name__ == '__main__':
    main()