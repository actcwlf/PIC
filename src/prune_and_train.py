import pathlib
import sys
from loguru import logger


from simple_parsing import ArgumentParser

from configurations import ModelParams, PipelineParams, OptimizationParams
from pipeline.prune_and_train import prune_and_train_model
from utils.common import safe_state
from utils.logging import setup_tensorboard


def main():
    parser = ArgumentParser(add_config_path_arg=True)

    parser.add_arguments(ModelParams, dest="model")
    parser.add_arguments(PipelineParams, dest="pipeline")
    parser.add_arguments(OptimizationParams, dest="optimization")

    args = parser.parse_args(sys.argv[1:])

    mp: ModelParams = args.model
    pp: PipelineParams = args.pipeline
    op: OptimizationParams = args.optimization

    # 创建输出目录
    pp.output_dir = pathlib.Path(pp.output_dir)
    pp.output_dir.mkdir(exist_ok=True)

    pp.checkpoint_dir = pathlib.Path(pp.output_dir / 'checkpoints')
    pp.checkpoint_dir.mkdir(exist_ok=True)

    # 初始化日志文件
    logger.add(pp.output_dir / 'output.log')

    logger.info(f'args: {args}')
    # logger.info(f'running tag: {args.tag}')

    dataset = pathlib.Path(pp.train_data_dir).name

    tb_writer = setup_tensorboard(pp.output_dir)
    tb_writer.prefix = dataset

    logger.info("Output dir" + str(pp.output_dir.absolute()))

    # 初始化随机种子
    safe_state()

    prune_and_train_model(
        mp=mp,
        pp=pp,
        op=op,
        tb_writer=tb_writer
    )


if __name__ == '__main__':
    main()