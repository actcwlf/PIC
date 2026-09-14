import pathlib
import sys
import os
from loguru import logger


from simple_parsing import ArgumentParser


from pipeline.finetune import finetune_model

# from utils.slang_module.linear import SlangLinear

from configurations import ModelParams, PipelineParams, OptimizationParams, FinetunePipelineParams
# from pipeline.train import train_model
from utils.common import safe_state
from utils.logging import setup_tensorboard




def main():
    parser = ArgumentParser(add_config_path_arg=True)

    parser.add_arguments(ModelParams, dest="model")
    parser.add_arguments(FinetunePipelineParams, dest="pipeline")
    parser.add_arguments(OptimizationParams, dest="optimization")

    args = parser.parse_args(sys.argv[1:])

    mp: ModelParams = args.model
    pp: FinetunePipelineParams = args.pipeline
    op: OptimizationParams = args.optimization

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

    image_name = pathlib.Path(pp.image_path).name

    tb_writer = setup_tensorboard(pp.output_dir)
    tb_writer.prefix = image_name

    logger.info("Output dir" + str(pp.output_dir.absolute()))

    # 初始化随机种子
    safe_state()

    finetune_model(
        mp=mp,
        pp=pp,
        op=op,
        tb_writer=tb_writer
    )


if __name__ == '__main__':
    main()