import slangtorch
import torch
import os
import pathlib

# import os
# os.environ["PATH"] = '/usr/local/cuda/bin:' + os.environ["PATH"] + ":/data1/liuxiang/anaconda3/envs/py310/bin"

launchBlockSize = (32, 8, 1)  # warp size由硬件决定为32，



# Before loading our module, we're going to pre-process the TORCH_CUDA_ARCH_LIST tag from the environment.
# This is to get around a bug with how torch constructs arch flags: explicitly providing arch flags seems to
# not override torch's default arch flags and cause compiler errors due to setting the SM version too low.
#
# We therefore replace the env list with a restricted list of arches that we know will work.
#
if "TORCH_CUDA_ARCH_LIST" in os.environ:
    # Go through space-separated list of arches and remove any that are below 8.0
    arches = os.environ["TORCH_CUDA_ARCH_LIST"].split(" ")
    arches = [arch for arch in arches if not int(arch.split(".")[0]) <= 7]
    os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(arches)

local_dir = pathlib.Path(__file__).parent

# print(os.environ["PATH"])

m = slangtorch.loadModule(local_dir / 'image-model.slang',
                                  defines={
                                      'NUM_THREADS_PER_BLOCK': launchBlockSize[0] * launchBlockSize[1] *
                                                               launchBlockSize[2],
                                      'WARP_SIZE': 32})

def init_slang_module():
    pass

    # # Before loading our module, we're going to pre-process the TORCH_CUDA_ARCH_LIST tag from the environment.
    # # This is to get around a bug with how torch constructs arch flags: explicitly providing arch flags seems to
    # # not override torch's default arch flags and cause compiler errors due to setting the SM version too low.
    # #
    # # We therefore replace the env list with a restricted list of arches that we know will work.
    # #
    # if "TORCH_CUDA_ARCH_LIST" in os.environ:
    #     # Go through space-separated list of arches and remove any that are below 8.0
    #     arches = os.environ["TORCH_CUDA_ARCH_LIST"].split(" ")
    #     arches = [arch for arch in arches if not int(arch.split(".")[0]) <= 7]
    #     os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(arches)
    #
    # local_dir = pathlib.Path(__file__).parent
    #
    # # print(os.environ["PATH"])
    # global SLANG_MOD
    # SLANG_MOD = slangtorch.loadModule(local_dir / 'image-model.slang',
    #                           defines={
    #                               'NUM_THREADS_PER_BLOCK': launchBlockSize[0] * launchBlockSize[1] * launchBlockSize[2],
    #                               'WARP_SIZE': 32})



class SlangLinear(torch.autograd.Function):
    def forward(ctx, x, weight, bias):
        width, height = x.shape[:2]

        # weight = weight.permute(1, 0)
        out_dim = weight.shape[1]

        output = torch.zeros((width, height, out_dim), dtype=torch.float).cuda()

        linear_layers = m.LinearLayer(layer=m.Linear(weights=weight, bias=bias))

        blockSize = launchBlockSize
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], (height + blockSize[1] - 1) // blockSize[1], 1)

        # print('grid', gridSize, 'block', blockSize)

        m.simpleLinear(linear=linear_layers, imageInput=x, imageOutput=output).launchRaw(blockSize=blockSize,
                                                                                                 gridSize=gridSize)

        ctx.save_for_backward(output, x, weight, bias)

        return output

    def backward(ctx, grad_output):
        output, x, weight, bias = ctx.saved_tensors
        # weight = args[0]
        # bias = args[1]

        # print(weight)

        weight_d = torch.zeros_like(weight)
        bias_d = torch.zeros_like(bias)
        x_d = torch.zeros_like(x)

        width, height, _ = output.shape

        linear_layers = m.LinearLayer(m.Linear(weights=(weight, weight_d), bias=(bias, bias_d)))
        # mlp = m.MLP(layers=linear_layers)

        blockSize = launchBlockSize
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], (height + blockSize[1] - 1) // blockSize[1], 1)

        m.simpleLinear.bwd(linear=linear_layers, imageInput=(x, x_d), imageOutput=(output, grad_output)).launchRaw(
            blockSize=blockSize, gridSize=gridSize)

        # print(weight_d)
        return x_d, weight_d, bias_d







