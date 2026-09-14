import torch
from torch import Tensor

class NoiseQuantizer(torch.autograd.Function):
    noise_derivative: float = 100.

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        y = x + (torch.rand_like(x) - 0.5)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out * NoiseQuantizer.noise_derivative


class STEQuantizer(torch.autograd.Function):
    ste_derivative: float = 1e-2

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        y = torch.round(x)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out * STEQuantizer.ste_derivative

def quantize(x: Tensor, training: bool) -> Tensor:
    """Quantize a tensor with a unitary quantization step

    Args:
        x (Tensor): Tensor to be quantized
        training (bool): True if we're training. In this case we use the
            additive noise model. Otherwise, the actual quantization (round)
            is used
        log_2_gain (Tensor): Tensor of shape [1] containing the quantization gain.

    Returns:
        Tensor: The quantized version of x.
    """
    return x + (torch.rand_like(x) - 0.5) if training else torch.round(x)
