import time
from typing import Optional, List

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


from models.entropy_model import MixedEntropyBottleneck
from utils.fft import dct_2d, dct_2d_v2, idct_2d_v2
from models.mnif_generator import pad_for_patch_embedding, crop_from_padded
from utils.quantizer import NoiseQuantizer
from utils.metrics import cosine_similarity_matrix_func


class MNIFModuleMixin:
    def __init__(self, latent_dim: int, n_nif: int):
        self.mix_layer = nn.Sequential(
            nn.Linear(latent_dim, n_nif),
        )
        self.n_nif = n_nif
        self.newest_route_z_loss = None
        self.newest_cosine_sim_mat = None

    def flatten_weight(self, check=False):
        raise NotImplementedError()

    def forward_mix_weight(self, latent_pack):
        latent, temperature = latent_pack
        logits = self.mix_layer(latent)
        weight = F.softmax(logits / temperature, dim=1)
        route_z_loss = logits.exp().sum(dim=1).log().unsqueeze(1)
        self.newest_route_z_loss = route_z_loss

        flatten_weight = self.flatten_weight(check=False)
        if flatten_weight is not None:
            self.newest_cosine_sim_mat = cosine_similarity_matrix_func(flatten_weight)
        return weight


class MNIFLinear(nn.Module, MNIFModuleMixin):
    def __init__(self, in_ft: int, out_ft: int, latent_dim: int, n_nif: int):
        nn.Module.__init__(self)
        MNIFModuleMixin.__init__(self, latent_dim, n_nif)

        self.weight = nn.Parameter(
            torch.randn((n_nif, out_ft, in_ft), requires_grad=True) / out_ft ** 2)
        self.bias = nn.Parameter(torch.zeros((n_nif, out_ft), requires_grad=True))



    def forward(self, x, latent):
        mix_factor = self.forward_mix_weight(latent)
        weight = self.weight.unsqueeze(0) * mix_factor.unsqueeze(-1).unsqueeze(-1)
        weight = weight.sum(dim=1)
        bias = self.bias.unsqueeze(0) * mix_factor.unsqueeze(-1)
        bias = bias.sum(dim=1, keepdim=True)

        if len(x.shape) == 2:
            x_i = rearrange(x, 'b i -> b 1 i')
        elif len(x.shape) == 4:
            x_i = rearrange(x, 'b d1 d2 i -> b (d1 d2) i')
        else:
            x_i = x

        x_o = torch.bmm(x_i, weight.permute(0, 2, 1))
        x_o = x_o + bias
        if len(x.shape) == 2:
            x_o = x_o.squeeze(1)
        elif len(x.shape) == 4:
            x_o = rearrange(x_o, 'b (d1 d2) o -> b d1 d2 o', d1=x.shape[1], d2=x.shape[2],)

        return x_o

    def flatten_weight(self, check=False):
        if check and self.weight.numel() < self.n_nif ** 2:
            return None
        flatten_weight = self.weight.reshape(self.n_nif, -1)
        return flatten_weight




class MNIFLinearRes(MNIFLinear):
    def __init__(self, in_ft: int, out_ft: int, latent_dim: int, n_nif: int):
        super().__init__(in_ft, out_ft, latent_dim, n_nif)
        assert in_ft == out_ft

    def forward(self, x, latent):
        return super().forward(x, latent) + x


class MNIFConv2d(nn.Conv2d, MNIFModuleMixin):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            latent_dim: int,
            n_nif: int,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None,
    ) -> None:
        assert padding_mode == 'zeros'
        nn.Conv2d.__init__(
            self,
            in_channels * n_nif,
            out_channels * n_nif,
            kernel_size,
            stride,
            padding,
            dilation,
            groups * n_nif,
            bias,
            padding_mode,
            device,
            dtype
        )

        MNIFModuleMixin.__init__(self, latent_dim, n_nif)

        self.groups = groups
        self.register_parameter('dummy_bias', None)
        self.n_nif = n_nif


    def forward(self, x, latent) -> Tensor:
        mix_factor = self.forward_mix_weight(latent)
        b, c, h, w = x.shape

        weight = rearrange(self.weight, '(n c_o) c_i k1 k2 -> 1 n c_o c_i k1 k2', n=self.n_nif)
        weight = weight * rearrange(mix_factor, 'b n -> b n 1 1 1 1')
        weight = weight.sum(dim=1)
        weight = rearrange(weight, 'b c_o c_i k1 k2 -> (b c_o) c_i k1 k2')

        # print('w', weight.shape, weight[:4, ...].view(-1))

        bias = rearrange(self.bias, '(n c_o) -> 1 n c_o', n=self.n_nif)
        bias = bias * rearrange(mix_factor, 'b n -> b n 1')
        bias = bias.sum(dim=1)
        bias = rearrange(bias, 'b c_o -> (b c_o)')

        x = rearrange(x, 'b c_i h w -> 1 (b c_i) h w')
        # print('x', x.shape)

        out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups * b)
        # out = out + bias

        out = rearrange(out, '1 (b c_o) h w -> b c_o h w', b=b)
        return out

    def flatten_weight(self, check=False):
        if check and self.weight.numel() < self.n_nif ** 2:
            return None
        flatten_weight = rearrange(self.weight, '(n c_o) c_i k1 k2 -> n (c_o c_i k1 k2)', n=self.n_nif)
        return flatten_weight


class MNIFConvTranspose2d(nn.ConvTranspose2d, MNIFModuleMixin):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            latent_dim: int,
            n_nif: int,
            stride= 1,
            padding = 0,
            output_padding = 0,
            groups: int = 1,
            bias: bool = True,
            dilation = 1,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        assert padding_mode == 'zeros'
        nn.ConvTranspose2d.__init__(self,
            in_channels * n_nif,
            out_channels * n_nif,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups * n_nif,
            bias,
            dilation,
            padding_mode,
            device,
            dtype
        )
        MNIFModuleMixin.__init__(self, latent_dim, n_nif)
        self.groups = groups
        self.register_parameter('dummy_bias', None)
        self.n_nif = n_nif

    def forward(self, input: Tensor, latent, output_size: Optional[List[int]] = None) -> Tensor:
        mix_factor = self.forward_mix_weight(latent)
        b, c, h, w = input.shape

        weight = rearrange(self.weight, '(n c_i) c_o k1 k2 -> 1 n c_i c_o k1 k2', n=self.n_nif)
        weight = weight * rearrange(mix_factor, 'b n -> b n 1 1 1 1')
        weight = weight.sum(dim=1)
        weight = rearrange(weight, 'b c_i c_o k1 k2 ->  (b c_i)  c_o k1 k2')

        # print('w', weight.shape, weight[:4, ...].view(-1))

        if self.bias is not None:
            bias = rearrange(self.bias, '(n c_o) -> 1 n c_o', n=self.n_nif)
            bias = bias * rearrange(mix_factor, 'b n -> b n 1')
            bias = bias.sum(dim=1)
            bias = rearrange(bias, 'b c_o -> (b c_o)')
        else:
            bias = self.bias

        x = rearrange(input, 'b c_i h w -> 1 (b c_i) h w')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        out = F.conv_transpose2d(
            x, weight, bias, self.stride, self.padding,
            output_padding, self.groups * b, self.dilation)

        out = rearrange(out, '1 (b c_o) h w -> b c_o h w', b=b)
        return out


class MNIFSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input, net_latent):
        for module in self:
            if isinstance(module, MNIFModuleMixin):
                input = module(input, net_latent)
            else:
                input = module(input)
        return input



class DCTLayer(nn.Module):
    def __init__(self, n_latent, patch_size=8, q_scale=16):
        super().__init__()
        self.patch_size = patch_size
        self.n_latent = n_latent
        # self.quant_table = nn.ParameterList(
        #     [nn.Parameter(torch.ones((1, 1, patch_size, patch_size)) * 16) for _ in range(n_latent)]
        # )
        self.noise_quantizer = NoiseQuantizer()
        # self.l0_norm_list = nn.ModuleList([
        #     L0Norm() for i in range(n_latent - 1)
        # ])

        self.bottleneck = MixedEntropyBottleneck(channels=64)

        self.q_scale = q_scale


    def dct_mask(self):
        return [
            l0_norm.sample_z(x=None, sample=False) for l0_norm in self.l0_norm_list
        ]

    def dct_and_quant(self, latents):
        shapes = [l.shape for l in latents]

        padded_latent = [
            pad_for_patch_embedding(l, patch_size=8) for l in latents
        ]
        n_patchs = [(l.shape[-2] // self.patch_size, l.shape[-1] // self.patch_size) for l in padded_latent]

        reshaped_latents = [
            rearrange(l, 'b c (nh p1) (nw p2) -> b (c nh nw) p1 p2', p1=self.patch_size, p2=self.patch_size) for l in padded_latent
        ]

        dct_latents = [
            dct_2d(l, norm='ortho') for l in reshaped_latents
        ]

        scaled_dct_latents = [
            dct_latents[i] * self.q_scale for i in range(self.n_latent)
        ]

        quant_dct_latent = [
            torch.round(l) for l in scaled_dct_latents
        ]

        return quant_dct_latent


    def forward(self, latents, l2_penalty=0):
        shapes = [l.shape for l in latents]

        padded_latent = [
            pad_for_patch_embedding(l, patch_size=8) for l in latents
        ]
        n_patchs = [(l.shape[-2] // self.patch_size, l.shape[-1] // self.patch_size) for l in padded_latent]

        reshaped_latents = [
            rearrange(l, 'b c (nh p1) (nw p2) -> b (c nh nw) p1 p2', p1=self.patch_size, p2=self.patch_size) for l in padded_latent
        ]

        dct_latents = [
            dct_2d_v2(l, norm='ortho') for l in reshaped_latents
        ]

        scaled_dct_latents = [
            dct_latents[i] * self.q_scale for i in range(self.n_latent)
        ]

        # if self.training:
        #     quant_dct_latent = [
        #         self.noise_quantizer.apply(l) for l in scaled_dct_latents
        #     ]
        # else:
        #     quant_dct_latent = [
        #         torch.round(l) for l in scaled_dct_latents
        #     ]

        rearranged_latents = [
            rearrange(l, 'b n p1 p2 -> b (p1 p2) n') for l in scaled_dct_latents
        ]

        latents_and_likelihoods = [
            self.bottleneck(latent, self.training) for latent in rearranged_latents
        ]


        restored_latent_dcts = [
            rearrange(p[0], 'b (p1 p2) n -> b n p1 p2', p1=self.patch_size, p2=self.patch_size) for p in latents_and_likelihoods
        ]

        # check_tensor(restored_latent_dcts[0])

        likelihoods = [
            p[1] for p in latents_and_likelihoods
        ]

        rates = [
            torch.log(llh).sum() / (-math.log(2)) for llh in likelihoods
        ]

        unscaled_dct_latents = [
            restored_latent_dcts[i] / self.q_scale for i in range(self.n_latent)
        ]
        # start_time = time.time()
        idct_latents = [
            idct_2d_v2(l, norm='ortho') for l in unscaled_dct_latents
        ]

        recovered_latents = [
            rearrange(
                idct_latents[i],
                'b (c nh nw) p1 p2 -> b c (nh p1) (nw p2 )',
                nh=n_patchs[i][0], nw=n_patchs[i][1], c=1, p1=self.patch_size, p2=self.patch_size
            ) for i in range(self.n_latent)
        ]

        cropped_padded_latents = [
            crop_from_padded(recovered_latents[i], shapes[i][-2:], patch_size=self.patch_size) for i in range(self.n_latent)
        ]

        # check_tensor(cropped_padded_latents[0])
        # torch.cuda.synchronize()
        # print('idct', time.time() - start_time)


        return cropped_padded_latents, sum(rates)




# LIMIT_A, LIMIT_B, EPSILON = -.1, 1.1, 1e-6

class L0Norm(nn.Module):
    def __init__(self, lmbda=0.04, drop_rate_init=0.5, l2_penalty=0.02, temperature=2./3.,):
        super().__init__()
        self.qz_loga = nn.Parameter(torch.Tensor(1, 1, 8, 8))
        self.temperature = temperature
        self.drop_rate_init = drop_rate_init if drop_rate_init != 0. else 0.5
        # self.l2_penalty = l2_penalty
        self.limit_a = -.1
        self.limit_b = 1.1
        self.epsilon = 1e-6
        # self.lmbda = lmbda

        self.reset_parameters()
        # print(self)

    def reset_parameters(self):


        # Initialize the log-alpha parameters (qz_loga) for the concrete distribution
        # The mean is set to log(1 - droprate_init) - log(droprate_init), which centers the initial dropout rate
        # around the specified droprate_init value. The standard deviation is set to 1e-2.
        self.qz_loga.data.normal_(math.log(1 - self.drop_rate_init) - math.log(self.drop_rate_init), 1e-2)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        # Normalize the input x to the range [0, 1] using the limits limit_a and limit_b
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)

        # Compute the logits for the concrete distribution. This is done by taking the log-odds of xn,
        # which is log(xn / (1 - xn)). This transforms the normalized input into a logit space.
        logits = math.log(xn) - math.log(1 - xn)

        # Apply the temperature-scaled sigmoid function to the logits, subtracting the log-alpha parameters (qz_loga).
        # The result is clamped to the range [epsilon, 1 - epsilon] to avoid numerical instability.
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=self.epsilon, max=1 - self.epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        # Compute the quantile (inverse CDF) of the 'stretched' concrete distribution.
        # The input `x` is a sample from a uniform distribution in the range [0, 1].
        # The formula applies the sigmoid function to the log-odds of `x`, adjusted by the log-alpha parameters (qz_loga)
        # and scaled by the temperature. This transforms the uniform sample into a sample from the concrete distribution.
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)

        # Stretch the output `y` from the range [0, 1] to the range [limit_a, limit_b].
        # This ensures that the output values are within the desired bounds for the hard concrete distribution.
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def reg_x(self, x, l2_penalty):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        # Compute the log-prior for the weights (logpw_col) by summing over the input features.
        # The log-prior is calculated as the negative of the sum of two terms:
        # 1. The L2 regularization term: 0.5 * self.prior_prec * self.weights.pow(2)
        # 2. The L0 regularization term: self.lamba
        # The result is summed over the input features (dimension 1).

        # 对应原论文的形式为
        # logpw_col = torch.sum(- (.5 * l2_penalty * x.pow(2)) - lmbda, 1, keepdim=True)
        # 为了便于调节码率这里实际使用的是变形后的形式
        # logpw_col = torch.sum(- (.5 * l2_penalty / lmbda * x.pow(2)) - 1, 1, keepdim=True)
        # 这种形式下  l2_penalty 是原公式中 l2_penalty / lmbda 的复合参数
        logpw_col = torch.sum(- (.5 * l2_penalty * x.pow(2)) - 1, 1, keepdim=True)

        # Compute the expected log-prior for the weights (logpw) by weighting logpw_col with the probability
        # that the corresponding gate is active (1 - self.cdf_qz(0)). This accounts for the stochastic nature
        # of the gates in the L0 regularization.

        # cdf_qz_val = self.cdf_qz(0).unsqueeze(0).unsqueeze(0)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col, (1, 2, 3))



        # Return the total regularization term, which is the sum of the log-prior for the weights and the bias.
        return logpw

    def regularization(self):
        return self._reg_w()



    def get_eps(self, x):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.empty_like(x).uniform_(self.epsilon, 1 - self.epsilon)

        return eps


    def sample_z(self, x, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(x)
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga)
            return F.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)


    def forward(self, x, l2_penalty):
        z = self.sample_z(x, sample=self.training)

        # if not self.training:
        #     print(z[0, 0])
        masked_x = x * z
        reg = self.reg_x(x, l2_penalty)
        return masked_x, reg