from typing import Optional, Tuple

import numpy as np
import torch
from compressai.entropy_models import EntropyBottleneck
from torch import Tensor

# from utils.codec.pic_bitstream_codec import MAX_BOUND
# from utils.inspector import check_tensor

MAX_BOUND = 126
from utils.quantizer import NoiseQuantizer, STEQuantizer


class MixedEntropyBottleneck(EntropyBottleneck):
    def __init__(self,
        channels: int,
        *args,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters = (3, 3, 3, 3),
        **kwargs):
        super().__init__(channels, *args,
                         tail_mass=tail_mass, init_scale=init_scale, filters=filters, **kwargs
                         )


        self.noise_quantizer = NoiseQuantizer()
        self.ste_quantizer = STEQuantizer()

    # def quantize(
    #     self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    # ) -> Tensor:
    #
    #     if mode not in ("noise", "dequantize", "symbols"):
    #         raise ValueError(f'Invalid quantization mode: "{mode}"')
    #
    #     mask = (inputs.abs() < 1).long()  # 使用ste
    #
    #     if mode == "noise":
    #         half = float(0.5)
    #         noise = torch.empty_like(inputs).uniform_(-half, half)
    #         noise_part = (inputs + noise) * (1 - mask)
    #
    #         ste_part = self.ste_quantizer.apply(inputs) * mask
    #
    #         return noise_part + ste_part
    #
    #     outputs = inputs.clone()
    #     if means is not None:
    #         outputs -= means
    #
    #     outputs = torch.round(outputs)
    #
    #     if mode == "dequantize":
    #         if means is not None:
    #             outputs += means
    #         return outputs
    #
    #     assert mode == "symbols", mode
    #     outputs = outputs.int()
    #     return outputs

    def forward(
        self, x: Tensor, training: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor]:


        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            raise NotImplementedError()
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            # perm = (1, 2, 3, 0)
            # inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize

        # outputs = self.quantize(
        #     values, "noise" if training else "dequantize", self._get_medians()
        # )

        outputs = self.quantize(
            values, "noise" if training else "dequantize"
        )

        if not training:
            outputs = outputs.clamp(min=-MAX_BOUND + 1, max=MAX_BOUND - 1)  # 约束符号在uint8可表示的范围内

        if not torch.jit.is_scripting():
            likelihood, _, _ = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
            # TorchScript not yet supported
            # likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        # check_tensor(means)
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        mask = (inputs.abs() < 1).long()  # 使用ste

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            noise_part = (inputs + noise) * (1 - mask)

            ste_part = self.ste_quantizer.apply(inputs) * mask

            return noise_part + ste_part

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

