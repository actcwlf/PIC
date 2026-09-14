import time
from typing import cast

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
from torch import Tensor

from models.encoder import ResidualEncoder, ModulationEncoder
from models.layers import DCTLayer
from models.synthesis import  ModulationSynthesis, DecoderSynthesis
from models.upsampling import  BilinearUpSample
from utils.codec.pic_bitstream_codec import PICBitstreamCodec, MAX_BOUND



class DCC(nn.Module):
    def __init__(self, n_latent, n_channel, n_nif, q_scale=16):
        super().__init__()
        self.encoder = ResidualEncoder(n_latent=n_latent, n_channel=n_channel)
        # self.mnif_generator = MNIFGenerator(in_channels=n_channel, n_latent=n_latent)
        self.upsample_layer = BilinearUpSample()
        self.synthesis = ModulationSynthesis(in_ft=n_latent)

        self.modulation_encoder = ModulationEncoder(n_channel=32, out_dim=16)

        self.temperature = 1

        self.dct_layer = DCTLayer(n_latent=n_latent, q_scale=q_scale)

        # self.codec = HuffmanCodec()

        self.codec = PICBitstreamCodec()

        # self.noise_quantizer = NoiseQuantizer()
        # self.ste_quantizer = STEQuantizer()

    def forward(self, x, l2_penalty=0):
        features, latents = self.encoder(x)
        # scaled_latent = [
        #     latent * 16 for latent in latents
        # ]
        gamma, beta = self.modulation_encoder(x)

        # net_latents = []
        # for feat in features:
        #     feat = feat.mean(dim=(-2, -1))
        #     net_latents.append(feat)
        # net_latent = torch.cat(net_latents, dim=1)

        # quantized_latent = [
        #     self.noise_quantizer.apply(cur_latent) for cur_latent in scaled_latent
        # ]

        quantized_latent, rate_y = self.dct_layer(latents, l2_penalty)

        # entropy_net_latent = self.entropy_mnif_generator(quantized_latent)
        # quantized_latent = scaled_latent
        # proba_params = self.entropy_net(quantized_latent, (entropy_net_latent, self.temperature))

        # Compute the rate (i.e. the entropy of flat latent knowing mu and scale)
        # rate_y = torch.Tensor([0]).to(latents[0].device)
        # for latent, proba in zip(quantized_latent, proba_params):
        #     # print(latent.shape, proba.shape)
        #     cur_rate_y, _, _ = compute_rate(latent, proba)
        #     rate_y += cur_rate_y.sum()
        # start_time = time.time()
        # Reconstruct the output
        synthesis_input = self.upsample_layer(quantized_latent)
        # torch.cuda.synchronize()
        # print('rec', time.time() - start_time)

        # syn_net_latent = self.mnif_generator(features)

        # check_tensor(synthesis_input)
        synthesis_output = self.synthesis(synthesis_input, gamma, beta)

        # torch.cuda.synchronize()
        # print('rec', time.time() - start_time)

        x_hat = torch.clamp(synthesis_output, 0., 1.)

        out = {'x_hat': x_hat, 'rate_y': rate_y}
        return out


    def inference_forward(self, quantized_latent, gamma, beta):
        synthesis_input = self.upsample_layer(quantized_latent)
        # torch.cuda.synchronize()
        # print('rec', time.time() - start_time)

        # syn_net_latent = self.mnif_generator(features)
        synthesis_output = self.synthesis(synthesis_input, gamma, beta)

        # torch.cuda.synchronize()
        # print('rec', time.time() - start_time)

        x_hat = torch.clamp(synthesis_output, 0., 1.)

        out = {'x_hat': x_hat}
        return out


    @torch.no_grad()
    def encode(self, x, idx=None):
        self.eval()
        features, latents = self.encoder(x)
        gamma, beta = self.modulation_encoder(x)

        # for i in range(len(latents)):
        #     check_tensor(latents[i])


        latents = [torch.clamp(latent, min=-MAX_BOUND + 1, max=MAX_BOUND -1) for latent in latents]

        pack = (
            gamma, beta,
            self.synthesis.weight0, self.synthesis.bias0,
            self.synthesis.weight1, self.synthesis.bias1,
            self.synthesis.weight2, self.synthesis.bias2,
            self.synthesis.weight3, self.synthesis.bias3,
            latents,

        )

        if idx is not None:
            # name =  f'_data/evals/img_idx{idx + 1:02d}.pth'
            # name = f'_data/eval_clip/img_idx14_clip.pth'
            name = f'_data/clic/img{idx + 1:02d}.pth'
            print('saving', name)
            torch.save(pack, name)

        x_hat, bpp, bs = self.codec.encode(pack)

        # x_hat, bs = self.codec.forward(pack, self.dct_layer.bottleneck)





        # encoded, bit_len = self.dct_layer.encode(self.codec, latents)

        # quantized_latent, reg = self.dct_layer(latents)
        #
        # diff = [
        #     (a - b).abs().max() for a, b in zip(encoded, quantized_latent)
        # ]
        #
        # print(diff)

        return x_hat, bs, gamma, beta

    def aux_loss(self) -> Tensor:
        r"""Returns the total auxiliary loss over all ``EntropyBottleneck``\s.

        In contrast to the primary "net" loss used by the "net"
        optimizer, the "aux" loss is only used by the "aux" optimizer to
        update *only* the ``EntropyBottleneck.quantiles`` parameters. In
        fact, the "aux" loss does not depend on image data at all.

        The purpose of the "aux" loss is to determine the range within
        which most of the mass of a given distribution is contained, as
        well as its median (i.e. 50% probability). That is, for a given
        distribution, the "aux" loss converges towards satisfying the
        following conditions for some chosen ``tail_mass`` probability:

        * ``cdf(quantiles[0]) = tail_mass / 2``
        * ``cdf(quantiles[1]) = 0.5``
        * ``cdf(quantiles[2]) = 1 - tail_mass / 2``

        This ensures that the concrete ``_quantized_cdf``\s operate
        primarily within a finitely supported region. Any symbols
        outside this range must be coded using some alternative method
        that does *not* involve the ``_quantized_cdf``\s. Luckily, one
        may choose a ``tail_mass`` probability that is sufficiently
        small so that this rarely occurs. It is important that we work
        with ``_quantized_cdf``\s that have a small finite support;
        otherwise, entropy coding runtime performance would suffer.
        Thus, ``tail_mass`` should not be too small, either!
        """
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)






    def get_net_latent(self, x):
        features, latents = self.encoder(x)

        # features -> latent
        net_latent = self.mnif_generator(features)
        syn_mnif_weight = self.synthesis.get_mnif_weight(net_latent)
        aru_mnif_weight = self.entropy_net.get_mnif_weight(net_latent)

        mnif_weight = {**syn_mnif_weight, **aru_mnif_weight}

        return net_latent, mnif_weight
    def capture(self):
        return self.state_dict()

    def restore(self, state_dict):
        self.load_state_dict(state_dict)

    def collect_z_loss(self):
        return torch.cat([
            self.synthesis.collect_z_loss(),
            # self.entropy_net.collect_z_loss()
        ], dim=1)

    def collect_weight_sim(self):
        return torch.cat([
            self.synthesis.collect_weight_sim(),
            # self.entropy_net.collect_weight_sim()
        ], dim=0)


class PICWrapper(nn.Module):

    # @torch.no_grad()
    def  __init__(self, pic: DCC, target_img):
        super().__init__()
        # self.pic_model = pic
        pic.eval()
        bs, gamma, beta, inter_y = pic.encode(target_img)
        self.dct_layer: DCTLayer = pic.dct_layer

        features, latents = pic.encoder(target_img)

        # print('pic', check_tensor(latents[0]))
        self.latents = nn.ParameterList([
            nn.Parameter(l, requires_grad=True) for l in latents
        ])

        self.upsample_layer = BilinearUpSample()
        self.synthesis = DecoderSynthesis(pic.synthesis, inter_y, gamma, beta)

        self.codec = PICBitstreamCodec()

    def forward(self):

        quantized_latent, rate_y = self.dct_layer(self.latents, 0)


        synthesis_input = self.upsample_layer(quantized_latent)
        synthesis_output = self.synthesis(synthesis_input)

        x_hat = torch.clamp(synthesis_output, 0., 1.)

        out = {'x_hat': x_hat, 'rate_y': rate_y}
        return out


    def encode(self):
        pack = (
            self.synthesis.weight0, self.synthesis.bias0,
            self.synthesis.weight1, self.synthesis.bias1,
            self.synthesis.weight2, self.synthesis.bias2,
            self.synthesis.weight3, self.synthesis.bias3,
            self.latents,

        )
        return self.codec.encode_finetuned(pack)

    def decode(self, bitstream):
        return self.codec.decode(bitstream)

    def aux_loss(self) -> Tensor:
        r"""Returns the total auxiliary loss over all ``EntropyBottleneck``\s.

        In contrast to the primary "net" loss used by the "net"
        optimizer, the "aux" loss is only used by the "aux" optimizer to
        update *only* the ``EntropyBottleneck.quantiles`` parameters. In
        fact, the "aux" loss does not depend on image data at all.

        The purpose of the "aux" loss is to determine the range within
        which most of the mass of a given distribution is contained, as
        well as its median (i.e. 50% probability). That is, for a given
        distribution, the "aux" loss converges towards satisfying the
        following conditions for some chosen ``tail_mass`` probability:

        * ``cdf(quantiles[0]) = tail_mass / 2``
        * ``cdf(quantiles[1]) = 0.5``
        * ``cdf(quantiles[2]) = 1 - tail_mass / 2``

        This ensures that the concrete ``_quantized_cdf``\s operate
        primarily within a finitely supported region. Any symbols
        outside this range must be coded using some alternative method
        that does *not* involve the ``_quantized_cdf``\s. Luckily, one
        may choose a ``tail_mass`` probability that is sufficiently
        small so that this rarely occurs. It is important that we work
        with ``_quantized_cdf``\s that have a small finite support;
        otherwise, entropy coding runtime performance would suffer.
        Thus, ``tail_mass`` should not be too small, either!
        """
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)
