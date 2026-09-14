import math
import time
import os
import sys
import pickle
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from tqdm import tqdm
from einops import rearrange
import torch
import torch.nn.functional as F


from csrc.pic_codec import HuffmanCodec
from utils.inspector import check_tensor
# from slang_mlp.image_model import RenderImage, m, launchBlockSize
from utils.slang_module.linear import SlangLinear

from models.mnif_generator import pad_for_patch_embedding, crop_from_padded
from utils.fft import dct_2d_v2, idct_2d_v2

# from utils.libs import rec_binding, cuhd_binding, dct_binding
from csrc import pic_codec

MAX_BOUND = 126
BOUND_OFFSET = 128

ZIGZAG_ORDER = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,
                7, 14, 21, 28, 35, 42,
                49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
                47, 55, 62, 63]


def inplace_idct(x, scale):
    pic_codec.fast_idct(x, scale)



def expand_blocks_zigzag(symbols, indices, fill_value, total_blocks):
    # print(indices.shape, indices[-5:])
    return pic_codec.expand_blocks_zigzag(symbols, indices, fill_value, total_blocks)


class PICBitstreamCodec:
    def __init__(self, max_bit_length=11):
        self.max_bit_length = max_bit_length
        self.codec = HuffmanCodec(self.max_bit_length, 8)
        self.local_pic_codec = pic_codec.PICCodec(self.max_bit_length, 8)
        self.device = 'cuda'

        self.w0_base = torch.zeros((16, 16), dtype=torch.float, device=self.device)
        self.w1_base = torch.zeros((16, 16), dtype=torch.float, device=self.device)
        self.w2_base = torch.zeros((16, 16), dtype=torch.float, device=self.device)
        self.w3_base = torch.zeros((16, 16), dtype=torch.float, device=self.device)

        self.b0_base = torch.zeros((16,), dtype=torch.float, device=self.device)
        self.b1_base = torch.zeros((16,), dtype=torch.float, device=self.device)
        self.b2_base = torch.zeros((16,), dtype=torch.float, device=self.device)
        self.b3_base = torch.zeros((16,), dtype=torch.float, device=self.device)

        # print('before render')
        self.output = torch.zeros((768, 512, 3), dtype=torch.float, device=self.device)
        self.patch_size = 8
        # self.n_latent = 7
        self.q_scale = 16

    def reset_codec(self):
        self.codec = HuffmanCodec(self.max_bit_length, 8)

    def encode_latent(self, latents):
        patch_size = 8
        q_scale = 16


        padded_latent = [
            pad_for_patch_embedding(l, patch_size=8) for l in latents
        ]
        # n_patchs = [(l.shape[-2] // patch_size, l.shape[-1] // patch_size) for l in padded_latent]

        # plt.imshow(padded_latent[0].squeeze(0).squeeze(0).detach().cpu().data.numpy())
        # plt.savefig('_data/latent_inter.png')
        # plt.show()

        reshaped_latents = [
            rearrange(l, 'b c (nh p1) (nw p2) -> b (c nh nw) p1 p2', p1=patch_size, p2=patch_size) for l in
            padded_latent
        ]

        # print(reshaped_latents[0][0, 0])

        # dct_latents = [
        #     dct_2d_v2(l, norm='ortho') for l in reshaped_latents
        # ]

        merged_blocks = torch.cat(reshaped_latents, dim=1).contiguous()
        # ref_merged_blocks = merged_blocks.clone()
        merged_blocks = dct_2d_v2(merged_blocks, norm='ortho')
        merged_blocks = merged_blocks * q_scale
        merged_blocks = torch.round(merged_blocks)

        # check_tensor(merged_blocks)

        merged_blocks = merged_blocks.clamp(min=-MAX_BOUND + 1, max=MAX_BOUND - 1 - 17)
        total_blocks = merged_blocks.shape[1]

        # print('in encoding', merged_blocks.max())

        # ret = encode_blocks(codec, merged_blocks)

        full_blocks = merged_blocks.squeeze(0).to(torch.int32).contiguous()

        # print('current max dev', full_blocks.abs().max())

        new_symbols = pic_codec.zigzag_trim_and_reduce_blocks(full_blocks, MAX_BOUND).cpu()

        # tensor_symbols = torch.cat([new_symbols, torch.tensor([0])])  # workaround to solve bug in gpuhd

        tensor_symbols = (new_symbols + BOUND_OFFSET).to(torch.uint8)

        # print('new_symbols', tensor_symbols[-5:], tensor_symbols.shape)

        ret = self.codec.encode(tensor_symbols)

        # decoded = self.codec.decode(
        #     ret[0],
        #     ret[1],
        #     ret[2],
        #     ret[3].cuda(),
        #     tensor_symbols.shape[0])

        drop_last = True

        # workaround to solve bug in gpuhd
        # padding_symbol = 1
        # while not (decoded.cpu() == tensor_symbols.cpu()).all():
        #     print('reencode')
        #     tensor_symbols[-1] = padding_symbol
        #     ret = self.codec.encode(tensor_symbols)

        #     decoded = self.codec.decode(
        #         ret[0],
        #         ret[1],
        #         ret[2],
        #         ret[3].cuda(),
        #         tensor_symbols.shape[0])

        #     padding_symbol += 1
        #     if padding_symbol > 31:
        #         break

        #     drop_last =False

        return {
            'compact_tensor_dec_table': ret[0],
            'num_entries': ret[1],
            'compressed_size': ret[2],
            'tensor_compressed': ret[3],
            'total_symbols': tensor_symbols.shape[0],
            'total_blocks': total_blocks,
            'merged_blocks': full_blocks,
            'raw_symbols': tensor_symbols[:-1],
            'padded_symbols': tensor_symbols,
            'drop_last': drop_last
        }

    def encode(self, model_output_pack, debug=False):
        (
            gamma, beta,
            w0, b0,
            w1, b1,
            w2, b2,
            w3, b3,
            latents,

        ) = model_output_pack

        # print(latents[0].shape)
        latents = [latent.contiguous() for latent in latents]


        _, _, h, w = latents[0].shape
        padded_h = (h + 7) // 8 * 8
        padded_w = (w + 7) // 8 * 8
        n_latent = len(latents)

        ret = self.encode_latent(latents)

        # print('total_symbols', ret['total_symbols'])

        start = time.time()
        decoded = self.codec.decode(
            ret['compact_tensor_dec_table'],
            ret['num_entries'],
            ret['compressed_size'],
            ret['tensor_compressed'].cuda(),
            ret['total_symbols'])

        # print(ret['compressed_size'], ret['tensor_compressed'].shape)
        # exit()

        # print(decoded[-10:])

        # pack = {
        #     'padded_symbols': ret['padded_symbols'].cpu().clone(),
        #     'decoded_symbols': decoded.cpu().clone()
        # }
        # print()
        # print(decoded[-5:])
        # print(ret['padded_symbols'][-5:])

        # if ret['drop_last']:

        # decoded = decoded[:-1]  # 编码阶由于gpuhd的bug，多编码了一个占位符，此处丢弃
        # try:
        #     assert (decoded.cpu()[:-5] == ret['raw_symbols'].cpu()[:-5]).all()
        # except:
        #     torch.save(pack, '_data/decoded_error_pth')
        #     exit()
        # decoded[-1] = 254 # bug fix for gpuhd
        tmp_v = (decoded == MAX_BOUND + BOUND_OFFSET)
        # print(decoded.shape, decoded.reshape(-1)[-5:], tmp_v.reshape(-1)[-5:])
        indices = tmp_v.nonzero().to(torch.int32)
        # print(indices.shape,indices[:5], indices[-5:])
        bd = expand_blocks_zigzag(decoded, indices, BOUND_OFFSET, ret['total_blocks'])  # .to(torch.float) - 128
        merged_blocks = bd.reshape(-1, 8, 8)

        # ref_merged_blocks = ret['merged_blocks']

        # print(ref_merged_blocks[0,  ...])
        # print( merged_blocks[0, ...])

        # assert torch.allclose(ref_merged_blocks.squeeze(0), merged_blocks)

        # check_tensor(merged_blocks)

        inplace_idct(merged_blocks, 1 / 16.0)

        # print(merged_blocks.shape, merged_blocks[0])

        # check_tensor(merged_blocks)

        # num_blocks0 = 512 * 768 // 64
        # patch1 = merged_blocks[:num_blocks0]

        # ref_latent0 = rearrange(patch1, '( nh nw) p1 p2 -> (nh p1) (nw p2)', p1=8, p2=8, nh=padded_h //8, nw = padded_w // 8)

        # plt.imshow(ref_latent0.detach().cpu().data.numpy())
        # plt.savefig('_data/latent_inter_ref.png')
        # plt.show()

        cloned_merged_blocks = merged_blocks.clone()
        y = pic_codec.split_reshape_crop_interpolate(merged_blocks, w, h, n_latent)

        # inter_y = y[:padded_h * padded_w].reshape(padded_h, padded_w)

        # plt.imshow(inter_y.detach().cpu().data.numpy())
        # plt.savefig('_data/latent_inter.png')
        # plt.show()

        # exit()

        inter_y = pic_codec.crop_and_interpolate(y, w, h, n_latent)

        # check_tensor(inter_y)

        # x = inter_y.reshape(7, 768, 512).permute(1, 2, 0)
        x = inter_y.reshape(n_latent, h, w).permute(1, 2, 0)

        # print(x.shape)

        # plt.imshow(x[ ..., 6].detach().cpu().data.numpy())
        # plt.savefig('_data/latent_inter.png')
        # plt.show()
        # exit()
        torch.cuda.synchronize()
        # print(time.time() - start)

        # print(x.shape)

        # print(gamma.shape)

        # print(w0.shape)

        linear_func = SlangLinear.apply

        h, w, c = x.size()
        # Convert 4D to 2D for the MLP...
        # x = rearrange(x, 'h w c ->  h c')

        # w0_extend = torch.cat([w0, torch.zeros((16, 9), device=w0.device, dtype=w0.dtype)], dim=1)
        x_extend = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], 16 - n_latent), device=w0.device, dtype=x.dtype)],
                             dim=2)

        # check_tensor(x_extend)

        # print(x_extend.shape, w0.shape, b0.shape)
        x1 = linear_func(x_extend, w0, b0)
        x1 = F.relu(x1)

        x2 = linear_func(x1, w1, b1)
        x2 = F.relu(x2)
        x3 = linear_func(x2, w2, b2)
        x3 = F.relu(x3)

        mu = x3.mean(dim=(0, 1), keepdim=False).unsqueeze(0)
        sigma = (x3.std(dim=(0, 1), keepdim=False) + 1e-8).unsqueeze(0)
        w3_t = (gamma / sigma).permute(1, 0) * w3
        t = (beta - gamma * mu / sigma)
        b3_t = (t @ w3 + b3).squeeze(0)

        x_hat = linear_func(x3, w3_t, b3_t)[..., :3]

        x_hat = torch.clamp(x_hat, min=0., max=1.)
        # plt.imshow(x_hat.detach().cpu().data.numpy())
        # plt.savefig('_data/recon.png')
        # plt.show()
        # exit()

        # check_tensor(x_hat)

        # plt.imshow(x_hat.detach().cpu().data.numpy())
        # plt.show()
        # exit()

        # print(b0)

        bs = pic_codec.encode_bitstream(
            h, w, n_latent, ret['total_symbols'], ret['total_blocks'],
            ret['compact_tensor_dec_table'].detach().cpu(),
            ret['tensor_compressed'].detach().cpu(),
            w0.detach().cpu(),
            b0.detach().cpu(),
            w1.detach().cpu(),
            b1.detach().cpu(),
            w2.detach().cpu(),
            b2.detach().cpu(),
            w3_t[:, :3].detach().cpu(),
            b3_t[:3].detach().cpu(),
        )
        bpp = len(bs) * 8 / (h * w)

        if debug:
            return x_hat, bpp, bs, x, y, cloned_merged_blocks
        else:
            return x_hat, bpp, bs

    def encode_finetuned(self, model_output_pack):
        (
            w0, b0,
            w1, b1,
            w2, b2,
            w3, b3,
            latents,

        ) = model_output_pack

        # print(b3)
        n_latent = len(latents)
        ret = self.encode_latent(latents)

        _, _, h, w = latents[0].shape

        bs = pic_codec.encode_bitstream(
            h, w, n_latent, ret['total_symbols'], ret['total_blocks'],
            ret['compact_tensor_dec_table'].detach().cpu(),
            ret['tensor_compressed'].detach().cpu(),
            w0.detach().cpu(),
            b0.detach().cpu(),
            w1.detach().cpu(),
            b1.detach().cpu(),
            w2.detach().cpu(),
            b2.detach().cpu(),
            w3[:, :3].detach().cpu(),
            b3[:3].detach().cpu(),
        )
        bpp = len(bs) * 8 / (h * w)

        return bpp, bs

    @torch.no_grad()
    def decode(self, bs):
        # total_blocks = 8192

        # start = time.time()
        self.local_pic_codec.decode_bitstream(bs)
        self.local_pic_codec.reconstruct_feature(MAX_BOUND + BOUND_OFFSET, BOUND_OFFSET, 1 / 16.0)
        x_hat2 = self.local_pic_codec.fused_mlp_render()

        # plt.imshow(x_hat2.detach().cpu().data.numpy())
        # plt.show()
        # exit()
        return x_hat2

    def forward(self, pack, bottleneck):

        x_hat_in_encode, bpp, bs, x_base, y_base, merged_blocks = self.encode(pack, debug=True)
        (
            gamma, beta,
            w0, b0,
            w1, b1,
            w2, b2,
            w3, b3,
            latents,

        ) = pack

        n_latent = len(latents)

        # x_hat_in_decode, x_rec = self.decode(bs)

        shapes = [l.shape for l in latents]

        # print(shapes)

        padded_latent = [
            pad_for_patch_embedding(l, patch_size=8) for l in latents
        ]
        n_patchs = [(l.shape[-2] // self.patch_size, l.shape[-1] // self.patch_size) for l in padded_latent]

        reshaped_latents = [
            rearrange(l, 'b c (nh p1) (nw p2) -> b (c nh nw) p1 p2', p1=self.patch_size, p2=self.patch_size) for l in
            padded_latent
        ]

        dct_latents = [
            dct_2d_v2(l, norm='ortho') for l in reshaped_latents
        ]

        scaled_dct_latents = [
            dct_latents[i] * self.q_scale for i in range(n_latent)
        ]

        rearranged_latents = [
            rearrange(l, 'b n p1 p2 -> b (p1 p2) n') for l in scaled_dct_latents
        ]

        # latents_and_likelihoods = [
        #     bottleneck(latent, training=False)[0] for latent in rearranged_latents
        # ]
        #
        latents_and_likelihoods = [
            torch.round(latent) for latent in rearranged_latents
        ]

        # check_tensor(latents_and_likelihoods[0])

        restored_latent_dcts = [
            rearrange(p, 'b (p1 p2) n -> b n p1 p2', p1=self.patch_size, p2=self.patch_size) for p in
            latents_and_likelihoods
        ]

        unscaled_dct_latents = [
            restored_latent_dcts[i] / self.q_scale for i in range(n_latent)
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
            ) for i in range(n_latent)
        ]

        # for l in recovered_latents:
        #     print(l.shape)

        numels = [l.numel() for l in recovered_latents]

        # print(numels)

        # base_latent = y_base[numels[0]:numels[0]+numels[1]].reshape(recovered_latents[1].shape)

        base_latent = y_base[numels[0]:numels[0] + numels[1]].reshape(recovered_latents[1].shape)

        # print(merged_blocks.shape)
        start_block = numels[0] // 64 - 1
        # print(merged_blocks[start_block + 1, ...])
        # print(idct_latents[1][0][0])

        # print('base_latent', base_latent.shape)

        ref_latent = recovered_latents[1].squeeze(0).squeeze(0)
        base_latent = base_latent.squeeze(0).squeeze(0)

        # print('ref')
        # for i in range(ref_latent.shape[1]):
        #     print(ref_latent[0, i])

        # print('base')
        # for i in range(base_latent.shape[1]):
        #     print(base_latent[0, i])

        # print(ref_latent[0, -16:])
        # print(base_latent[0, -16:])
        # diff = ( ref_latent - base_latent)
        # plt.imshow(diff.detach().cpu().data.numpy())
        # plt.savefig('_data/latent_diff.png')
        # plt.show()

        # for i in range(0, n_latent):
        #     print(recovered_latents[i].shape)
        # results.append(F.interpolate(pyramid_latent[i], (h, w), mode='bilinear', align_corners=True))

        # print(recovered_latents[-1])

        cropped_padded_latents = [
            crop_from_padded(recovered_latents[i], shapes[i][-2:], patch_size=self.patch_size) for i in
            range(n_latent)
        ]
        # check_tensor(cropped_padded_latents[0])
        # torch.cuda.synchronize()
        # print('idct', time.time() - start_time)

        pyramid_latent = cropped_padded_latents
        b, c, h, w = pyramid_latent[0].shape
        # print(pyramid_latent[0][0,0,0,0], pyramid_latent[0][0,0,-1,-1])
        results = [pyramid_latent[0]]
        for i in range(1, len(pyramid_latent)):
            # print(pyramid_latent[i][0,0,0,0], pyramid_latent[i][0,0,-1,-1])
            # print(pyramid_latent[i].shape)
            results.append(F.interpolate(pyramid_latent[i], (h, w), mode='bilinear', align_corners=True))
        result = torch.cat(results, dim=1)

        x = result.squeeze(0).permute(2, 1, 0)

        # check_tensor(x)

        x_base = x_base.permute(1, 0, 2)
        # print(x.shape, x_base.shape)
        # for i in range(7):
        #     check_tensor(x[... , i])
        #     check_tensor(x_base[..., i])
        # print(i, (x[... , i] - x_base[..., i]).abs().max() / x[i].abs().max())

        # print(x[:8, -8:, 6])
        # print(x_base[:8, -8:, 6])

        cum_el = 0
        for i in range(n_latent):
            check_tensor(recovered_latents[i])
            check_tensor(y_base[cum_el:cum_el + recovered_latents[i].numel()])

            # print(i, (x[... , i] - x_base[..., i]).abs().max() / x[i].abs().max())
            cum_el += recovered_latents[i].numel()

        # print()
        # print(x[..., -1])
        # print(x_base[..., -1])

        # print(pyramid_latent[-1])

        # last_pyramid_y = y_base[-pyramid_latent[-1].numel():]

        # last_pyramid_y = last_pyramid_y.reshape(pyramid_latent[-1].shape)

        # print(last_pyramid_y)

        # print(merged_blocks.shape)
        # start_block = numels[0] // 64 -1
        # print(merged_blocks[0, ...])

        # print(merged_blocks.contiguous().reshape(-1)[numels[0]:numels[0]+16])
        # print(y_base)
        # print(x[-8:, -8:, 0])
        # print(x_base[-8:, -8:, 0])

        # print(x[:8, :8, 0])
        # print(x_base[:8, :8, 0])
        h, w, c = x.size()
        # Convert 4D to 2D for the MLP...
        # x = rearrange(x, 'h w c ->  h c')

        # print(x.shape, x_rec.shape)
        # print('rec', ((x - x_rec.permute(2, 1, 0))**2).mean())

        # w0_extend = torch.cat([w0, torch.zeros((16, 9), device=w0.device, dtype=w0.dtype)], dim=1)
        x_extend = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], 16 - n_latent), device=w0.device, dtype=x.dtype)],
                             dim=2)
        # check_tensor(x_extend)
        # x_extend[..., 0] = x_base[..., 0]

        # check_tensor(x_extend)

        linear_func = SlangLinear.apply

        # print(x_extend.shape, w0.shape, b0.shape)
        x1 = linear_func(x_extend, w0, b0)
        x1 = F.relu(x1)

        x2 = linear_func(x1, w1, b1)
        x2 = F.relu(x2)
        x3 = linear_func(x2, w2, b2)
        x3 = F.relu(x3)

        # check_tensor(x3)

        mu = x3.mean(dim=(0, 1), keepdim=False).unsqueeze(0)
        sigma = (x3.std(dim=(0, 1), keepdim=False) + 1e-8).unsqueeze(0)
        w3_t = (gamma / sigma).permute(1, 0) * w3
        t = (beta - gamma * mu / sigma)
        b3_t = (t @ w3 + b3).squeeze(0)

        # check_tensor(mu)
        # check_tensor(sigma)
        # check_tensor(w3_t)
        # check_tensor(b3_t)

        x4 = linear_func(x3, w3_t, b3_t)
        # check_tensor(x4)

        # check_tensor(x4)

        x_hat = torch.clamp(x4[..., :3], min=0., max=1.)

        # print('in forward for-enc', ((x_hat - x_hat_in_encode.permute(1, 0, 2))**2).mean())
        # print('in forward for-dec', ((x_hat - x_hat_in_decode.permute(1, 0, 2))**2).mean())
        # print(x_hat.shape)
        return x_hat.permute(1, 0, 2), bs
