import torch
from torch import  nn

import torch.nn.functional as F

from einops import rearrange
from utils.slang_module.linear import init_slang_module, SlangLinear




class ModulationSynthesis(nn.Module):
    def __init__(self,  in_ft):
        super().__init__()
        # self.inner_dim = 24
        # conv_in = in_ft
        # n_transform_layers = 2
        # transform_layers = []
        # for i in range(n_transform_layers):
        #     transform_layers.append(
        #         CustomConv(in_channels=conv_in, out_channels=conv_in, kernel_size=5, padding=2, groups=conv_in))
        #     transform_layers.append(
        #         CustomConv(in_channels=conv_in, out_channels=conv_in, kernel_size=1, padding=0, groups=1))
        #     transform_layers.append(nn.ReLU())
        #
        # transform_layers.append(
        #     CustomConv(in_channels=conv_in, out_channels=conv_in, kernel_size=5, padding=2, groups=conv_in))
        # transform_layers.append(
        #     CustomConv(in_channels=conv_in, out_channels=conv_in, kernel_size=1, padding=0, groups=1))
        # self.conv = nn.Sequential(*transform_layers)

        init_slang_module()
        self.in_ft = in_ft

        out_ft = 16
        self.weight0 = nn.Parameter(torch.randn((out_ft, out_ft), requires_grad=True) / out_ft ** 2)
        self.bias0 = nn.Parameter(torch.zeros((out_ft), requires_grad=True))
        self.weight1 = nn.Parameter(torch.randn((out_ft, out_ft), requires_grad=True) / out_ft ** 2)
        self.bias1 = nn.Parameter(torch.zeros((out_ft), requires_grad=True))
        self.weight2 = nn.Parameter(torch.randn((out_ft, out_ft), requires_grad=True) / out_ft ** 2)
        self.bias2 = nn.Parameter(torch.zeros((out_ft), requires_grad=True))
        self.weight3 = nn.Parameter(torch.randn((out_ft, out_ft), requires_grad=True) / out_ft ** 2)
        self.bias3 = nn.Parameter(torch.zeros((out_ft), requires_grad=True))

        self.linear_func = SlangLinear.apply


    def forward(self, x, gamma, beta):
        b, c, h, w = x.size()
        # Convert 4D to 2D for the MLP...
        x = rearrange(x, 'b c h w -> (b h) w c') # SlangLinear 要求第一维度必须是32的倍数，第二维度是16的倍数

        assert x.shape[0] % 32 == 0 and x.shape[1] % 16 == 0

        x = torch.cat([
            x, torch.zeros((x.shape[0], x.shape[1], 16 - self.in_ft), device=x.device, dtype=x.dtype)
        ], dim=-1)

        # check_tensor(x)

        x1 = self.linear_func(x, self.weight0, self.bias0)
        x1 = F.relu(x1)

        x2 = self.linear_func(x1, self.weight1, self.bias1)
        x2 = F.relu(x2)
        x3 = self.linear_func(x2, self.weight2, self.bias2)
        x3 = F.relu(x3)

        # check_tensor(x3)



        x3 = rearrange(x3, '(b h) w c -> b (h w) c', b=b, h=h, w=w)
        #
        # gamma = gamma.unsqueeze(1)
        # beta = beta.unsqueeze(1)
        # norm_x3 = (x3 - x3.mean(dim=1, keepdim=True)) / (x3.std(dim=1, keepdim=True) + 1e-8)
        # x3 = gamma * norm_x3 + beta

        x4_list = []
        mu_full = x3.mean(dim=1, keepdim=False) #.unsqueeze(0)
        sigma_full = (x3.std(dim=1, keepdim=False) + 1e-8)#.unsqueeze(0)
        gamma_full = gamma
        beta_full = beta
        for i in range(b):
            mu = mu_full[i:i+1]
            sigma = sigma_full[i:i+1]
            gamma = gamma_full[i:i+1]
            beta = beta_full[i:i+1]
            w3_t = (gamma / sigma).permute(1, 0) * self.weight3
            t = (beta - gamma * mu / sigma)
            b3_t = (t @ self.weight3 + self.bias3).squeeze(0)
        #
            x3_ins = rearrange(x3[i], '(h w) c -> h w c', h=h, w=w)
            # x3_ins = x3[i]

            # check_tensor(self.weight3)

            # check_tensor(w3_t)
            # check_tensor(b3_t)

            x4 = self.linear_func(x3_ins,  w3_t, b3_t)
            # check_tensor(x4)
            x4_list.append(x4)

        x4 = torch.cat(x4_list, dim=0)

        # check_tensor(x4)

        # Go back from 2D to 4D. We output 3 features (i.e. RGB)
        x = rearrange(x4[..., :3], '(b h) w c -> b c h w', c=3, h=h, w=w)

        return x


class DecoderSynthesis(nn.Module):
    def __init__(self, syn: ModulationSynthesis, latent_tensor, gamma, beta):
        super().__init__()

        init_slang_module()
        self.in_ft = syn.in_ft
        self.ref_syn = syn

        out_ft = 16
        self.weight0 = nn.Parameter(torch.randn((out_ft, out_ft), requires_grad=True) / out_ft ** 2)
        self.bias0 = nn.Parameter(torch.zeros((out_ft), requires_grad=True))
        self.weight1 = nn.Parameter(torch.randn((out_ft, out_ft), requires_grad=True) / out_ft ** 2)
        self.bias1 = nn.Parameter(torch.zeros((out_ft), requires_grad=True))
        self.weight2 = nn.Parameter(torch.randn((out_ft, out_ft), requires_grad=True) / out_ft ** 2)
        self.bias2 = nn.Parameter(torch.zeros((out_ft), requires_grad=True))
        self.weight3 = nn.Parameter(torch.randn((out_ft, out_ft), requires_grad=True) / out_ft ** 2)
        self.bias3 = nn.Parameter(torch.zeros((out_ft), requires_grad=True))

        self.linear_func = SlangLinear.apply

        self.init_weight_and_bias(latent_tensor, gamma, beta)
        self.ref_syn = None

    @torch.no_grad()
    def init_weight_and_bias(self, x, gamma, beta):
        # linear_func =

        # h, w, c = x.size()
        #     # Convert 4D to 2D for the MLP...
        # x = rearrange(x, 'c h w c ->  h w c')

        # x = x.permute(1, 2, 0)

        # w0_extend = torch.cat([w0, torch.zeros((16, 9), device=w0.device, dtype=w0.dtype)], dim=1)
        x_extend = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], 9), device=x.device, dtype=x.dtype)],dim=2)

        # check_tensor(x_extend)

        # print(x_extend.shape, w0.shape, b0.shape)
        x1 = self.linear_func(x_extend, self.ref_syn.weight0, self.ref_syn.bias0)
        x1 = F.relu(x1)

        x2 = self.linear_func(x1, self.ref_syn.weight1, self.ref_syn.bias1)
        x2 = F.relu(x2)
        x3 = self.linear_func(x2, self.ref_syn.weight2, self.ref_syn.bias2)
        x3 = F.relu(x3)


        mu = x3.mean(dim=(0, 1), keepdim=False).unsqueeze(0)
        sigma =  (x3.std(dim=(0, 1), keepdim=False) + 1e-8).unsqueeze(0)
        w3_t = (gamma/ sigma).permute(1, 0) * self.ref_syn.weight3
        t = (beta - gamma * mu / sigma)
        b3_t = (t @ self.ref_syn.weight3 + self.ref_syn.bias3).squeeze(0)

        x_hat = self.linear_func(x3, w3_t, b3_t)[...,:3]

        # check_tensor(x_hat)

        # plt.imshow(x_hat.detach().cpu().data.numpy())
        # plt.show()
        # exit()


        self.weight0 = nn.Parameter(self.ref_syn.weight0, requires_grad=True)
        self.weight1 = nn.Parameter(self.ref_syn.weight1, requires_grad=True)
        self.weight2 = nn.Parameter(self.ref_syn.weight2, requires_grad=True)
        self.weight3 = nn.Parameter(w3_t, requires_grad=True)

        self.bias0 = nn.Parameter(self.ref_syn.bias0, requires_grad=True)
        self.bias1 = nn.Parameter(self.ref_syn.bias1, requires_grad=True)
        self.bias2 = nn.Parameter(self.ref_syn.bias2, requires_grad=True)
        self.bias3 = nn.Parameter(b3_t, requires_grad=True)


    def forward(self, x):
        b, c, h, w = x.size()
        # Convert 4D to 2D for the MLP...
        x = rearrange(x, 'b c h w -> (b h) w c')  # SlangLinear 要求第一维度必须是32的倍数，第二维度是16的倍数

        assert x.shape[0] % 32 == 0 and x.shape[1] % 16 == 0

        x = torch.cat([
            x, torch.zeros((x.shape[0], x.shape[1], 16 - self.in_ft), device=x.device, dtype=x.dtype)
        ], dim=-1)

        x1 = self.linear_func(x, self.weight0, self.bias0)
        x1 = F.relu(x1)
        x2 = self.linear_func(x1, self.weight1, self.bias1)
        x2 = F.relu(x2)
        x3 = self.linear_func(x2, self.weight2, self.bias2)
        x3 = F.relu(x3)
        x4 = self.linear_func(x3, self.weight3, self.bias3)

        # Go back from 2D to 4D. We output 3 features (i.e. RGB)
        x = rearrange(x4[..., :3], '(b h) w c -> b c h w', c=3, h=h, w=w)

        # plt.imshow(x.squeeze(0).permute(1, 2, 0).detach().cpu().data.numpy())
        # plt.show()
        # exit()
        x = torch.clamp(x, min=0, max=1)

        return x
