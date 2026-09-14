import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from einops.layers.torch import Rearrange
from torch import Tensor


def pad_for_patch_embedding(x, patch_size=30):
    b, c, h, w = x.size()
    n_h = math.ceil(h / patch_size)
    padded_h = n_h * patch_size

    n_w = math.ceil(w / patch_size)
    padded_w = n_w * patch_size
    padding = (
        (padded_w - w) // 2,
        padded_w - w - (padded_w - w) // 2,
        (padded_h - h) // 2,
        padded_h - h - (padded_h - h) // 2
    )
    # print(padding)
    padded_x = F.pad(x, padding)

    return padded_x



def crop_from_padded(x, target_size, patch_size):
    h, w = target_size
    n_h = math.ceil(h / patch_size)
    padded_h = n_h * patch_size

    n_w = math.ceil(w / patch_size)
    padded_w = n_w * patch_size
    padding = (
        (padded_w - w) // 2,
        - (padded_w - w - (padded_w - w) // 2),
        (padded_h - h) // 2,
        - (padded_h - h - (padded_h - h) // 2)
    )

    # print(padding)

    def drop_zero(v):
        if v == 0:
            return None
        return v

    padding = [drop_zero(v) for v in padding]

    # return x[..., padding[2]:-padding[3], padding[0]:-padding[1]]
    return x[..., slice(padding[2], padding[3]), slice(padding[0], padding[1])]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(dim, inner_dim, bias=False)
        self.k_linear = nn.Linear(dim, inner_dim, bias=False)
        self.v_linear = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):

        q = self.q_linear(self.norm(q))
        k = self.q_linear(self.norm(k))
        v = self.q_linear(self.norm(v))


        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class CrossTransformerBlock(nn.Module):
    def __init__(self, dim,  heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.cross_attention_layer = CrossAttention(dim, heads=heads, dim_head = dim_head, dropout = dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)

    def forward(self, q, k, v):
        out = self.cross_attention_layer(q, k, v)
        out = self.ffn(out) + out

        return out


class RIFEBlock(nn.Module):
    """
    resolution independent feature extractor
    """
    def __init__(self, in_channels, dim=256, patch_size=30):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.self_attention_layer = Transformer(
            dim=dim, depth=1, heads=8, dim_head=dim, mlp_dim=dim * 2
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)

    def forward(self, feat):
        b, c, h, w = feat.shape
        padded_x = pad_for_patch_embedding(feat, self.patch_size)

        tokens = self.to_patch_embedding(padded_x)

        tokens = torch.cat([self.cls_token.repeat(b, 1, 1), tokens], dim=1)

        tokens = self.self_attention_layer(tokens)

        latent = tokens[:, 0, :]

        return latent


class MNIFGenerator(nn.Module):
    def __init__(self, in_channels, n_latent, dim=128, patch_size=30):
        super().__init__()
        self.n_latent = n_latent
        self.dim = dim
        self.blocks = nn.ModuleList(
            [RIFEBlock(in_channels, dim, patch_size) for _ in range(n_latent)]
        )

    def forward(self, feat_list):

        latents = []

        for feat, layer in zip(feat_list, self.blocks):
            latent = layer(feat)
            latents.append(latent)

        latent = torch.cat(latents, dim=1)
        return latent

    @property
    def latent_dim(self):
        return self.n_latent * self.dim

