import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import lpips



def psnr_func(img1, img2, data_range=1):
    if len(img1.shape) > 3:
        dim = (-1, -2, -3)
    else:
        dim = None
    err = torch.mean((img1 - img2) ** 2, dim=dim)
    # err = mean_squared_error(img1, img2)
    psnr_vec = 10 * torch.log10((data_range ** 2) / err)

    return psnr_vec.mean()


def cosine_similarity_matrix_func(x):
    b, n = x.shape
    x = x.unsqueeze(0).repeat(b, 1, 1)
    sim = F.cosine_similarity(x, x.permute(1, 0, 2), dim=-1)
    return sim


def msssim_fn(output, target, data_range=1):
    assert output.size(-2) >= 160
    msssim_val = ms_ssim(output.float().detach(), target.detach(), data_range=data_range, size_average=True)
    return msssim_val

lpips_fn = lpips.LPIPS().cuda()