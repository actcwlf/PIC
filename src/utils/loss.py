import torch

from utils.metrics import psnr_func


def l1_loss_func(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss_func(network_output, gt, dim=None):
    return torch.pow((network_output - gt), 2.0).mean(dim)



def rd_loss(out, gt, lmbda):
    l2_loss = l2_loss_func(out['x_hat'], gt, dim=(1, 2, 3))
    train_psnr = psnr_func(out['x_hat'], gt)
    batch_size = gt.shape[0]
    n_pixels = gt.shape[-2] * gt.shape[-1]
    rate_bpp = (out['rate_y'].sum()) / n_pixels / batch_size
    # sparsity_reg = -out['reg'].sum() / n_pixels / batch_size
    total_loss = l2_loss.mean() + lmbda * rate_bpp
    return total_loss, l2_loss, train_psnr, rate_bpp, 0