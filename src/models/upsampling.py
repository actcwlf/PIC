import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List



class BilinearUpSample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pyramid_latent: List[Tensor]) -> Tensor:
        """
        """
        b, c, h, w = pyramid_latent[0].shape
        results = [pyramid_latent[0]]
        for i in range(1, len(pyramid_latent)):
            results.append(F.interpolate(pyramid_latent[i], (h, w), mode='bilinear', align_corners=True))
        result = torch.cat(results, dim=1)
        return result
