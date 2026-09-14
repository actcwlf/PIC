import torch
import torch.nn as nn

from utils.inspector import check_tensor


class ResidualConvBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x2 = self.norm(x)
        x2 = self.conv1(x2)
        x2 = self.act(x2)
        x2 = self.conv2(x2)
        return x + x2


class ResidualEncoder(nn.Module):
    def __init__(self, n_latent, n_channel):
        super().__init__()
        self.n_latent = n_latent

        self.input_layers = nn.Sequential(
            nn.Conv2d(3, n_channel, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.Conv2d(n_channel, n_channel, kernel_size=3, padding=1),
            ResidualConvBlock(n_channel),
            ResidualConvBlock(n_channel)
        )

        self.projector0 = nn.Sequential(
            nn.BatchNorm2d(n_channel),
            nn.Conv2d(n_channel, 1, kernel_size=1)
        )

        self.down_sample_layers = nn.ModuleList()
        self.projectors = nn.ModuleList()

        for i in range(1, n_latent):
            self.down_sample_layers.append(nn.Sequential(
                ResidualConvBlock(n_channel),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                ResidualConvBlock(n_channel),
            ))
            self.projectors.append(nn.Sequential(
                nn.BatchNorm2d(n_channel),
                nn.Conv2d(n_channel, 1, kernel_size=1)
            ))

    def forward(self, x):
        # check_tensor(x)
        feat = self.input_layers(x)
        features = [feat]
        latent0 = self.projector0(feat)
        latents = [latent0]

        for i in range(self.n_latent - 1):
            feat = self.down_sample_layers[i](feat)
            features.append(feat)
            latent = self.projectors[i](feat)
            latents.append(latent)
        return features, latents


class ModulationEncoder(nn.Module):
    def __init__(self, n_channel, out_dim):
        super().__init__()
        self.input_layers = nn.Sequential(
            nn.Conv2d(3, n_channel, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.Conv2d(n_channel, n_channel, kernel_size=3, padding=1),
            ResidualConvBlock(n_channel),
            ResidualConvBlock(n_channel),
            nn.Conv2d(n_channel, out_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features: torch.Tensor = self.input_layers(x)
        gamma = features.std(dim=(-2, -1))
        beta = features.mean(dim=(-2, -1))
        return gamma, beta