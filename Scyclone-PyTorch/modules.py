import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    k5s1c256 -> LReLU(a=0.01) -> k5s1c256
    """

    def __init__(self):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.Conv1d(256, 256, 5),
            nn.LeakyReLU(0.01),
            nn.Conv1d(256, 256, 5),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResidualSNBlock(nn.Module):
    """
    k5s1c256 -> LReLU(a=0.2) -> k5s1c256
    """

    def __init__(self):
        super(ResidualSNBlock, self).__init__()
        conv_block = [
            nn.utils.spectral_norm(nn.Conv1d(256, 256, 5)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(256, 256, 5)),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    """
    Scyclone Generator
    """

    def __init__(self, ngf, n_residual_blocks=7):
        super(Generator, self).__init__()

        model = [
            nn.Conv1d(128, 256, 1),
            nn.LeakyReLU(0.01),
        ]

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock()]

        model += [
            nn.Conv1d(256, 128, 1),
            nn.LeakyReLU(0.01),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    Scyclone Discriminator
    """

    def __init__(self, ndf):
        super(Discriminator, self).__init__()

        model = [
            nn.utils.spectral_norm(nn.Conv1d(128, 256, 1)),
            nn.LeakyReLU(0.2),
        ]

        # Residual blocks
        for _ in range(6):
            model += [ResidualBlock()]

        model += [
            nn.utils.spectral_norm(nn.Conv1d(256, 1, 1)),
            nn.LeakyReLU(0.2),
        ]
        model += [nn.AdaptiveAvgPool1d(1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
