from json import encoder
from typing import List
import torch
import torch.nn as nn
from torch import Tensor
import os

class ChannelNorm2D(nn.Module):
    """
    Channel normalization layer.

    This implements the channel normalization layer as described in the
    following paper:

    High-Fidelity Generative Image Compression
    F. Mentzer, G. Toderici, M. Tschannen, E. Agustsson

    Using this layer provides more stability to model outputs when there is a
    shift in image resolutions between the training set and the test set.

    Args:
        input_channels: Number of channels to normalize.
        epsilon: Divide-by-0 protection parameter.
        affine: Whether to include affine parameters for the noramlized output.
    """

    def __init__(self, input_channels: int, epsilon: float = 1e-3, affine: bool = True):
        super().__init__()

        if input_channels <= 1:
            raise ValueError(
                "ChannelNorm only valid for channel counts greater than 1."
            )

        self.epsilon = epsilon
        self.affine = affine

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.var(x, dim=1, keepdim=True)

        x_normed = (x - mean) * torch.rsqrt(variance + self.epsilon)

        if self.affine is True:
            x_normed = self.gamma * x_normed + self.beta

        return x_normed

def _channel_norm_2d(input_channels, affine=True):
    return ChannelNorm2D(
        input_channels,
        affine=affine,
    )

class HiFiCEncoder(torch.nn.Module):
    """
    High-Fidelity Generative Image Compression (HiFiC) encoder.

    Args:
        input_dimensions: shape of the input tensor
        latent_features: number of bottleneck features
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_features: int = 220,
    ):
        super().__init__()

        blocks: List[nn.Module] = []
        for index, out_channels in enumerate((60, 120, 240, 480, 960)):
            if index == 0:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )

            in_channels = out_channels
            blocks += [block]

        blocks += [nn.Conv2d(out_channels, latent_features, kernel_size=3, padding=1)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)

class _ResidualBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding=padding),
            _channel_norm_2d(channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, stride, padding=padding),
            _channel_norm_2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.sequence(x)

class HiFiCGenerator(torch.nn.Module):
    """
    High-Fidelity Generative Image Compression (HiFiC) generator.

    Args:
        input_dimensions: shape of the input tensor
        batch_size: number of images per batch
        latent_features: number of bottleneck features
        n_residual_blocks: number of residual blocks
    """

    def __init__(
        self,
        image_channels: int = 3,
        latent_features: int = 220,
        n_residual_blocks: int = 9,
    ):
        super(HiFiCGenerator, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        filters = [960, 480, 240, 120, 60]

        self.block_0 = nn.Sequential(
            _channel_norm_2d(latent_features, affine=True),
            nn.Conv2d(latent_features, filters[0], kernel_size=3, padding=1),
            _channel_norm_2d(filters[0], affine=True),
        )

        resid_blocks = []
        for _ in range(self.n_residual_blocks):
            resid_blocks.append(_ResidualBlock((filters[0])))

        self.resid_blocks = nn.Sequential(*resid_blocks)

        blocks: List[nn.Module] = []
        in_channels = filters[0]
        for out_channels in filters[1:]:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        output_padding=1,
                        stride=2,
                        padding=1,
                    ),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )
            )

            in_channels = out_channels

        blocks.append(
            nn.Conv2d(
                filters[-1], out_channels=image_channels, kernel_size=7, padding=3
            )
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block_0(x)
        x = x + self.resid_blocks(x)

        return self.blocks(x)
    

def get_teacher_encoder(quality=3, path=r"D:\IOT_project\models\teacher"):
    encoder = HiFiCEncoder()
    state = torch.load(os.path.join(path, f"Encoder_Q{quality}_Weights.pt"))
    encoder.load_state_dict(state)
    return encoder


def get_teacher_decoder(quality=3, path=r"D:\IOT_project\models\teacher"):
    decoder = HiFiCGenerator()
    state = torch.load(os.path.join(path, f"Decoder_Q{quality}_Weights.pt"))
    decoder.load_state_dict(state)
    return decoder
