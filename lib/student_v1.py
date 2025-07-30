import torch
import torch.nn as nn
from torch import Tensor

class ChannelNorm2D(nn.Module):
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
    

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class StudentEncoderBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Sequential(
                DepthwiseSeparableConv(3, 60, kernel_size=7, stride=1, padding=3),
                ChannelNorm2D(60),
                nn.ReLU()
            ),
            nn.Sequential(
                DepthwiseSeparableConv(60, 120, kernel_size=3, stride=2, padding=1),
                ChannelNorm2D(120),
                nn.ReLU()
            ),
            nn.Sequential(
                DepthwiseSeparableConv(120, 240, kernel_size=3, stride=2, padding=1),
                ChannelNorm2D(240),
                nn.ReLU()
            ),
            nn.Sequential(
                DepthwiseSeparableConv(240, 480, kernel_size=3, stride=2, padding=1),
                ChannelNorm2D(480),
                nn.ReLU()
            ),
            nn.Sequential(
                DepthwiseSeparableConv(480, 960, kernel_size=3, stride=2, padding=1),
                ChannelNorm2D(960),
                nn.ReLU()
            ),
            DepthwiseSeparableConv(960, 220, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.blocks(x)


class StudentEncoderPruned_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Sequential(
                DepthwiseSeparableConv(3, 120, kernel_size=7, stride=1, padding=3),
                nn.AvgPool2d(kernel_size=2),
                ChannelNorm2D(120),
                nn.ReLU()
            ),

            nn.Sequential(
                DepthwiseSeparableConv(120, 480, kernel_size=3, stride=2, padding=1),
                nn.AvgPool2d(kernel_size=2),
                ChannelNorm2D(480),
                nn.ReLU()
            ),
            
            DepthwiseSeparableConv(480, 220, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.blocks(x)


class StudentEncoderPruned_V2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = nn.Sequential(
                DepthwiseSeparableConv(3, 120, kernel_size=7, stride=1, padding=3),
                nn.AvgPool2d(kernel_size=2),
                ChannelNorm2D(120),
                nn.ReLU())
        
        self.block2 = nn.Sequential(
                DepthwiseSeparableConv(120, 480, kernel_size=3, stride=2, padding=1),
                nn.AvgPool2d(kernel_size=2),
                ChannelNorm2D(480),
                nn.ReLU())
        
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(480, 220, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.block3(x)


class StudentEncoderBase_V2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = nn.Sequential(
                DepthwiseSeparableConv(3, 60, kernel_size=7, stride=1, padding=3),
                ChannelNorm2D(60),
                nn.ReLU()
            )

        self.block2 = nn.Sequential(
                DepthwiseSeparableConv(60, 120, kernel_size=3, stride=2, padding=1),
                ChannelNorm2D(120),
                nn.ReLU()
            )

        self.block3 = nn.Sequential(
                DepthwiseSeparableConv(120, 240, kernel_size=3, stride=2, padding=1),
                ChannelNorm2D(240),
                nn.ReLU()
            )

        self.block4 = nn.Sequential(
                DepthwiseSeparableConv(240, 480, kernel_size=3, stride=2, padding=1),
                ChannelNorm2D(480),
                nn.ReLU()
            )
        
        self.block5 = nn.Sequential(
                DepthwiseSeparableConv(480, 960, kernel_size=3, stride=2, padding=1),
                ChannelNorm2D(960),
                nn.ReLU()
            )

        self.block6 = DepthwiseSeparableConv(960, 220, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.block6(x)
