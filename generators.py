"""
Courtsey of: https://github.com/Muzammal-Naseer/Cross-domain-perturbations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from gaussian_smoothing import get_gaussian_kernel
from torch.nn import init
from einops import rearrange

from utils import get_hc

###########################
# Generator: Resnet
###########################

# To control feature map in generator
ngf = 64

# generate adv img
class GeneratorResnet(nn.Module):
    def __init__(self, inception=False):
        """
        :param inception: if True crop layer will be added to go from 3x300x300 to 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be added otherwise only 2.
        """
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):
        x = self.block1(input)  # 64x224x224
        x = self.block2(x)      # 128x112x112
        x = self.block3(x)      # 256x56x56
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.upsampl1(x)    # 128x112x112
        x = self.upsampl2(x)    # 64x224x224
        x = self.blockf(x)      # 3x224x224
        if self.inception:
            x = self.crop(x)

        return (torch.tanh(x) + 1) / 2  # Output range [0 1]


# generate adv noise (for TTAA)
# class GeneratorResnet_Noise(nn.Module):
#     def __init__(self, inception=False):
#         """
#         :param inception: if True crop layer will be added to go from 3x300x300 to 3x299x299.
#         :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be added otherwise only 2.
#         """
#         super(GeneratorResnet_Noise, self).__init__()
#         self.inception = inception
#         # Input_size = 3, n, n
#         self.block1 = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True)
#         )
#
#         # Input size = 3, n, n
#         self.block2 = nn.Sequential(
#             nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True)
#         )
#
#         # Input size = 3, n/2, n/2
#         self.block3 = nn.Sequential(
#             nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True)
#         )
#
#         # Input size = 3, n/4, n/4
#         # Residual Blocks: 6
#         self.resblock1 = ResidualBlock(ngf * 4)
#         self.resblock2 = ResidualBlock(ngf * 4)
#         self.resblock3 = ResidualBlock(ngf * 4)
#         self.resblock4 = ResidualBlock(ngf * 4)
#         self.resblock5 = ResidualBlock(ngf * 4)
#         self.resblock6 = ResidualBlock(ngf * 4)
#
#         # Input size = 3, n/4, n/4
#         self.upsampl1 = nn.Sequential(
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True)
#         )
#
#         # Input size = 3, n/2, n/2
#         self.upsampl2 = nn.Sequential(
#             nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True)
#         )
#
#         # Input size = 3, n, n
#         self.blockf = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
#         )
#
#         self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
#
#     def forward(self, input):
#         x = self.block1(input)  # 64x224x224
#         x = self.block2(x)      # 128x112x112
#         x = self.block3(x)      # 256x56x56
#         x = self.resblock1(x)
#         x = self.resblock2(x)
#         x = self.resblock3(x)
#         x = self.resblock4(x)
#         x = self.resblock5(x)
#         x = self.resblock6(x)
#         x = self.upsampl1(x)    # 128x112x112
#         x = self.upsampl2(x)    # 64x224x224
#         x = self.blockf(x)      # 3x224x224
#         if self.inception:
#             x = self.crop(x)
#
#         return (torch.tanh(x) + 1) / 2  # Output range [0 1]
#         # return torch.tanh(x)  # Output range [-1 1]
#--------------------------

# for Unet
# class GeneratorUnet_HF(nn.Module):
#     def __init__(self, inception=False):
#         """
#         :param inception: if True crop layer will be added to go from 3x300x300 to 3x299x299.
#         :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be added otherwise only 2.
#         """
#         super(GeneratorUnet_HF, self).__init__()
#         self.inception = inception
#         # Input_size = 3, n, n
#         self.block1 = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True)
#         )
#
#         # Input size = 3, n, n
#         self.block2 = nn.Sequential(
#             nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True)
#         )
#
#         # Input size = 3, n/2, n/2
#         self.block3 = nn.Sequential(
#             nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True)
#         )
#
#         # Input size = 3, n/4, n/4
#         # Residual Blocks: 6
#         self.resblock1 = ResidualBlock(ngf * 4)
#         self.resblock2 = ResidualBlock(ngf * 4)
#         self.resblock3 = ResidualBlock(ngf * 4)
#         self.resblock4 = ResidualBlock(ngf * 4)
#         self.resblock5 = ResidualBlock(ngf * 4)
#         self.resblock6 = ResidualBlock(ngf * 4)
#
#         # Input size = 3, n/4, n/4
#         self.upsampl1 = nn.Sequential(
#             nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True)
#         )
#
#         # Input size = 3, n/2, n/2
#         self.upsampl2 = nn.Sequential(
#             nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True)
#         )
#
#         # Input size = 3, n, n
#         self.blockf = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(ngf * 2, 3, kernel_size=7, padding=0)
#         )
#
#         self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
#
#     def forward(self, input, input_match=None, alpha=0.2):
#         x = self.block1(input)  # 64x224x224
#         x = self.block2(x)      # 128x112x112
#         x = self.block3(x)      # 256x56x56
#
#         x = self.resblock1(x)
#         x = self.resblock2(x)
#         x = self.resblock3(x)
#         x = self.resblock4(x)
#         x = self.resblock5(x)
#         x = self.resblock6(x)   # 256x56x56
#
#         x = self.upsampl1(x)    # 128x112x112
#         x = self.upsampl2(x)    # 64x224x224
#         x = self.blockf(x)      # 3x224x224
#         if self.inception:
#             x = self.crop(x)
#
#         return (torch.tanh(x) + 1) / 2  # Output range [0 1]
#--------------------------
# class double_conv2d_bn(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
#         super(double_conv2d_bn, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels,
#                                kernel_size=kernel_size,
#                                stride=strides, padding=padding, bias=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels,
#                                kernel_size=kernel_size,
#                                stride=strides, padding=padding, bias=True)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         return out
#
#
# class deconv2d_bn(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
#         super(deconv2d_bn, self).__init__()
#         self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
#                                         kernel_size=kernel_size,
#                                         stride=strides, bias=True)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         return out
#
#
# class GeneratorUnet_Fre(nn.Module):
#     def __init__(self, inception=False):
#         super(GeneratorUnet_Fre, self).__init__()
#         self.inception = inception
#
#         self.layer1_conv = double_conv2d_bn(3, 8)
#         self.layer2_conv = double_conv2d_bn(8, 16)
#         self.layer3_conv = double_conv2d_bn(16, 32)
#         self.layer4_conv = double_conv2d_bn(32, 64)
#         self.layer5_conv = double_conv2d_bn(64, 128)
#         self.layer6_conv = double_conv2d_bn(128, 64)
#         self.layer7_conv = double_conv2d_bn(64, 32)
#         self.layer8_conv = double_conv2d_bn(32, 16)
#         self.layer9_conv = double_conv2d_bn(16, 8)
#         self.layer10_conv = nn.Conv2d(8, 3, kernel_size=3,
#                                       stride=1, padding=1, bias=True)
#
#         self.deconv1 = deconv2d_bn(128, 64)
#         self.deconv2 = deconv2d_bn(64, 32)
#         self.deconv3 = deconv2d_bn(32, 16)
#         self.deconv4 = deconv2d_bn(16, 8)
#
#         self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, x_match):
#         conv1 = self.layer1_conv(x)
#         pool1 = F.max_pool2d(conv1, 2)
#         conv1_t = self.layer1_conv(x_match).detach()
#         conv1_t_hc = get_hc(conv1_t).detach()
#
#         conv2 = self.layer2_conv(pool1)
#         pool2 = F.max_pool2d(conv2, 2)
#         conv2_t = self.layer2_conv(F.max_pool2d(conv1_t, 2)).detach()
#         conv2_t_hc = get_hc(conv2_t).detach()
#
#         conv3 = self.layer3_conv(pool2)
#         pool3 = F.max_pool2d(conv3, 2)
#         conv3_t = self.layer3_conv(F.max_pool2d(conv2_t, 2)).detach()
#         conv3_t_hc = get_hc(conv3_t).detach()
#
#         conv4 = self.layer4_conv(pool3)
#         pool4 = F.max_pool2d(conv4, 2)
#         conv4_t = self.layer4_conv(F.max_pool2d(conv3_t, 2)).detach()
#         conv4_t_hc = get_hc(conv4_t).detach()
#
#         conv5 = self.layer5_conv(pool4)
#
#         convt1 = self.deconv1(conv5)
#         concat1 = torch.cat([convt1, conv4_t_hc], dim=1)
#         conv6 = self.layer6_conv(concat1)
#
#         convt2 = self.deconv2(conv6)
#         concat2 = torch.cat([convt2, conv3_t_hc], dim=1)
#         conv7 = self.layer7_conv(concat2)
#
#         convt3 = self.deconv3(conv7)
#         concat3 = torch.cat([convt3, conv2_t_hc], dim=1)
#         conv8 = self.layer8_conv(concat3)
#
#         convt4 = self.deconv4(conv8)
#         concat4 = torch.cat([convt4, conv1_t_hc], dim=1)
#         conv9 = self.layer9_conv(concat4)
#         outp = self.layer10_conv(conv9)
#         if self.inception:
#             outp = self.crop(outp)
#
#         # outp = self.sigmoid(outp)
#         outp = (torch.tanh(outp) + 1) / 2
#         return outp
#--------------------------


# for My method
class GeneratorResnet_R(nn.Module):
    def __init__(self, inception=False):
        """
        :param inception: if True crop layer will be added to go from 3x300x300 to 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be added otherwise only 2.
        """
        super(GeneratorResnet_R, self).__init__()
        self.inception = inception
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.rewe_net = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):
        x = self.block1(input)  # 64x224x224
        x = self.block2(x)      # 128x112x112
        x = self.block3(x)      # 256x56x56
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.upsampl1(x)    # 128x112x112
        x = self.upsampl2(x)    # 64x224x224
        x = self.blockf(x)      # 3x224x224
        if self.inception:
            x = self.crop(x)

        x = (torch.tanh(x) + 1) / 2  # noise:Output range [0 1]
        x_hc, _ = get_hc(x)
        x_lc = x - x_hc

        # weight = (torch.tanh(self.rewe_net(x_hc)) + 1) / 2
        weight = self.rewe_net(x_hc)
        r_x_hc = weight * x_hc
        x = x_lc + r_x_hc

        return x,  r_x_hc


class GeneratorResnet_W(nn.Module):
    def __init__(self, inception=False):
        """
        :param inception: if True crop layer will be added to go from 3x300x300 to 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be added otherwise only 2.
        """
        super(GeneratorResnet_W, self).__init__()
        self.inception = inception
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):
        x = self.block1(input)  # 64x224x224
        x = self.block2(x)      # 128x112x112
        x = self.block3(x)      # 256x56x56
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.upsampl1(x)    # 128x112x112
        x = self.upsampl2(x)    # 64x224x224
        x = self.blockf(x)      # 3x224x224
        if self.inception:
            x = self.crop(x)

        return (torch.tanh(x) + 1) / 2  # Output range [0 1]


# for C-GSP
def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


class ConGeneratorResnet(nn.Module):
    def __init__(self, inception=False, nz=16, layer=1, loc=[1, 1, 1], data_dim='high'):
        """
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        """
        super(ConGeneratorResnet, self).__init__()
        self.inception = inception
        self.data_dim = data_dim
        self.layer = layer
        self.snlinear = snlinear(in_features=1000, out_features=nz, bias=False)
        if self.layer > 1:
            self.snlinear2 = snlinear(in_features=nz, out_features=nz, bias=False)
        if self.layer > 2:
            self.snlinear3 = snlinear(in_features=nz, out_features=nz, bias=False)
        self.loc = loc
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3 + nz * self.loc[0], ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf + nz * self.loc[1], ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2 + nz * self.loc[2], ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
            # self.resblock7 = ResidualBlock(ngf*4)
            # self.resblock8 = ResidualBlock(ngf*4)
            # self.resblock9 = ResidualBlock(ngf*4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
        self.alf_layer = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1)

    def forward(self, input, z_one_hot, eps=0, gap=False):
        z_cond = self.snlinear(z_one_hot)
        if self.layer > 1:
            z_cond = self.snlinear2(z_cond)
        if self.layer > 2:
            z_cond = self.snlinear3(z_cond)
        # loc 0
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
            z_cond.size(0), z_cond.size(1), input.size(2), input.size(3))
        assert self.loc[0] == 1
        x = self.block1(torch.cat((input, z_img), dim=1))
        # loc 1
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
            z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        if self.loc[1]:
            x = self.block2(torch.cat((x, z_img), dim=1))
        else:
            x = self.block2(x)
        # loc 2
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
            z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        if self.loc[2]:
            x = self.block3(torch.cat((x, z_img), dim=1))
        else:
            x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.data_dim == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
            # x = self.resblock7(x)
            # x = self.resblock8(x)
            # x = self.resblock9(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)

        x = torch.tanh(x)
        x = self.alf_layer(x)

        return x * eps


class ConGeneratorResnet_adv(nn.Module):
    def __init__(self, inception=False, nz=16, layer=1, loc=[1, 1, 1], data_dim='high'):
        """
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        """
        super(ConGeneratorResnet_adv, self).__init__()
        self.inception = inception
        self.data_dim = data_dim
        self.layer = layer
        self.snlinear = snlinear(in_features=1000, out_features=nz, bias=False)
        if self.layer > 1:
            self.snlinear2 = snlinear(in_features=nz, out_features=nz, bias=False)
        if self.layer > 2:
            self.snlinear3 = snlinear(in_features=nz, out_features=nz, bias=False)
        self.loc = loc
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3 + nz * self.loc[0], ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf + nz * self.loc[1], ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2 + nz * self.loc[2], ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
            # self.resblock7 = ResidualBlock(ngf*4)
            # self.resblock8 = ResidualBlock(ngf*4)
            # self.resblock9 = ResidualBlock(ngf*4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input, z_one_hot, eps=0, gap=False):
        z_cond = self.snlinear(z_one_hot)
        if self.layer > 1:
            z_cond = self.snlinear2(z_cond)
        if self.layer > 2:
            z_cond = self.snlinear3(z_cond)
        # loc 0
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
            z_cond.size(0), z_cond.size(1), input.size(2), input.size(3))
        assert self.loc[0] == 1
        x = self.block1(torch.cat((input, z_img), dim=1))
        # loc 1
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
            z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        if self.loc[1]:
            x = self.block2(torch.cat((x, z_img), dim=1))
        else:
            x = self.block2(x)
        # loc 2
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
            z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        if self.loc[2]:
            x = self.block3(torch.cat((x, z_img), dim=1))
        else:
            x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.data_dim == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
            # x = self.resblock7(x)
            # x = self.resblock8(x)
            # x = self.resblock9(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)

        return (torch.tanh(x) + 1) / 2


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


# ESMA
class TargetEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.targetEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels, embedding_dim=d_model, padding_idx=0),  # turn label to embedding
            nn.Linear(d_model, dim),    # 32 -> 128
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.targetEmbedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Sequential(nn.GELU(),
                                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, padding_mode='reflect'))
        self.c2 = nn.Sequential(nn.GELU(),
                                nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2, padding_mode='reflect'))

    def forward(self, x, target_emb):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.t = nn.Sequential(nn.GELU(), nn.GroupNorm(16, in_ch),
                               nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1))

    def forward(self, x, target_emb):
        _, _, H, W = x.shape
        x = self.t(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.norm = nn.GroupNorm(16, in_ch)

        self.to_kv = nn.Conv2d(in_ch, in_ch * 2, 1)

        self.out = nn.Sequential(nn.GroupNorm(16, in_ch),
                                 nn.GELU(),
                                 nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
                                 )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        q_scale = int(C) ** (-0.5)
        kv = self.to_kv(x).chunk(2, dim=1)
        k, v = map(lambda t: rearrange(t, 'b c x y -> b c (x y)'), kv)
        q = F.softmax(k, dim=-2)
        k = F.softmax(k, dim=-1)
        q = q * q_scale
        context = torch.einsum('b d n, b e n -> b d e', k, v)
        assert list(context.shape) == [B, C, C]
        out = torch.einsum('b d e, b d n -> b e n', context, q)

        assert list(out.shape) == [B, C, H * W]
        out = rearrange(out, 'b c (i j) -> b c i j', i=H, j=W)
        out = self.out(out)
        return x + out


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, attn=True):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GELU(),
            nn.GroupNorm(16, in_ch),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, padding_mode='reflect'),
        )

        self.target_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(tdim, out_ch),
        )

        self.block2 = nn.Sequential(
            nn.GELU(),
            nn.GroupNorm(16, out_ch),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, padding_mode='reflect'),

        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, target):
        h = self.block1(x)
        h += self.target_proj(target)[:, :, None, None]
        h = self.block2(h)
        B, C, H, W = h.size()
        h = nn.LayerNorm([C, H, W], device=h.device)(h)
        h = h + self.shortcut(x)
        h = self.attn(h)

        return h


class GCT(nn.Module):

    def __init__(self, num_channels, tdim, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        b, c, h, w = x.shape
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                   (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / \
                   (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class Generator(nn.Module):
    def __init__(self, num_target, ch, ch_mult, num_res_blocks):
        super().__init__()
        tdim = ch * 4
        self.target_embedding = TargetEmbedding(num_target, ch, tdim)
        self.head = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.GELU(),
            nn.GroupNorm(16, ch)

        )
        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim))
                now_ch = out_ch

                chs.append(now_ch)

            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, attn=True),
            ResBlock(now_ch, now_ch, tdim, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult

            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, attn=False))
                now_ch = out_ch

            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        assert len(chs) == 0

        self.gct = GCT(num_channels=now_ch, tdim=tdim)

        self.tail = nn.Sequential(
            nn.GroupNorm(16, now_ch),

            nn.GELU(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1, padding_mode='reflect'),
        )

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.1)
            elif isinstance(m, torch.nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, target):

        targetemb = self.target_embedding(target)
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            h = layer(h, targetemb)
            hs.append(h)
        for layer in self.middleblocks:
            h = layer(h, targetemb)
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, targetemb)

        h = self.gct(h)
        h = self.tail(h)

        assert len(hs) == 0
        return (torch.tanh(h) + 1) / 2

if __name__ == '__main__':
    netG = GeneratorResnet()
    netG = GeneratorResnet_W()
    # Unet = GeneratorUnet_HF()

    # load pretrained model
    # checkpoint = {}
    # ckpt = torch.load('pretrained_generators/My/fn/netG_resnet50_0_fn_t24.pth')
    # if list(ckpt.keys())[0].startswith('module'):
    #     for k in ckpt.keys():
    #         checkpoint[k[7:]]=ckpt[k]
    # else:
    #     checkpoint = ckpt
    # Unet.load_state_dict(checkpoint)
    # Unet.eval()

    test_sample = torch.rand(1, 3, 224, 224)
    print('\n')
    # print('Generator output:', Unet(test_sample,test_sample).size())
    print('Generator output:', netG(test_sample).size())
    print('Generator parameters:', sum(p.numel() for p in netG.parameters() if p.requires_grad) / 1000000)