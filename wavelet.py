import numpy as np
import torch
import torch.nn as nn


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)

    return LL, LH, HL, HH


def get_wav_two(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav_two(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(Generator, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.feat_256_to_128 = UpBlock(256, 128)
        self.feat_256_to_64 = UpBlock(256, 64)
        self.feat_512_to_128 = UpBlock(512, 128)
        self.feat_1024_to_256 = UpBlock(1024, 256)
        self.feat_512_to_512 = UpBlock(512, 512)

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])
        # WaveUnpool
        self.recon_block1 = WaveUnpool(128, "sum")
        self.recon_block2 = WaveUnpool(256, "sum")
        self.recon_block3 = WaveUnpool(512, "sum")
        self.recon_block4 = WaveUnpool(1024, "sum")
        self.conv_f1 = conv2d(1024, 1024, 7, 1, 1, bias=False)
        self.conv_f2 = conv2d(512, 512, 11, 1, 1, bias=False)
        self.conv_f3 = conv2d(256, 256, 19, 1, 1, bias=False)
        # WavePool
        self.pool64 = WavePool(64).cuda()
        self.pool128 = WavePool(128).cuda()
        self.pool256 = WavePool(256).cuda()
        self.pool512 = WavePool(512).cuda()

    def forward(self, input, skips):
        if skips:
            feat_4 = self.init(input)
            feat_8 = self.feat_8(feat_4)
            LL_8, LH_8, HL_8, HH_8 = self.pool512(feat_8)
            original_8 = self.recon_block4(LL_8, LH_8, HL_8, HH_8)
            original_8 = self.feat_1024_to_256(original_8)
            fres_8 = LH_8 + HL_8 + HH_8
            feat_16 = self.feat_16(feat_8)
            LL_16, LH_16, HL_16, HH_16 = self.pool256(feat_16)
            original_16 = self.recon_block3(LL_16, LH_16, HL_16, HH_16)
            original_16 = self.feat_512_to_128(original_16)
            fres_16 = LH_16 + HL_16 + HH_16
            feat_32 = self.feat_32(feat_16)
            LL_32, LH_32, HL_32, HH_32 = self.pool128(feat_32)
            original_32 = self.recon_block2(LL_32, LH_32, HL_32, HH_32)
            original_32 = self.feat_256_to_128(original_32)
            fres_32 = LH_32 + HL_32 + HH_32
            feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
            LL_64, LH_64, HL_64, HH_64 = self.pool128(feat_64)
            original_64 = self.recon_block2(LL_64, LH_64, HL_64, HH_64)
            original_64 = self.feat_256_to_64(original_64)

            feat_128 = self.se_128(feat_8, self.feat_128(feat_64))

            feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

            if self.im_size == 256:
                return [self.to_big(feat_256), self.to_128(feat_128)]

            feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
            if self.im_size == 512:
                return [self.to_big(feat_512), self.to_128(feat_128)]

            feat_1024 = self.feat_1024(feat_512)

            im_128 = torch.tanh(self.to_128(feat_128))
            im_1024 = torch.tanh(self.to_big(feat_1024))

            return [im_1024, im_128]
        else:
            feat_4 = self.init(input)
            feat_8 = self.feat_8(feat_4)
            LL_8, LH_8, HL_8, HH_8 = self.pool512(feat_8)
            original_8 = self.recon_block4(LL_8, LH_8, HL_8, HH_8)
            original_8 = self.feat_1024_to_256(original_8)
            fres_8 = LH_8 + HL_8 + HH_8
            feat_16 = self.feat_16(feat_8)
            LL_16, LH_16, HL_16, HH_16 = self.pool256(feat_16)
            original_16 = self.recon_block3(LL_16, LH_16, HL_16, HH_16)
            original_16 = self.feat_512_to_128(original_16)
            fres_16 = LH_16 + HL_16 + HH_16
            feat_32 = self.feat_32(feat_16 + original_8)
            LL_32, LH_32, HL_32, HH_32 = self.pool128(feat_32)
            original_32 = self.recon_block2(LL_32, LH_32, HL_32, HH_32)
            original_32 = self.feat_256_to_128(original_32)
            fres_32 = LH_32 + HL_32 + HH_32

            feat_64 = self.se_64(feat_4, self.feat_64(feat_32 + original_16))
            LL_64, LH_64, HL_64, HH_64 = self.pool128(feat_64)
            original_64 = self.recon_block2(LL_64, LH_64, HL_64, HH_64)
            #  original_64 = self.feat_256_to_64(original_64)
            feat_128 = self.se_128(feat_8, self.feat_128(feat_64 + original_32))

            feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

            if self.im_size == 256:
                return [self.to_big(feat_256), self.to_128(feat_128)], fres_8, fres_16, fres_32

            feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
            if self.im_size == 512:
                return [self.to_big(feat_512), self.to_128(feat_128)], fres_8, fres_16, fres_32

            feat_1024 = self.feat_1024(feat_512)

            im_128 = torch.tanh(self.to_128(feat_128))
            im_1024 = torch.tanh(self.to_big(feat_1024))

            return [im_1024, im_128], fres_8, fres_16, fres_32


if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms

    x = torch.randn(2, 3, 224, 224)
    # 读取图片
    img_path = r"C:\Users\Arlse\Pictures\Saved Pictures\bob.jpg"
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = transforms.ToTensor()(img).unsqueeze(0)

    wavepool = WavePool(3)
    waveunpool = WaveUnpool(6, "sum")
    LL, LH, HL, HH = wavepool(img)
    print(LL.size(), LH.size(), HL.size(), HH.size())
    print(waveunpool(LL, LH, HL, HH).size())
    # print(waveunpool(LL, LH, HL, HH, img).size())