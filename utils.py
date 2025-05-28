import os
import math
import numbers
import numpy as np
import random
import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


# Transformations
class TwoCropTransform:
    def __init__(self, transform, img_size):
        self.transform = transform
        self.img_size = img_size
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.img_size),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomApply([color_jitter], p=0.8),
                                                   transforms.RandomGrayscale(p=0.2),
                                                   transforms.ToTensor()])

    def __call__(self, x):
        return [self.transform(x), self.data_transforms(x)]


def rotation(input):
    batch = input.shape[0]
    target = torch.tensor(np.random.permutation([0, 1, 2, 3] * (int(batch / 4) + 1)), device=input.device)[:batch]
    target = target.long()
    image = torch.zeros_like(input)
    image.copy_(input)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(input[i, :, :, :], target[i], [1, 2])

    return image, target


def low_freq_mask(input, ratio=0.15):
    # 获取输入的维度
    batch, channels, height, width = input.shape

    # 创建一个与输入相同大小的全0张量作为掩膜
    mask = torch.zeros((batch, channels, height, width), device=input.device)

    # 计算频谱中心
    if height % 2 == 0:
        center_height = height / 2 - 1 / 2
        center_width = width / 2 - 1 / 2
    else:
        center_height = height // 2
        center_width = width // 2

    # 计算掩膜半径
    radius = min(center_height, center_width) * ratio * torch.sqrt(torch.tensor(2.0, device=input.device))

    # 在频谱中心的一定半径内将掩膜设为1
    # for i in range(height):
    #     for j in range(width):
    #         if (i - center_height)**2 + (j - center_width)**2 < radius**2:
    #             mask[:, :, i, j] = 1

    # 创建一个表示每个像素到中心的距离的张量
    y, x = torch.meshgrid(torch.arange(-center_height, height - center_height, device=input.device),
                          torch.arange(-center_width, width - center_width, device=input.device))

    dist = x**2 + y**2


    # 在频谱中心的一定半径内将掩膜设为1
    mask[:, :, :, :] = (dist <= radius**2).float()

    return mask


###
# low pass filter
###
def ILPF(input, cutoff_frequency=224/4):
    batch, channels, height, width = input.shape
    mask = torch.zeros(input.shape, device=input.device)

    center_height, is_even_height = divmod(height, 2)
    center_width, is_even_width = divmod(width, 2)

    if is_even_height == 0:
        center_height -= 0.5
    if is_even_width == 0:
        center_width -= 0.5

    y, x = torch.meshgrid(torch.arange(-center_height, height - center_height, device=input.device),
                            torch.arange(-center_width, width - center_width, device=input.device))

    dist = torch.sqrt(x**2 + y**2)
    mask[:, :, :, :] = (dist <= cutoff_frequency).float()

    return mask


def GLPF(input, sigma):
    """
    :param input: input tensor
    :param sigma: standard deviation / cutoff frequency 可以近似理解为半径
    """
    batch, channels, height, width = input.shape

    center_height, is_even_height = divmod(height, 2)
    center_width, is_even_width = divmod(width, 2)

    if is_even_height == 0:
        center_height -= 0.5
    if is_even_width == 0:
        center_width -= 0.5

    y, x = torch.meshgrid(torch.arange(-center_height, height - center_height, device=input.device),
                            torch.arange(-center_width, width - center_width, device=input.device))

    dist = torch.sqrt(x**2 + y**2)  # distance from the center
    filter= torch.exp(-dist**2 / (2 * sigma)**2).unsqueeze(0).unsqueeze(0).repeat(batch, channels, 1, 1).to(input.device)

    return filter


def get_hf(img):
    img_freq = torch.fft.fft2(img, norm='ortho')
    img_freq = torch.fft.fftshift(img_freq)
    img_high_freq = img_freq * (1 - GLPF(img_freq, 28))
    # img_high_freq_mag = torch.log(1 + torch.sqrt(img_high_freq.real**2 + img_high_freq.imag**2 + 1e-8))
    # high_freq = torch.stack([img_high_freq.real, img_high_freq.imag], dim=-1)

    return img_high_freq.real


def get_lf(img):
    img_freq = torch.fft.fft2(img, norm='ortho')
    img_freq = torch.fft.fftshift(img_freq)
    img_low_freq = img_freq * GLPF(img_freq, 28)

    return img_low_freq.real


def get_hc(img):
    img_freq = torch.fft.fft2(img, norm='ortho')
    img_freq = torch.fft.fftshift(img_freq)
    img_high_freq = img_freq * (1 - GLPF(img_freq, 28))
    # img_high_freq = img_freq * (1 - ILPF(img_freq, 56))
    img_hf = torch.stack([img_high_freq.real, img_high_freq.imag], dim=-1)

    img_high_freq = torch.fft.ifftshift(img_high_freq)
    img_hc = torch.fft.ifft2(img_high_freq, norm='ortho').real

    if img_hc.size(1) == 3:
        img_hc = torch.clamp(img_hc, 0.0, 1.0)
    else:
        img_hc = torch.relu(img_hc)

    return img_hc, img_hf


def get_lc(img):
    img_freq = torch.fft.fft2(img, norm='ortho')
    img_freq = torch.fft.fftshift(img_freq)
    img_low_freq = img_freq * GLPF(img_freq, 28)
    img_lf = torch.stack([img_low_freq.real, img_low_freq.imag], dim=-1)

    img_low_freq = torch.fft.ifftshift(img_low_freq)
    img_lc = torch.fft.ifft2(img_low_freq, norm='ortho').real

    if img_lc.size(1) == 3:
        img_lc = torch.clamp(img_lc, 0.0, 1.0)
    else:
        img_lc = torch.relu(img_lc)

    return img_lc, img_lf


### function for TTAA ###
def get_RPD_mask(img):
    batch, channels, height, width = img.shape
    mask = torch.ones_like(img)
    square_size = 36
    for i in range(batch):
        squares = [(random.randint(0, height - square_size), random.randint(0, width - square_size)) for _
                   in range(3)]
        for top_left in squares:
            mask[i, :, top_left[0]:top_left[0] + square_size, top_left[1]:top_left[1] + square_size] = 0

    return mask


class Layer_out:
    features = None

    def __init__(self, feature_layer):
        self.features = []
        self.hook = feature_layer.register_forward_hook(self.hook_fn) # 获取model.features中某一层的output

    def hook_fn(self, module, input, output):
        # self.features.append(output.detach().cpu())
        self.features.append(output.detach())

    def remove(self):
        self.hook.remove()


class Layer_grad:
    grad = None

    def __init__(self, feature_layer):
        self.hook = feature_layer.register_full_backward_hook(self.hook_fn)  # 获取model.features中某一层的grad

    def hook_fn(self, module, input, output):
        self.grad = output[0]

    def remove(self):
        self.hook.remove()



if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    # img_path = r"C:\Users\Arlse\Pictures\Saved Pictures\bob.jpg"
    # from PIL import Image
    # img = Image.open(img_path)
    # img = transforms.ToTensor()(img).unsqueeze(0)
    # low_freq_mask(x, ratio=0.9)

    # mask = get_RPD_mask(x)
    # plt.imshow(mask[1].detach().permute(1,2,0).cpu().numpy())
    # plt.show()

    filter = ILPF(x, 224/4)
    plt.imshow(filter[0].detach().permute(1,2,0).cpu().numpy())
    plt.show()

    # img_f = torch.fft.fft2(img)
    # img_freq = torch.fft.fftshift(img_f)
    # img_high_freq = img_freq * (1 - GLPF(img_freq, 224/8))
    # img_high_freq = torch.fft.ifftshift(img_high_freq)
    # img_hc = torch.fft.ifft2(img_high_freq).real
    # plt.imshow(img_hc[0].detach().permute(1, 2, 0).cpu().numpy())
    # plt.show()
    # img_low_freq = img_freq * GLPF(img_freq, 224/8)
    # img_low_freq = torch.fft.ifftshift(img_low_freq)
    # img_lc = torch.fft.ifft2(img_low_freq).real
    # plt.imshow(img_lc[0].detach().permute(1, 2, 0).cpu().numpy())
    # plt.show()


    print()