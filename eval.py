"""
TTP Evaluation on ImageNet validation.
For each target, we have 49950 samples of the other classes.
"""

import argparse
import os
import socket

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchvision.models as models
import timm
from tqdm import tqdm

from GAP.material.models.generators import ResnetGenerator
from generators import GeneratorResnet, GeneratorResnet_R, GeneratorResnet_W
from gaussian_smoothing import *
from shapley import get_mask
from utils import *

# Purifier
from NRP import *

import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Targeted Transferable Perturbations')
# parser.add_argument('--test_dir', default='imagenet')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for evaluation')
parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget')
parser.add_argument('--epoch', type=int, default=19, help='which epoch to load for generator')

parser.add_argument('--re_noise', action='store_true', help='reweight noise use frequency information')
# parser.add_argument('--hf_disc', action='store_true', help='Add high frequency discriminator')
parser.add_argument('--triplet_loss', action='store_true', help='Use triplet loss')
parser.add_argument('--wavelet', action='store_true', help='Use wavelet transform')
parser.add_argument('--class_freq_mask', action='store_true', help='Use class frequency mask')

parser.add_argument('--checkpoint', type=str, default='pretrained_generators/TTP_new/resnet50/24/netG_resnet50_19_ttp_t24.pth', help='path to generator checkpoint')
parser.add_argument('--target_model', type=str, default='vgg19_bn', help='Black-Box(unknown) model: SIN, Augmix etc')
parser.add_argument('--target', type=int, default=24, help='Target label to transfer')
parser.add_argument('--source_model', type=str, default='resnet50', help='TTP Discriminator: \
{resnet18, resnet50, resnet101, resnet152, dense121, dense161, dense169, dense201,\
 vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,\
 ens_vgg16_vgg19_vgg11_vgg13_all_bn,\
 ens_res18_res50_res101_res152\
 ens_dense121_161_169_201}')

parser.add_argument('--source_domain', type=str, default='IN',
                    help='Source Domain (TTP): Natural Images (IN) or comic_books')
# For purification (https://github.com/Muzammal-Naseer/NRP)
parser.add_argument('--NRP', action='store_true', help='Apply Neural Purification to reduce adversarial effect')
args = parser.parse_args()
print(args)

if args.source_domain == 'IN':
    hostname = socket.gethostname()
    if hostname in ['user-Precision-7920-Tower', 'dell-Precision-7920-Tower', 'dell-Precision-7960-Tower']:  # 3091
        test_dir = '/datasets/Imagenet2012/val'
    elif hostname == 'ubuntu':  # 503
        test_dir = '/datasets/ILSVRC2012/val2'
    elif hostname == 'R2S1-gpu':  # 5014
        test_dir = '/datasets/ImageNet2012/val'
    elif hostname == 'LAPTOP-01RUAH3M':  # Laptop
        test_dir = 'E:/datasets/ILSVRC2012/val'
elif args.source_domain == 'comic_books':
    test_dir = '/datasets/comic books/test'

# Set-up log file
os.makedirs('1T_subsrc', exist_ok=True)
if args.re_noise:
    logfile = os.path.join('1T_subsrc',
                           're_noise_plus_1T_target{}_eval_eps{}_{}_to_{}_NRP_{}.log'.format(args.target, args.eps,
                                                                                         args.source_model,
                                                                                         args.target_model, args.NRP))
# elif args.fl:
#     logfile = os.path.join('1T_subsrc',
#                            'TTP_1T_target{}_eval_eps{}_{}_to_{}_fl_NRP_{}.log'.format(args.target, args.eps,
#                                                                                       args.source_model,
#                                                                                       args.target_model, args.NRP))
else:
    logfile = os.path.join('1T_subsrc', 'TTP_1T_target{}_eval_eps{}_{}_to_{}_NRP_{}.log'.format(args.target, args.eps,
                                                                                                args.source_model,
                                                                                                args.target_model,
                                                                                                args.NRP))

logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=logfile)

eps = args.eps / 255.0

# Set-up Kernel
kernel_size = 3
pad = 2
sigma = 1
kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()

# Load pretrained Generator
if args.re_noise:
    netG = GeneratorResnet_R()
elif args.wavelet:
    netG = GeneratorResnet_W()
elif 'GAP' in args.checkpoint:
    netG = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=[0])
else:
    netG = GeneratorResnet()

checkpoint = {}
ckpt = torch.load(args.checkpoint)
if list(ckpt.keys())[0].startswith('module'):
    for k in ckpt.keys():
        checkpoint[k[7:]]=ckpt[k]
else:
    checkpoint = ckpt
netG.load_state_dict(checkpoint)
netG = netG.cuda()

# if args.checkpoint.split('/')[0] == 'TTP_models':
#     netG.load_state_dict(torch.load(args.checkpoint))
#     netG = nn.DataParallel(netG).cuda()
# else:
#     netG = nn.DataParallel(netG).cuda()
#     netG.load_state_dict(torch.load(args.checkpoint))

netG.eval()

# Load Targeted Model
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

if args.target_model in model_names:
    model = models.__dict__[args.target_model](pretrained=True)
elif args.target_model == 'inception_v4':
    model = timm.create_model('inception_v4', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_v4.tf_in1k/model.safetensors')
elif args.target_model == 'incres_v2':
    model = timm.create_model('inception_resnet_v2', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_resnet_v2.tf_in1k/model.safetensors')
elif args.target_model == 'vit':
    model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg_in1k/model.safetensors')
# adversarial training
elif args.target_model == "ens_adv_inception_resnet_v2":
    model = timm.create_model("ens_adv_inception_resnet_v2", pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_resnet_v2.tf_ens_adv_in1k/model.safetensors')
elif args.target_model == "adv_inc_v3":
    model = timm.create_model('inception_v3.tf_adv_in1k', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_v3.tf_adv_in1k/model.safetensors')
elif args.target_model == "adv_resnet50":
    model = timm.create_model('resnet50.tf_adv_in1k', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--resnet50.tf_adv_in1k/model.safetensors')
elif args.target_model == 'SIN':
    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/resnet50_train_60_epochs-c8e5653e.pth.tar')
    model.load_state_dict(checkpoint["state_dict"])
elif args.target_model == 'Augmix':
    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/checkpoint.pth.tar')
    model.load_state_dict(checkpoint["state_dict"])
else:
    assert (args.target_model in model_names), 'Please provide correct target model names: {}'.format(model_names)

model = nn.DataParallel(model).cuda()
model.eval()

if args.NRP:
    purifier = NRP(3, 3, 64, 23)
    purifier.load_state_dict(torch.load('pretrained_purifiers/NRP.pth'))
    purifier = purifier.cuda()

####################
# Data
####################
# Input dimensions
scale_size = 256
img_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

if 'GAP' in args.checkpoint:
    normalize = transforms.Normalize(mean=mean, std=std)
    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
else:
    data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t

test_set = datasets.ImageFolder(test_dir, data_transform)

# Remove samples that belong to the target attack label.
source_samples = []
for img_name, label in test_set.samples:
    if label != args.target:
        source_samples.append((img_name, label))
test_set.samples = source_samples
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                          pin_memory=True)

test_size = len(test_set)
print('Test data size:', test_size)

def process_subplot(ax, data, title):
    img = torch.log(torch.abs(data)).detach().cpu().numpy()
    im = ax.imshow(img, vmin=-6, vmax=1)
    # im = ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
    return im

def main():
    acc = 0
    distance = 0

    if 'GAP' in args.checkpoint:
        def normalize_and_scale(delta_im, mode='train'):
            if args.target_model == 'inception_v3':
                delta_im = nn.ConstantPad2d((0, -1, -1, 0), 0)(delta_im)  # crop slightly to match inception

            delta_im = delta_im + 1  # now 0..2
            delta_im = delta_im * 0.5  # now 0..1

            # normalize image color channels
            for c in range(3):
                delta_im[:, c, :, :] = (delta_im[:, c, :, :].clone() - mean[c]) / std[c]

            # threshold each channel of each image in deltaIm according to inf norm
            # do on a per image basis as the inf norm of each image could be different
            # bs = args.batch_size if (mode == 'train') else args.batch_size
            for i in range(delta_im.size(0)):
                # do per channel l_inf normalization
                for ci in range(3):
                    l_inf_channel = delta_im[i, ci, :, :].detach().abs().max()
                    mag_in_scaled_c = args.eps / (255.0 * std[ci])
                    # gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
                    delta_im[i, ci, :, :] = delta_im[i, ci, :, :].clone() * np.minimum(1.0,
                                                                                       mag_in_scaled_c / l_inf_channel.cpu().numpy())

            return delta_im

        for i, (img, label) in tqdm(enumerate(test_loader)):
            # if (i+1) % 10 == 0:
            #     print('At Batch:', i+1)
            img, label = img.cuda(), label.cuda()

            target_label = torch.LongTensor(img.size(0))
            target_label.fill_(args.target)
            target_label = target_label.cuda()

            delta_im = netG(img).detach()
            delta_im = normalize_and_scale(delta_im, 'test')

            recons = torch.add(img, delta_im[0:img.size(0)])

            for cii in range(3):
                recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(img[:, cii, :, :].min(),
                                                                          img[:, cii, :, :].max())

            # plot
            # for c2 in range(3):
            #     recons[:, c2, :, :] = (recons[:, c2, :, :] * std[c2]) + mean[c2]
            #     img[:, c2, :, :] = (img[:, c2, :, :] * std[c2]) + mean[c2]
            #     delta_im[:, c2, :, :] = (delta_im[:, c2, :, :] * std[c2]) + mean[c2]
            # plt.imshow(delta_im[0].detach().permute(1, 2, 0).cpu().numpy())

            # plt.imshow(recons[0].detach().permute(1, 2, 0).cpu().numpy())
            # plt.show()

            out = model(recons)
            acc += torch.sum(out.argmax(dim=-1) == target_label).item()

            distance += (img - recons).max() * 255


    else:
        for i, (img, label) in tqdm(enumerate(test_loader)):
            # if (i+1) % 10 == 0:
            #     print('At Batch:', i+1)
            img, label = img.cuda(), label.cuda()

            target_label = torch.LongTensor(img.size(0))
            target_label.fill_(args.target)
            target_label = target_label.cuda()

            if args.re_noise:
                unr_adv = netG(img)[0].detach()
            elif args.class_freq_mask:
                unr_adv = netG(img).detach()
                mask_players = []
                # percent_range = [0, 7, 16, 26, 38, 51, 66, 83, 102, 127, 224]    # valid

                percent_range = [0, 5, 11, 18, 27, 38, 51, 66, 83, 103, 224]    # 245
                shapley_values = torch.tensor([8382.949,4237.4526,4382.732,4275.8066,3294.2754,2249.6772,1663.04,
                                               1519.0676,1401.532,894.4356], device='cuda')
                # percent_range = [0, 6, 13, 22, 32, 43, 55, 69, 85, 104, 224]    # 802
                # shapley_values = torch.tensor([9056.482,6023.7725,6145.063,5034.214,3519.6013,2733.2007,
                #                                2131.4336,1469.4655,724.2866,229.44354], device='cuda')
                for i in range(len(percent_range)-1):
                    mask_players.append(1 - get_mask(img[0][0], percent_range[i], percent_range[i + 1]))
                shapley_mask = torch.zeros((img_size, img_size), device='cuda')
                cum_weights = shapley_values.cumsum(dim=0) / shapley_values.sum()
                for i, mask_player in enumerate(mask_players):
                    if cum_weights[i].item() < 0.98:
                        shapley_mask += mask_player
                # for i, mask_player in enumerate(mask_players):
                #     shapley_mask += mask_player * shapley_weight[i].item()
                unr_adv_freq = torch.fft.fftshift(torch.fft.fft2(unr_adv, norm='ortho'))
                unr_adv_freq = unr_adv_freq * shapley_mask
                unr_adv = torch.fft.ifft2(torch.fft.ifftshift(unr_adv_freq), norm='ortho').real
            # elif 'CDA' in args.checkpoint:
            #     unr_adv = kernel(netG(img)).detach()
            else:
                # unr_adv = kernel(netG(img)).detach()
                unr_adv = netG(img).detach()

            adv = torch.min(torch.max(unr_adv, img - eps), img + eps)
            adv = torch.clamp(adv, 0.0, 1.0)

            # adv_freq = torch.fft.fft2(adv, norm='ortho')
            # adv_freq = torch.fft.fftshift(adv_freq)
            # adv_freq = torch.abs(adv_freq)
            # adv_freq = adv_freq.mean(dim=[0,1])
            # plt.imshow(torch.log(adv_freq).detach().cpu().numpy(), vmin=-4, vmax=1)
            # plt.colorbar()
            # plt.show()
            # print()

            # adv_freq = torch.fft.fft2(adv[0], norm='ortho')
            # adv_freq = torch.fft.fftshift(adv_freq)
            # adv_freq = torch.abs(adv_freq)
            # adv_freq = adv_freq.mean(dim=[0])
            # plt.imshow(torch.log(adv_freq).detach().cpu().numpy())
            # plt.colorbar()
            # plt.show()

            # img_freq = torch.fft.fft2(img, norm='ortho')
            # img_freq = torch.fft.fftshift(img_freq)
            # img_freq = torch.abs(img_freq)
            # img_freq = img_freq.mean(dim=[0,1])
            # plt.imshow(torch.log(img_freq).detach().cpu().numpy())
            # plt.colorbar()
            # plt.show()

            # visualize the adv
            # unr_noise = (unr_adv - img)
            # noise = (adv - img)
            #
            # index = 0
            # fig, ax = plt.subplots(2, 2)
            #
            # data_list = [torch.fft.fftshift(torch.fft.fft2(data, norm='ortho')) for data in [img, unr_adv, noise, adv]]
            # title_list = ['Frequency of Original Image', 'Frequency of Unconstrained Adversaries',
            #               'Frequency of Noise', 'Frequency of Adversaries']
            #
            # for i, axi in enumerate(ax.flat):
            #     # im = process_subplot(axi, data_list[i][index, 0, :, :], title_list[i])
            #     im = process_subplot(axi, torch.mean(data_list[i][index, :, :, :], dim=0), title_list[i])
            #     fig.colorbar(im, ax=axi)
            #
            # plt.tight_layout()
            # plt.show()
            # print()

            # fig, ax = plt.subplots(2, 2)
            # ax[0][0].imshow(img[index].permute(1, 2, 0).cpu().numpy())
            # ax[0][0].set_title('Original Image')
            # ax[0][0].axis('off')
            # ax[0][1].imshow(unr_adv[index].permute(1, 2, 0).cpu().numpy())
            # ax[0][1].set_title('Unconstrained Adversaries')
            # ax[0][1].axis('off')
            # ax[1][0].imshow((10*unr_noise[index]+0.5).permute(1, 2, 0).cpu().numpy())
            # ax[1][0].set_title('Unconstrained Noise')
            # ax[1][0].axis('off')
            # ax[1][1].imshow(adv[index].permute(1, 2, 0).cpu().numpy())
            # ax[1][1].set_title('Adversaries')
            # ax[1][1].axis('off')
            # plt.tight_layout()
            # plt.show()
            # print()

            # visualize high frequency and low frequency
            # adv_hf = get_hf(adv)
            # img_hf = get_hf(img)
            # fig, ax = plt.subplots(1, 2)
            # process_subplot(ax[0], torch.mean(img_hf[index],dim=0), 'Original Image HF')
            # process_subplot(ax[1], torch.mean(adv_hf[index],dim=0), 'Adversaries HF')
            # plt.tight_layout()
            # plt.show()

            # hc
            # fig, ax = plt.subplots(3, 2)
            # ax[0][0].imshow(img_hc[0].permute(1, 2, 0).cpu().numpy())
            # ax[0][0].set_title('Original Image HC')
            # ax[0][0].axis('off')
            # im_01 = process_subplot(ax[0][1], torch.mean(torch.fft.fftshift(torch.fft.fft2(img_hc[0], norm='ortho')),dim=0), 'Original Image HF')
            # fig.colorbar(im_01, ax=ax[0][1])
            # ax[1][0].imshow(adv_hc[0].permute(1, 2, 0).cpu().numpy())
            # ax[1][0].set_title('Adversaries HC')
            # ax[1][0].axis('off')
            # im_02 = process_subplot(ax[1][1], torch.mean(torch.fft.fftshift(torch.fft.fft2(adv_hc[0], norm='ortho')),dim=0), 'Adversaries HF')
            # fig.colorbar(im_02, ax=ax[1][1])
            # ax[2][0].imshow(unr_adv_hc[0].permute(1, 2, 0).cpu().numpy())
            # ax[2][0].set_title('Unconstrained Adversaries HC')
            # ax[2][0].axis('off')
            # im_03 = process_subplot(ax[2][1], torch.mean(torch.fft.fftshift(torch.fft.fft2(unr_adv_hc[0], norm='ortho')),dim=0), 'Unconstrained Adversaries HF')
            # fig.colorbar(im_03, ax=ax[2][1])
            # plt.tight_layout()
            # plt.show()

            # lc
            # unr_adv_lc = unr_adv - unr_adv_hc
            # adv_lc = adv - adv_hc
            # img_lc = img - img_hc
            # fig, ax = plt.subplots(3, 2)
            # ax[0][0].imshow(img_lc[0].permute(1, 2, 0).cpu().numpy())
            # ax[0][0].set_title('Original Image LC')
            # ax[0][0].axis('off')
            # im_01 = process_subplot(ax[0][1], torch.mean(torch.fft.fftshift(torch.fft.fft2(img_lc[0], norm='ortho')),dim=0), 'Original Image LF')
            # fig.colorbar(im_01, ax=ax[0][1])
            # ax[1][0].imshow(adv_lc[0].permute(1, 2, 0).cpu().numpy())
            # ax[1][0].set_title('Adversaries LC')
            # ax[1][0].axis('off')
            # im_02 = process_subplot(ax[1][1], torch.mean(torch.fft.fftshift(torch.fft.fft2(adv_lc[0], norm='ortho')),dim=0), 'Adversaries LF')
            # fig.colorbar(im_02, ax=ax[1][1])
            # ax[2][0].imshow(unr_adv_lc[0].permute(1, 2, 0).cpu().numpy())
            # ax[2][0].set_title('Unconstrained Adversaries HF')
            # ax[2][0].axis('off')
            # im_03 = process_subplot(ax[2][1], torch.mean(torch.fft.fftshift(torch.fft.fft2(unr_adv_lc[0], norm='ortho')),dim=0), 'Unconstrained Adversaries LF')
            # fig.colorbar(im_03, ax=ax[2][1])
            # plt.tight_layout()
            # plt.show()

            if args.NRP:
                # Purify Adversary
                adv = purifier(adv).detach()

            out = model(normalize(adv.clone().detach()))
            acc += torch.sum(out.argmax(dim=-1) == target_label).item()

            distance += (img - adv).max() * 255

            # if i == 0:
            #     torchvision.utils.save_image(torchvision.utils.make_grid(adv, normalize=True, scale_each=True), 'adv.png')
            #     torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), 'org.png')

    print('Target:{}, Acc:{}'.format(args.target, acc / test_size))

    logger.info('{:<10} {:<10} {:<10} {:<10} {:<10}'.format('Epsilon', 'Target', 'Acc.', 'Epoch', 'Distance'))
    logger.info('{:<10} {:<10} {:<10.4f} {:<10} {:<10.4f}'.format(int(eps * 255), args.target, acc / test_size,
                                                                  args.epoch, distance / (i + 1)))


if __name__ == '__main__':
    main()