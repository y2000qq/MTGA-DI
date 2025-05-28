import argparse
import os
import logging

import numpy as np
import socket

# from focal_frequency_loss import FocalFrequencyLoss
from matplotlib import pyplot as plt

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
# from pytorch_wavelets import DWTForward, DWTInverse

from generators import *
from discriminator import *
from gaussian_smoothing import *
from high_frequency_discriminator import High_Frequency_Discriminator
from reweight_net import ReweightNet

from utils import *


parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
parser.add_argument('--src', default='IN_50k_new', help='Source Domain: imagenet, imagenet_10c, IN_50k, etc')
parser.add_argument('--match_target', type=int, default=24, help='Target Domain samples')
parser.add_argument('--model_type', type=str, default='resnet50', help='Model under attack (discrimnator)')
parser.add_argument('--gs', action='store_true', help='Apply gaussian smoothing')
parser.add_argument('--method', type=str, default='none',help='none, argu, single')
parser.add_argument('--save_dir', type=str, default='pretrained_generators', help='Directory to save generators')

parser.add_argument('--alpha', type=float, default=0.05, help='hyper-parameter of amp loss')
parser.add_argument('--beta', type=float, default=0.05, help='hyper-parameter of pha loss')

parser.add_argument('--match_dataset', default='imagenet', help='Target domain')
parser.add_argument('--batch_size', type=int, default=20, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget during training, eps')

parser.add_argument('--test', action='store_true', help='Test')
# todo
parser.add_argument('--resume', action='store_true', help='Resume training')

args = parser.parse_args()
print(args)

if args.test:
    logfile = os.path.join('train_loss', 'test.log')
else:
    logfile = None

if logfile:
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logging.info(args)

hostname = socket.gethostname()

if args.src == 'comic_books':
    source_path = '/datasets/comic books/train'
elif args.src == 'imagenet' or args.src == 'imagenet_10c':
    hostname = socket.gethostname()
    if hostname == 'user-Precision-7920-Tower':  # 3091
        source_path = '/datasets/Imagenet2012/train'
    elif hostname == 'ubuntu':  # 503
        source_path = '/datasets/ILSVRC2012/train'
    elif hostname == 'R2S1-gpu':  # 5014
        source_path = '/datasets/ImageNet2012/train'
elif args.src == 'IN_50k':
    if hostname == 'dell-PowerEdge-T640': # 4090
        source_path = '/data/ImageNet_50k'
    else:
        source_path = '/datasets/ImageNet_50k'
elif args.src == 'IN_50k_new':
    source_path = '/datasets/ImageNet_50k_990c'
else:
    assert False, 'Please provide correct source dataset names: {}'.format(args.src)

if args.match_dataset == 'imagenet':
    if hostname in ['user-Precision-7920-Tower', 'dell-Precision-7920-Tower', 'dell-Precision-7960-Tower']:  # 3091 or 3090 or A6000
        match_dir = '/datasets/Imagenet2012/train'
    elif hostname == 'ubuntu':  # 503
        match_dir = '/datasets/ILSVRC2012/train'
    elif hostname == 'R2S1-gpu':  # 5014
        match_dir = '/datasets/ImageNet2012/train'
    elif hostname == 'dell-PowerEdge-T640': # 4090
        match_dir = '/data/Imagenet2012/train'

def main():
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    eps = args.eps / 255

    # Discriminator
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    if args.model_type in model_names:
        model = models.__dict__[args.model_type](pretrained=True)
    else:
        assert (args.model_type in model_names), 'Please provide correct target model names: {}'.format(model_names)

    model = nn.DataParallel(model).cuda()
    model.eval()

    # Input dimensions
    if args.model_type == 'inception_v3':
        scale_size = 300
        img_size = 299
    else:
        scale_size = 256
        img_size = 224

    # Generator
    if args.model_type == 'inception_v3':
        netG = GeneratorResnet(inception=True)
    else:
        netG = GeneratorResnet()
    netG = nn.DataParallel(netG).cuda()
    # Optimizer
    optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Data
    train_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()])
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    def normalize(t):
        norm_t = torch.zeros_like(t)
        norm_t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
        norm_t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
        norm_t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
        return norm_t

    if 'argu' in args.method:
        train_set = torchvision.datasets.ImageFolder(source_path, TwoCropTransform(train_transform, img_size))
    else:
        train_set = torchvision.datasets.ImageFolder(source_path, train_transform)

    if args.src == 'imagenet_10c':
        source_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]

        train_set.samples = [train_set.samples[i] for i in range(len(train_set.targets))
                                   if train_set.targets[i] in source_classes]
        train_set.targets = [train_set.targets[i] for i in range(len(train_set.targets))
                                   if train_set.targets[i] in source_classes]

    train_size = len(train_set)
    if train_size % args.batch_size != 0:
        train_size = (train_size // args.batch_size) * args.batch_size
        train_set.samples = train_set.samples[0:train_size]
        train_set.targets = train_set.targets[0:train_size]

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)

    print('Training data size:', train_size)

    # original
    # match_dir = os.path.join(match_dir, target_dict[args.match_target])
    # train_set_match = torchvision.datasets.ImageFolder(match_dir, train_transform)
    # modified by Arlse
    train_set_match = torchvision.datasets.ImageFolder(match_dir, train_transform)
    train_set_match.samples = [train_set_match.samples[i] for i in range(len(train_set_match.targets))
                               if train_set_match.targets[i] == args.match_target]
    train_set_match.targets = [train_set_match.targets[i] for i in range(len(train_set_match.targets))
                               if train_set_match.targets[i] == args.match_target]
    # finish

    # original
    # if len(train_set_match) < 1300:
    #     train_set_match.samples = train_set_match.samples[0:1000]
    # modified by Arlse
    train_match_size = len(train_set_match)
    if train_match_size % args.batch_size != 0:
        train_match_size = (train_match_size // args.batch_size) * args.batch_size
        train_set_match.samples = train_set_match.samples[0:train_match_size]
        train_set_match.targets = train_set_match.targets[0:train_match_size]
    # finish
    train_loader_match = torch.utils.data.DataLoader(train_set_match, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=4, pin_memory=True)
    train_size_match = len(train_set_match)
    print('Training (Match) data size:', train_size_match)
    # Iterator
    dataiter = iter(train_loader_match)


    kernel_size = 3
    pad = 2
    sigma = 1
    kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()

    criterion_kl = nn.KLDivLoss(reduction='sum')
    criterion_bce = nn.BCELoss(reduction='sum')
    criterion_ce = nn.CrossEntropyLoss()
    criterion_L1 = nn.L1Loss(reduction='sum')
    criterion_L2 = nn.MSELoss(reduction='sum')

    if args.method == 'img_argu':
        # My method
        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs[0].cuda()
                img_rot = rotation(img)[0]
                img_aug = imgs[1].cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()

                adv = netG(img)
                adv_rot = netG(img_rot)
                adv_aug = netG(img_aug)

                loss_dcs = torch.tensor(0.0).cuda()

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_freq = torch.abs(adv_freq)
                adv_freq = adv_freq.mean(dim=0)

                adv_rot_freq = torch.fft.fft2(adv_rot, norm='ortho')
                adv_rot_freq = torch.fft.fftshift(adv_rot_freq)
                adv_rot_freq = torch.abs(adv_rot_freq)
                adv_rot_freq = adv_rot_freq.mean(dim=0)

                adv_aug_freq = torch.fft.fft2(adv_aug, norm='ortho')
                adv_aug_freq = torch.fft.fftshift(adv_aug_freq)
                adv_aug_freq = torch.abs(adv_aug_freq)
                adv_aug_freq = adv_aug_freq.mean(dim=0)

                # like img

                # DCS_img
                img_freq = torch.fft.fft2(img, norm='ortho')
                img_freq = torch.fft.fftshift(img_freq)
                img_freq = torch.abs(img_freq)
                img_freq = img_freq.mean(dim=0)

                # img_rot_freq = torch.fft.fft2(img_rot, norm='ortho')
                # img_rot_freq = torch.fft.fftshift(img_rot_freq)
                # img_rot_freq = torch.abs(img_rot_freq)
                # img_rot_freq = img_rot_freq.mean(dim=0)
                #
                # img_aug_freq = torch.fft.fft2(img_aug, norm='ortho')
                # img_aug_freq = torch.fft.fftshift(img_aug_freq)
                # img_aug_freq = torch.abs(img_aug_freq)
                # img_aug_freq = img_aug_freq.mean(dim=0)

                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_rot_freq, img_rot_freq)
                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_aug_freq, img_aug_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_rot_freq, img_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_aug_freq, img_freq)

                # like img_match

                # DCS_img_match
                # img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                # img_match_freq = torch.fft.fftshift(img_match_freq)
                # img_match_freq = torch.abs(img_match_freq)
                # img_match_freq = img_match_freq.mean(dim=0)
                #
                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq)
                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L2(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)
                    adv_rot = kernel(adv_rot)
                    adv_aug = kernel(adv_aug)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)
                adv_rot = torch.min(torch.max(adv_rot, img_rot - eps), img_rot + eps)
                adv_rot = torch.clamp(adv_rot, 0.0, 1.0)
                adv_aug = torch.min(torch.max(adv_aug, img_aug - eps), img_aug + eps)
                adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

                adv_out = model(normalize(adv))  # (batch_size, num_classes)
                img_match_out = model(normalize(img_match))
                adv_rot_out = model(normalize(adv_rot))
                adv_aug_out = model(normalize(adv_aug))

                loss_kl = 0.0
                for out in [adv_out, adv_rot_out, adv_aug_out]:
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(out, dim=1),
                                                                      F.softmax(img_match_out, dim=1))
                    # KL divergence is not symmetric
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                      F.softmax(out, dim=1))

                loss = loss_kl + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0

            file_name = '/netG_{}_{}_DCS_argu_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'img_match_argu':
        # My method
        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs[0].cuda()
                img_rot = rotation(img)[0]
                img_aug = imgs[1].cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()

                adv = netG(img)
                adv_rot = netG(img_rot)
                adv_aug = netG(img_aug)

                loss_dcs = torch.tensor(0.0).cuda()

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_freq = torch.abs(adv_freq)
                adv_freq = adv_freq.mean(dim=0)

                adv_rot_freq = torch.fft.fft2(adv_rot, norm='ortho')
                adv_rot_freq = torch.fft.fftshift(adv_rot_freq)
                adv_rot_freq = torch.abs(adv_rot_freq)
                adv_rot_freq = adv_rot_freq.mean(dim=0)

                adv_aug_freq = torch.fft.fft2(adv_aug, norm='ortho')
                adv_aug_freq = torch.fft.fftshift(adv_aug_freq)
                adv_aug_freq = torch.abs(adv_aug_freq)
                adv_aug_freq = adv_aug_freq.mean(dim=0)

                # like img

                # DCS_img
                # img_freq = torch.fft.fft2(img, norm='ortho')
                # img_freq = torch.fft.fftshift(img_freq)
                # img_freq = torch.abs(img_freq)
                # img_freq = img_freq.mean(dim=0)
                #
                # img_rot_freq = torch.fft.fft2(img_rot, norm='ortho')
                # img_rot_freq = torch.fft.fftshift(img_rot_freq)
                # img_rot_freq = torch.abs(img_rot_freq)
                # img_rot_freq = img_rot_freq.mean(dim=0)
                #
                # img_aug_freq = torch.fft.fft2(img_aug, norm='ortho')
                # img_aug_freq = torch.fft.fftshift(img_aug_freq)
                # img_aug_freq = torch.abs(img_aug_freq)
                # img_aug_freq = img_aug_freq.mean(dim=0)
                #
                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_rot_freq, img_rot_freq)
                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_aug_freq, img_aug_freq)

                # like img_match

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_freq = torch.abs(img_match_freq)
                img_match_freq = img_match_freq.mean(dim=0)

                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_rot_freq, img_match_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_aug_freq, img_match_freq)
                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L2(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)
                    adv_rot = kernel(adv_rot)
                    adv_aug = kernel(adv_aug)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)
                adv_rot = torch.min(torch.max(adv_rot, img_rot - eps), img_rot + eps)
                adv_rot = torch.clamp(adv_rot, 0.0, 1.0)
                adv_aug = torch.min(torch.max(adv_aug, img_aug - eps), img_aug + eps)
                adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

                img_match_out = model(normalize(img_match))
                adv_out = model(normalize(adv))  # (batch_size, num_classes)
                adv_rot_out = model(normalize(adv_rot))
                adv_aug_out = model(normalize(adv_aug))

                loss_kl = 0.0
                for out in [adv_out, adv_rot_out, adv_aug_out]:
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(out, dim=1),
                                                                      F.softmax(img_match_out, dim=1))
                    # KL divergence is not symmetric
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                      F.softmax(out, dim=1))


                # loss_sim = 0.0
                # # Neighbourhood similarity
                # St = torch.matmul(img_match_out, img_match_out.t())
                # norm = torch.matmul(torch.linalg.norm(img_match_out, dim=1, ord=2),
                #                     torch.linalg.norm(img_match_out, dim=1, ord=2).t())
                # St = St / norm
                # for out in [adv_rot_out, adv_aug_out]:
                #     Ss = torch.matmul(adv_out, out.t())
                #     norm = torch.matmul(torch.linalg.norm(adv_out, dim=1, ord=2),
                #                         torch.linalg.norm(out, dim=1, ord=2).t())
                #     Ss = Ss / norm
                #     loss_sim += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(Ss, dim=1),
                #                                                        F.softmax(St, dim=1))
                #     loss_sim += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(St, dim=1),
                #                                                        F.softmax(Ss, dim=1))

                loss = loss_kl + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()


                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0
                    running_loss_sim = 0
            if args.gs:
                file_name = '/netG_{}_{}_img_match_argu_gs_t{}.pth'
            else:
                file_name = '/netG_{}_{}_img_match_argu_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'single':
        # My method
        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()

                adv = netG(img)

                loss_dcs = torch.tensor(0.0).cuda()

                # like img


                # DCS_img
                img_freq = torch.fft.fft2(img, norm='ortho')
                img_freq = torch.fft.fftshift(img_freq)
                img_freq = torch.abs(img_freq)
                # img_freq = img_freq.mean(dim=1)


                # like img_match


                # DCS_img_match
                # img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                # img_match_freq = torch.fft.fftshift(img_match_freq)
                # img_match_freq = torch.abs(img_match_freq)
                # img_match_freq = img_match_freq.mean(dim=1)

                # plt.imshow(torch.log(img_match_freq).detach().cpu().numpy())
                # plt.colorbar()
                # plt.show()

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_freq = torch.abs(adv_freq)
                # adv_freq = adv_freq.mean(dim=1)

                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                loss_dcs += 0.001 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq)

                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L2(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_out = model(normalize(adv))
                img_match_out = model(normalize(img_match))

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                loss = loss_kl + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0

            file_name = '/netG_{}_{}_DCS_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'img_match':
        # My method
        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()

                adv = netG(img)

                loss_dcs = torch.tensor(0.0, device='cuda')

                # like img

                # DCS_img
                # img_freq = torch.fft.fft2(img, norm='ortho')
                # img_freq = torch.fft.fftshift(img_freq)
                # img_freq = torch.abs(img_freq)
                # img_freq = img_freq.mean(dim=0)

                # like img_match

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_freq = torch.abs(img_match_freq)
                img_match_freq = img_match_freq.mean(dim=0)
                img_match_freq = img_match_freq.unsqueeze(0)

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_freq = torch.abs(adv_freq)
                adv_freq = adv_freq.mean(dim=0)
                adv_freq = adv_freq.unsqueeze(0)

                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq)
                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq.repeat(args.batch_size, 1, 1, 1))

                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L2(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_out = model(normalize(adv))
                img_match_out = model(normalize(img_match))

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                loss = loss_kl + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0

            file_name = '/netG_{}_{}_DCS_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'img_match_attn_mask':
        # My method
        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                target_label = torch.LongTensor(img.size(0))
                target_label.fill_(args.match_target)
                target_label = target_label.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()
                img_match_temp = img_match.clone()
                img_match_temp.requires_grad = True

                img_match_temp_out = model(normalize(img_match_temp))
                loss_ce_img_match = criterion_ce(img_match_temp_out, target_label)
                loss_ce_img_match.backward()

                img_match_grad = img_match_temp.grad
                img_match_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
                img_match_grad_amp = torch.abs(img_match_grad_freq)
                img_match_grad_amp = img_match_grad_amp.mean(dim=[0,1])
                # normalize
                img_match_grad_amp = (img_match_grad_amp - img_match_grad_amp.min()) / (img_match_grad_amp.max() - img_match_grad_amp.min())
                attn_mask = img_match_grad_amp.clone().detach()

                netG.train()

                adv = netG(img)

                loss_dcs = torch.tensor(0.0).cuda()

                # like img

                # DCS_img
                # img_freq = torch.fft.fft2(img, norm='ortho')
                # img_freq = torch.fft.fftshift(img_freq)
                # img_freq = torch.abs(img_freq)
                # img_freq = img_freq.mean(dim=0)

                # like img_match

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_amp = torch.abs(img_match_freq)
                img_match_amp = img_match_amp.mean(dim=0)
                img_match_amp = img_match_amp * attn_mask
                # img_match_freq = img_match_freq.unsqueeze(0)

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_amp = torch.abs(adv_freq)
                adv_amp = adv_amp.mean(dim=0)
                adv_amp = adv_amp * attn_mask
                # adv_freq = adv_freq.unsqueeze(0)


                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_amp, img_match_amp)
                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq.repeat(args.batch_size, 1, 1, 1))

                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L2(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_out = model(normalize(adv))
                img_match_out = model(normalize(img_match))

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                loss = loss_kl + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0

            file_name = '/netG_{}_{}_DCS_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'img_match_attn_mask_pro':
        # My method
        attn_mask_dict = {24: 0.609957239,
                          99: 1.202504582,
                          245: 1.00565058,
                          344: 0.958460599,
                          471: 1.006566891,
                          555: 1.136835675,
                          661: 0.916463042,
                          701: 0.965791081,
                          802: 0.993891265,
                          919: 1.203879047}

        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                target_label = torch.LongTensor(img.size(0))
                target_label.fill_(args.match_target)
                target_label = target_label.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()
                # img_match_temp = img_match.clone()
                # img_match_temp.requires_grad = True
                #
                # img_match_temp_out = model(normalize(img_match_temp))
                # loss_ce_img_match = criterion_ce(img_match_temp_out, target_label)
                # loss_ce_img_match.backward()
                #
                # img_match_grad = img_match_temp.grad
                # img_match_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
                # img_match_grad_amp = torch.abs(img_match_grad_freq)
                # img_match_grad_amp = img_match_grad_amp.mean(dim=[0,1])
                # # normalize
                # img_match_grad_amp = (img_match_grad_amp - img_match_grad_amp.min()) / (img_match_grad_amp.max() - img_match_grad_amp.min())
                # attn_mask = img_match_grad_amp.clone().detach()

                attn_mask = torch.ones(img_match.shape[1:], device='cuda') * attn_mask_dict[args.match_target]
                attn_mask = torch.clamp(attn_mask, 0.8, 1.2)

                netG.train()

                adv = netG(img)

                loss_dcs = torch.tensor(0.0).cuda()

                # like img

                # DCS_img
                # img_freq = torch.fft.fft2(img, norm='ortho')
                # img_freq = torch.fft.fftshift(img_freq)
                # img_freq = torch.abs(img_freq)
                # img_freq = img_freq.mean(dim=0)

                # like img_match

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_amp = torch.abs(img_match_freq)
                img_match_amp = img_match_amp.mean(dim=0)
                img_match_amp = img_match_amp * attn_mask
                # img_match_freq = img_match_freq.unsqueeze(0)

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_amp = torch.abs(adv_freq)
                adv_amp = adv_amp.mean(dim=0)
                adv_amp = adv_amp * attn_mask
                # adv_freq = adv_freq.unsqueeze(0)

                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_amp, img_match_amp)
                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq.repeat(args.batch_size, 1, 1, 1))

                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L2(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_out = model(normalize(adv))
                img_match_out = model(normalize(img_match))

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                loss = loss_kl + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0

            file_name = '/netG_{}_{}_DCS_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'img_match_attn_mask_pro_plus':
        # My method
        attn_mask_dict = {
            'resnet50':{
                '24': 0.772123954,
                '99': 0.948461476,
                '245': 0.935379248,
                '344': 0.831811616,
                '471': 1.001062931,
                '555': 1.125071543,
                '661': 0.979531765,
                '701': 1.262162383,
                '802': 1.004333488,
                '919': 1.140061595
            },
            'densenet121': {
                '24': 0.800130102,
                '99': 1.044072207,
                '245': 1.005691982,
                '344': 0.913644495,
                '471': 1.025532607,
                '555': 1.102943568,
                '661': 1.055781428,
                '701': 1.025857863,
                '802': 1.011871849,
                '919': 1.014473898
            },
            'vgg19_bn': {
                '24': 0.680206847,
                '99': 0.923321712,
                '245': 0.969885579,
                '344': 0.836043709,
                '471': 1.038444439,
                '555': 1.184687741,
                '661': 1.027914921,
                '701': 1.295832651,
                '802': 0.980181108,
                '919': 1.063481293
            },
        }

        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                target_label = torch.LongTensor(img.size(0))
                target_label.fill_(args.match_target)
                target_label = target_label.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()
                # img_match_temp = img_match.clone()
                # img_match_temp.requires_grad = True
                #
                # img_match_temp_out = model(normalize(img_match_temp))
                # loss_ce_img_match = criterion_ce(img_match_temp_out, target_label)
                # loss_ce_img_match.backward()

                # img_match_grad = img_match_temp.grad
                # img_match_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
                # img_match_grad_amp = torch.abs(img_match_grad_freq)
                # img_match_grad_amp = img_match_grad_amp.mean(dim=[0,1])
                # # normalize
                # img_match_grad_amp = (img_match_grad_amp - img_match_grad_amp.min()) / (img_match_grad_amp.max() - img_match_grad_amp.min())
                # attn_mask = img_match_grad_amp.clone().detach()

                attn_mask = torch.ones(img_match.shape[1:], device='cuda') * attn_mask_dict['{}'.format(args.model_type)]['{}'.format(args.match_target)]
                attn_mask = torch.clamp(attn_mask, 0.8, 1.2)

                netG.train()

                adv = netG(img)

                loss_dcs = torch.tensor(0.0).cuda()

                # like img

                # DCS_img
                # img_freq = torch.fft.fft2(img, norm='ortho')
                # img_freq = torch.fft.fftshift(img_freq)
                # img_freq = torch.abs(img_freq)
                # img_freq = img_freq.mean(dim=0)

                # like img_match

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_amp = torch.abs(img_match_freq)
                img_match_amp = img_match_amp.mean(dim=0)
                img_match_amp = img_match_amp * attn_mask
                # img_match_freq = img_match_freq.unsqueeze(0)

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_amp = torch.abs(adv_freq)
                adv_amp = adv_amp.mean(dim=0)
                adv_amp = adv_amp * attn_mask
                # adv_freq = adv_freq.unsqueeze(0)

                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_amp, img_match_amp)
                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq.repeat(args.batch_size, 1, 1, 1))

                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L2(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_out = model(normalize(adv))
                img_match_out = model(normalize(img_match))

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                loss = loss_kl + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0

            file_name = '/netG_{}_{}_DCS_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'img_match_attn_mask_ssa':
        # My method
        rho = 0.5
        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                target_label = torch.LongTensor(img.size(0))
                target_label.fill_(args.match_target)
                target_label = target_label.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()
                # img_match.requires_grad = True

                # SSA
                gauss = torch.randn(img_match.size(), device='cuda') * eps
                img_match_argu_freq = torch.fft.fft2(img_match + gauss, norm='ortho')
                mask = (torch.rand_like(img_match) * 2 * rho + 1 - rho).cuda()
                # mask = torch.ones(img_match.size()).cuda()
                img_match_argu = torch.fft.ifft2(img_match_argu_freq * mask, norm='ortho').real
                img_match_argu.requires_grad = True

                img_match_argu_out = model(normalize(img_match_argu))
                loss_ce_img_match_argu = criterion_ce(img_match_argu_out, target_label)
                loss_ce_img_match_argu.backward(retain_graph=True)

                img_match_argu_grad = img_match_argu.grad
                img_match_argu_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_match_argu_grad, norm='ortho'))
                img_match_argu_grad_amp = torch.abs(img_match_argu_grad_freq)
                img_match_argu_grad_amp = img_match_argu_grad_amp.mean(dim=[0,1])
                # normalize
                attn_mask = (img_match_argu_grad_amp - img_match_argu_grad_amp.min()) / (img_match_argu_grad_amp.max() - img_match_argu_grad_amp.min())

                netG.train()

                adv = netG(img)

                loss_dcs = torch.tensor(0.0).cuda()

                # like img

                # DCS_img
                # img_freq = torch.fft.fft2(img, norm='ortho')
                # img_freq = torch.fft.fftshift(img_freq)
                # img_freq = torch.abs(img_freq)
                # img_freq = img_freq.mean(dim=0)

                # like img_match

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_amp = torch.abs(img_match_freq)
                img_match_amp = img_match_amp.mean(dim=0)
                img_match_amp = img_match_amp * attn_mask
                # img_match_freq = img_match_freq.unsqueeze(0)

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_amp = torch.abs(adv_freq)
                adv_amp = adv_amp.mean(dim=0)
                adv_amp = adv_amp * attn_mask
                # adv_freq = adv_freq.unsqueeze(0)


                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_amp, img_match_amp)
                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq.repeat(args.batch_size, 1, 1, 1))

                # loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L2(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_out = model(normalize(adv))
                img_match_out = model(normalize(img_match))

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                loss = loss_kl + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0

            file_name = '/netG_{}_{}_DCS_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'DCS_img_match_fea':
        if args.model_type == 'resnet50':
            target_layer = model.module._modules.get('layer4')         # layer1: 256*56*56 layer2: 512*28*28 layer3: 1024*14*14 layer4: 2048*7*7
        elif args.model_type == 'vgg19':
            target_layer = model.module._modules.get('features')[16]
        elif args.model_type == 'vgg19_bn':
            target_layer = model.module._modules.get('features')[23]    # layer 24 in vgg19_bn corresponds to layer 17 in vgg19: 256*56*56
        elif args.model_type == 'densenet121':
            target_layer = model.module._modules.get('features')[10]     # denseblock1:4 256*56*56 denseblock2:6 512*28*28 denseblock3:8 1024*14*14 denseblock4:10 1024*7*7
        else:
            assert False, 'Please provide correct target model names: {}'.format(model_names)

        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs_fea = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()

                adv = netG(img)

                if args.gs:
                    adv = kernel(adv)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                h = Layer_out(target_layer)
                adv_out = model(normalize(adv))
                adv_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                img_match_out = model(normalize(img_match))
                img_match_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                # pro
                # img_match_fea_att = img_match_fea.mean(dim=1).abs().unsqueeze(1)
                # img_match_fea = img_match_fea * img_match_fea_att

                img_match_fea_freq = torch.fft.fft2(img_match_fea, norm='ortho')
                img_match_fea_freq = torch.fft.fftshift(img_match_fea_freq)
                img_match_fea_freq = torch.abs(img_match_fea_freq)
                img_match_fea_freq = img_match_fea_freq.mean(dim=0)
                img_match_fea_freq = img_match_fea_freq.unsqueeze(0)

                # adv_fea_att = adv_fea.mean(dim=1).abs().unsqueeze(1)
                # adv_fea = adv_fea * adv_fea_att

                # adv_fea = adv_fea * img_match_fea_att
                adv_fea_freq = torch.fft.fft2(adv_fea, norm='ortho')
                adv_fea_freq = torch.fft.fftshift(adv_fea_freq)
                adv_fea_freq = torch.abs(adv_fea_freq)
                adv_fea_freq = adv_fea_freq.mean(dim=0)
                adv_fea_freq = adv_fea_freq.unsqueeze(0)

                loss_dcs_fea = torch.tensor(0.0, device='cuda')
                loss_dcs_fea += args.beta * (1.0 / args.batch_size) * criterion_L1(adv_fea_freq, img_match_fea_freq)
                # loss_dcs_fea += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_fea_freq, img_match_fea_freq.repeat(args.batch_size, 1, 1, 1))

                loss = loss_kl + loss_dcs_fea
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs_fea += loss_dcs_fea.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs_fea: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs_fea / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs_fea: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs_fea / 10))
                    running_loss_kl = 0
                    running_loss_dcs_fea = 0

            file_name = '/netG_{}_{}_DCS_fea_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'DCS_img_match_plus_fea':
        if args.model_type == 'resnet50':
            target_layer = model.module._modules.get('maxpool')         # 4: 64*56*56
        elif args.model_type == 'vgg19_bn':
            target_layer = model.module._modules.get('features')[12]
        elif args.model_type == 'densenet121':
            target_layer = model.module._modules.get('features')[5]
        else:
            assert False, 'Please provide correct target model names: {}'.format(model_names)

        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            running_loss_dcs_fea = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()

                adv = netG(img)

                loss_dcs = torch.tensor(0.0, device='cuda')

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_freq = torch.abs(img_match_freq)
                img_match_freq = img_match_freq.mean(dim=0)

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_freq = torch.abs(adv_freq)
                adv_freq = adv_freq.mean(dim=0)

                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                h = Layer_out(target_layer)
                adv_out = model(normalize(adv))
                adv_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                img_match_out = model(normalize(img_match))
                img_match_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                # h = Layer_out(target_layer)
                #
                # adv_out = model(normalize(adv))
                # adv_fea = h.features.cuda()
                # img_match_out = model(normalize(img_match))
                # img_match_fea = h.features.detach().cuda()
                # h.remove()

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                adv_fea_freq = torch.fft.fft2(adv_fea, norm='ortho')
                adv_fea_freq = torch.fft.fftshift(adv_fea_freq)
                adv_fea_freq = torch.abs(adv_fea_freq)
                adv_fea_freq = adv_fea_freq.mean(dim=0)

                img_match_fea_freq = torch.fft.fft2(img_match_fea, norm='ortho')
                img_match_fea_freq = torch.fft.fftshift(img_match_fea_freq)
                img_match_fea_freq = torch.abs(img_match_fea_freq)
                img_match_fea_freq = img_match_fea_freq.mean(dim=0)

                loss_dcs_fea = torch.tensor(0.0, device='cuda')
                loss_dcs_fea += args.beta * (1.0 / args.batch_size) * criterion_L1(adv_fea_freq, img_match_fea_freq)

                loss = loss_kl + loss_dcs_fea + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()
                running_loss_dcs_fea += loss_dcs_fea.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f} \t loss_dcs_fea: {4:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10, running_loss_dcs_fea / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f} \t loss_dcs_fea: {4:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10, running_loss_dcs_fea / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0
                    running_loss_dcs_fea = 0

            file_name = '/netG_{}_{}_DCS_img_match_plus_fea_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'DCS_img':
        # My method
        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()

                adv = netG(img)

                loss_dcs = torch.tensor(0.0).cuda()

                # like img

                # DCS_img
                img_freq = torch.fft.fft2(img, norm='ortho')
                img_freq = torch.fft.fftshift(img_freq)
                img_freq = torch.abs(img_freq)
                img_freq = img_freq.mean(dim=0)

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_freq = torch.abs(adv_freq)
                # adv_amp = torch.sqrt(adv_freq.real ** 2 + adv_freq.imag ** 2)
                # adv_phase = torch.atan2(adv_freq.imag, adv_freq.real)
                adv_freq = adv_freq.mean(dim=0)

                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)

                if args.gs:
                    adv = kernel(adv)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_out = model(normalize(adv))
                img_match_out = model(normalize(img_match))

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                loss = loss_kl + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0

            file_name = '/netG_{}_{}_DCS_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'amp_fix':
        # My method
        for epoch in range(args.epochs):
            running_loss_kl = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()
                # img's amplitude
                img_freq = torch.fft.fft2(img, norm='ortho')
                img_amp = torch.abs(img_freq)

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()

                adv = netG(img)

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_phase = torch.angle(adv_freq)
                mix_freq = img_amp * torch.cos(adv_phase) + img_amp * torch.sin(adv_phase) * 1j
                mix = torch.fft.ifft2(mix_freq, norm='ortho').real

                mix = torch.min(torch.max(mix, img - eps), img + eps)
                mix = torch.clamp(mix, 0.0, 1.0)

                mix_out = model(normalize(mix))
                img_match_out = model(normalize(img_match))

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(mix_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(mix_out, dim=1))

                loss = loss_kl
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f}'.format(
                        epoch, i, running_loss_kl / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f}'.format(
                        epoch, i, running_loss_kl / 10))
                    running_loss_kl = 0

            file_name = '/netG_{}_{}_amp_fix_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'amp_attn':
        # target samples are used to calculate the amplitude attention
        # img_match_amp = 0
        # for img, _ in train_loader_match:
        #     img_match = img.cuda()
        #     img_match.requires_grad = True
        #
        #     target_label = torch.LongTensor(img_match.size(0))
        #     target_label.fill_(args.match_target)
        #     target_label = target_label.cuda()
        #
        #     img_match_out = model(normalize(img_match))
        #     loss = criterion_ce(img_match_out, target_label)
        #     loss.backward()
        #
        #     img_match_grad = img_match.grad
        #     img_match_freq = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
        #     img_match_amp += torch.abs(img_match_freq).mean(dim=0)
        #
        # img_match_amp /= len(train_loader_match)
        # img_match_amp = (img_match_amp - img_match_amp.min()) / (img_match_amp.max() - img_match_amp.min())
        # img_match_amp = img_match_amp.mean(dim=0)
        # # plt.imshow(img_match_amp.detach().cpu().numpy(), vmin=0, vmax=1, cmap='jet')
        # # plt.colorbar()
        # # plt.show()
        def get_mask(image, radius1, radius2):
            # rows, cols = image.shape[0:2]
            # crow, ccol = int(rows / 2), int(cols / 2)
            #
            # mask = torch.ones((rows, cols), dtype=torch.uint8).to(image.device)
            # center = [crow, ccol]
            #
            # x, y = torch.meshgrid(torch.arange(rows), torch.arange(cols))
            # mask_area = (torch.abs(x - center[0]) + torch.abs(y - center[1]) > radius1) & (
            #             torch.abs(x - center[0]) + torch.abs(y - center[1]) <= radius2)
            # mask[mask_area] = 0
            # return mask

            height, width = image.shape[-2:]
            mask = torch.ones((height, width), dtype=torch.uint8).to(image.device)
            center_height, is_even_height = divmod(height, 2)
            center_width, is_even_width = divmod(width, 2)

            if is_even_height == 0:
                center_height -= 0.5
            if is_even_width == 0:
                center_width -= 0.5

            y, x = torch.meshgrid(torch.arange(-center_height, height - center_height, device=image.device),
                                  torch.arange(-center_width, width - center_width, device=image.device))
            mask_area = (torch.sqrt(x ** 2 + y ** 2) > radius1) & (torch.sqrt(x ** 2 + y ** 2) <= radius2)
            mask[mask_area] = 0
            return mask

        for epoch in range(args.epochs):
            running_loss_kl = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()
                img.requires_grad = True

                img_freq = torch.fft.fftshift(torch.fft.fft2(img, norm='ortho'))
                img_amp = torch.abs(img_freq)
                img_pha = torch.angle(img_freq)

                target_label = torch.LongTensor(img.size(0))
                target_label.fill_(args.match_target)
                target_label = target_label.cuda()

                out = model(normalize(img))
                loss_ce_img = criterion_ce(out, target_label)
                loss_ce_img.backward(retain_graph=True)

                img_grad = img.grad
                img_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_grad, norm='ortho'))
                img_grad_amp = torch.abs(img_grad_freq)
                img_grad_amp = img_grad_amp.view(img_grad_amp.size(0), 3, -1)
                img_grad_amp = (img_grad_amp - img_grad_amp.min(dim=2, keepdim=True)[0]) / (img_grad_amp.max(dim=2, keepdim=True)[0] - img_grad_amp.min(dim=2, keepdim=True)[0])
                img_grad_amp = img_grad_amp.view(img_grad_freq.size())
                # img_grad_amp = img_grad_amp.mean(dim=1)

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()
                img_match.requires_grad = True

                img_match_out = model(normalize(img_match))
                loss_ce_img_match = criterion_ce(img_match_out, target_label)
                loss_ce_img_match.backward(retain_graph=True)

                # img_match_grad = img_match.grad
                # img_match_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
                # img_match_grad_amp = torch.abs(img_match_grad_freq)
                # img_match_grad_amp = img_match_grad_amp.mean(dim=0)
                # img_match_grad_amp = img_match_grad_amp.view(3, -1)
                # img_match_grad_amp = (img_match_grad_amp - img_match_grad_amp.min(dim=1, keepdim=True)[0]) / (img_match_grad_amp.max(dim=1, keepdim=True)[0] - img_match_grad_amp.min(dim=1, keepdim=True)[0])
                # img_match_grad_amp = img_match_grad_amp.view(img_match_grad_freq.size()[1:4])
                # img_match_grad_amp = img_match_grad_amp.unsqueeze(0)
                # # img_match_grad_amp = img_match_grad_amp.mean(dim=1,keepdim=True)

                img_match_grad = img_match.grad
                img_match_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
                img_match_grad_amp = torch.abs(img_match_grad_freq)
                img_match_grad_amp = img_match_grad_amp.view(img_match_grad_amp.size(0), 3, -1)
                img_match_grad_amp = (img_match_grad_amp - img_match_grad_amp.min(dim=2, keepdim=True)[0]) / (img_match_grad_amp.max(dim=2, keepdim=True)[0] - img_match_grad_amp.min(dim=2, keepdim=True)[0])
                img_match_grad_amp = img_match_grad_amp.view(img_match_grad_freq.size())
                # img_match_grad_amp = img_match_grad_amp.mean(dim=1,keepdim=True)

                # randomization
                # percent_range = [4, 20, 51, 92, 210]
                # mask_players = []
                # for i in range(len(percent_range) - 1):
                #     mask_players.append(1 - get_mask(img, percent_range[i], percent_range[i + 1]))


                # diff of attention
                attn_amp = (img_match_grad_amp - img_grad_amp)
                attn_amp /= 10
                # attn_amp = torch.clamp(attn_amp, -0.1, 0.1)
                # attn_amp = 1 + attn_amp
                attn_amp = 1 - attn_amp

                # attn_amp = torch.exp(-attn_amp)
                # attn_amp = attn_amp.view(attn_amp.size(0), 3, -1)
                # attn_amp = F.sigmoid(attn_amp)
                # attn_amp = attn_amp.view(attn_amp.size(0), 3, img_match_grad_amp.size(2), img_match_grad_amp.size(3))

                # img_amp = img_amp * img_match_grad_amp
                # rho = 0.5
                # mask = (torch.rand_like(img_amp) * 2 * rho + 1 - rho)
                # img_amp = img_amp * mask

                img_amp = img_amp * attn_amp
                img_freq = img_amp * torch.cos(img_pha) + img_amp * torch.sin(img_pha) * 1j
                img_temp = torch.fft.ifft2(torch.fft.ifftshift(img_freq), norm='ortho').real
                img_temp = torch.clamp(img_temp, 0.0, 1.0)
                netG.train()

                adv = netG(img_temp)

                # noise = adv - img
                #
                # noise_freq = torch.fft.fftshift(torch.fft.fft2(noise, norm='ortho'))
                # noise_amp = torch.abs(noise_freq)
                # noise_pha = torch.angle(noise_freq)
                # noise_amp = noise_amp * attn_amp
                # noise_freq = noise_amp * torch.cos(noise_pha) + noise_amp * torch.sin(noise_pha) * 1j
                # noise = torch.fft.ifftshift(torch.fft.ifft2(noise_freq, norm='ortho')).real

                # noise = torch.clamp(noise, -eps, eps)
                # adv = img + noise

                adv = torch.min(torch.max(adv, img_temp - eps), img_temp + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_out = model(normalize(adv))

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                loss_dcs = 0.0
                # adv_freq = torch.fft.fft2(adv, norm='ortho')
                # adv_freq = torch.fft.fftshift(adv_freq)
                # adv_amp = torch.abs(adv_freq)
                # adv_amp = adv_amp.mean(dim=0)
                #
                # loss_dcs = args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_amp, img_match_amp)

                loss = loss_kl + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f}'.format(
                        epoch, i, running_loss_kl / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f}'.format(
                        epoch, i, running_loss_kl / 10))
                    running_loss_kl = 0

            file_name = '/netG_{}_{}_amp_attn_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'DCS_img_match_plus_fea_argu_gs':
        if args.model_type == 'resnet50':
            target_layer = model.module._modules.get('maxpool')         # 4: 64*56*56
        elif args.model_type == 'vgg19_bn':
            target_layer = model.module._modules.get('features')[12]
        elif args.model_type == 'densenet121':
            target_layer = model.module._modules.get('features')[5]
        else:
            assert False, 'Please provide correct target model names: {}'.format(model_names)

        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            running_loss_dcs_fea = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs[0].cuda()
                img_rot = rotation(img)[0]
                img_aug = imgs[1].cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()

                adv = netG(img)
                adv_rot = netG(img_rot)
                adv_aug = netG(img_aug)

                loss_dcs = torch.tensor(0.0, device='cuda')

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_freq = torch.abs(img_match_freq)
                img_match_freq = img_match_freq.mean(dim=0)

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_freq = torch.abs(adv_freq)
                adv_freq = adv_freq.mean(dim=0)

                adv_rot_freq = torch.fft.fft2(adv_rot, norm='ortho')
                adv_rot_freq = torch.fft.fftshift(adv_rot_freq)
                adv_rot_freq = torch.abs(adv_rot_freq)
                adv_rot_freq = adv_rot_freq.mean(dim=0)

                adv_aug_freq = torch.fft.fft2(adv_aug, norm='ortho')
                adv_aug_freq = torch.fft.fftshift(adv_aug_freq)
                adv_aug_freq = torch.abs(adv_aug_freq)
                adv_aug_freq = adv_aug_freq.mean(dim=0)


                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_rot_freq, img_match_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_aug_freq, img_match_freq)

                # gs
                adv = kernel(adv)
                adv_rot = kernel(adv_rot)
                adv_aug = kernel(adv_aug)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)
                adv_rot = torch.min(torch.max(adv_rot, img_rot - eps), img_rot + eps)
                adv_rot = torch.clamp(adv_rot, 0.0, 1.0)
                adv_aug = torch.min(torch.max(adv_aug, img_aug - eps), img_aug + eps)
                adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

                h = Layer_out(target_layer)
                adv_out = model(normalize(adv))
                adv_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                adv_rot_out = model(normalize(adv_rot))
                adv_rot_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                adv_aug_out = model(normalize(adv_aug))
                adv_aug_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                img_match_out = model(normalize(img_match))
                img_match_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                # h = Layer_out(target_layer)
                #
                # adv_out = model(normalize(adv))
                # adv_fea = h.features.cuda()
                # img_match_out = model(normalize(img_match))
                # img_match_fea = h.features.detach().cuda()
                # h.remove()

                loss_kl = 0.0
                for out in [adv_out, adv_rot_out, adv_aug_out]:
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(out, dim=1),
                                                                      F.softmax(img_match_out, dim=1))
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                      F.softmax(out, dim=1))

                adv_fea_freq = torch.fft.fft2(adv_fea, norm='ortho')
                adv_fea_freq = torch.fft.fftshift(adv_fea_freq)
                adv_fea_freq = torch.abs(adv_fea_freq)
                adv_fea_freq = adv_fea_freq.mean(dim=0)

                adv_rot_fea_freq = torch.fft.fft2(adv_rot_fea, norm='ortho')
                adv_rot_fea_freq = torch.fft.fftshift(adv_rot_fea_freq)
                adv_rot_fea_freq = torch.abs(adv_rot_fea_freq)
                adv_rot_fea_freq = adv_rot_fea_freq.mean(dim=0)

                adv_aug_fea_freq = torch.fft.fft2(adv_aug_fea, norm='ortho')
                adv_aug_fea_freq = torch.fft.fftshift(adv_aug_fea_freq)
                adv_aug_fea_freq = torch.abs(adv_aug_fea_freq)
                adv_aug_fea_freq = adv_aug_fea_freq.mean(dim=0)

                img_match_fea_freq = torch.fft.fft2(img_match_fea, norm='ortho')
                img_match_fea_freq = torch.fft.fftshift(img_match_fea_freq)
                img_match_fea_freq = torch.abs(img_match_fea_freq)
                img_match_fea_freq = img_match_fea_freq.mean(dim=0)

                loss_dcs_fea = torch.tensor(0.0, device='cuda')
                for fea_freq in [adv_fea_freq, adv_rot_fea_freq, adv_aug_fea_freq]:
                    loss_dcs_fea += args.beta * (1.0 / args.batch_size) * criterion_L1(fea_freq, img_match_fea_freq)
                # loss_dcs_fea += args.beta * (1.0 / args.batch_size) * criterion_L1(adv_fea_freq, img_match_fea_freq)

                loss = loss_kl + loss_dcs_fea + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()
                running_loss_dcs_fea += loss_dcs_fea.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f} \t loss_dcs_fea: {4:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10, running_loss_dcs_fea / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f} \t loss_dcs_fea: {4:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10, running_loss_dcs_fea / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0
                    running_loss_dcs_fea = 0

            file_name = '/netG_{}_{}_DCS_img_match_plus_fea_argu_gs_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'DCS_img_match_attn_mask_plus_fea_argu_gs':
        if args.model_type == 'resnet50':
            target_layer = model.module._modules.get('layer4')
        elif args.model_type == 'vgg19_bn':
            target_layer = model.module._modules.get('features')[23]
        elif args.model_type == 'densenet121':
            target_layer = model.module._modules.get('features')[8]
        else:
            assert False, 'Please provide correct target model names: {}'.format(model_names)

        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            running_loss_dcs_fea = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs[0].cuda()
                img_rot = rotation(img)[0]
                img_aug = imgs[1].cuda()

                target_label = torch.LongTensor(img.size(0))
                target_label.fill_(args.match_target)
                target_label = target_label.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()
                img_match_temp = img_match.clone()
                img_match_temp.requires_grad = True

                img_match_temp_out = model(normalize(img_match_temp))
                loss_ce_img_match = criterion_ce(img_match_temp_out, target_label)
                loss_ce_img_match.backward()

                img_match_grad = img_match_temp.grad
                img_match_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
                img_match_grad_amp = torch.abs(img_match_grad_freq)
                img_match_grad_amp = img_match_grad_amp.mean(dim=[0,1])
                # normalize
                img_match_grad_amp = (img_match_grad_amp - img_match_grad_amp.min()) / (img_match_grad_amp.max() - img_match_grad_amp.min())
                attn_mask = img_match_grad_amp.clone().detach()

                netG.train()

                adv = netG(img)
                adv_rot = netG(img_rot)
                adv_aug = netG(img_aug)

                loss_dcs = torch.tensor(0.0, device='cuda')

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_amp = torch.abs(img_match_freq)
                img_match_amp = img_match_amp.mean(dim=0)
                img_match_amp = img_match_amp * attn_mask

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_amp = torch.abs(adv_freq)
                adv_amp = adv_amp.mean(dim=0)
                adv_amp = adv_amp * attn_mask

                adv_rot_freq = torch.fft.fft2(adv_rot, norm='ortho')
                adv_rot_freq = torch.fft.fftshift(adv_rot_freq)
                adv_rot_amp = torch.abs(adv_rot_freq)
                adv_rot_amp = adv_rot_amp.mean(dim=0)
                adv_rot_amp = adv_rot_amp * attn_mask

                adv_aug_freq = torch.fft.fft2(adv_aug, norm='ortho')
                adv_aug_freq = torch.fft.fftshift(adv_aug_freq)
                adv_aug_amp = torch.abs(adv_aug_freq)
                adv_aug_amp = adv_aug_amp.mean(dim=0)
                adv_aug_amp = adv_aug_amp * attn_mask


                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_amp, img_match_amp)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_rot_amp, img_match_amp)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_aug_amp, img_match_amp)

                # gs
                adv = kernel(adv)
                adv_rot = kernel(adv_rot)
                adv_aug = kernel(adv_aug)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)
                adv_rot = torch.min(torch.max(adv_rot, img_rot - eps), img_rot + eps)
                adv_rot = torch.clamp(adv_rot, 0.0, 1.0)
                adv_aug = torch.min(torch.max(adv_aug, img_aug - eps), img_aug + eps)
                adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

                h = Layer_out(target_layer)
                adv_out = model(normalize(adv))
                adv_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                adv_rot_out = model(normalize(adv_rot))
                adv_rot_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                adv_aug_out = model(normalize(adv_aug))
                adv_aug_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                img_match_out = model(normalize(img_match))
                img_match_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                # h = Layer_out(target_layer)
                #
                # adv_out = model(normalize(adv))
                # adv_fea = h.features.cuda()
                # img_match_out = model(normalize(img_match))
                # img_match_fea = h.features.detach().cuda()
                # h.remove()

                loss_kl = 0.0
                for out in [adv_out, adv_rot_out, adv_aug_out]:
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(out, dim=1),
                                                                      F.softmax(img_match_out, dim=1))
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                      F.softmax(out, dim=1))

                adv_fea_freq = torch.fft.fft2(adv_fea, norm='ortho')
                adv_fea_freq = torch.fft.fftshift(adv_fea_freq)
                adv_fea_amp = torch.abs(adv_fea_freq)
                adv_fea_amp = adv_fea_amp.mean(dim=0)

                adv_rot_fea_freq = torch.fft.fft2(adv_rot_fea, norm='ortho')
                adv_rot_fea_freq = torch.fft.fftshift(adv_rot_fea_freq)
                adv_rot_fea_amp = torch.abs(adv_rot_fea_freq)
                adv_rot_fea_amp = adv_rot_fea_amp.mean(dim=0)

                adv_aug_fea_freq = torch.fft.fft2(adv_aug_fea, norm='ortho')
                adv_aug_fea_freq = torch.fft.fftshift(adv_aug_fea_freq)
                adv_aug_fea_amp = torch.abs(adv_aug_fea_freq)
                adv_aug_fea_amp = adv_aug_fea_amp.mean(dim=0)

                img_match_fea_freq = torch.fft.fft2(img_match_fea, norm='ortho')
                img_match_fea_freq = torch.fft.fftshift(img_match_fea_freq)
                img_match_fea_amp = torch.abs(img_match_fea_freq)
                img_match_fea_amp = img_match_fea_amp.mean(dim=0)

                loss_dcs_fea = torch.tensor(0.0, device='cuda')
                for fea_amp in [adv_fea_amp, adv_rot_fea_amp, adv_aug_fea_amp]:
                    loss_dcs_fea += args.beta * (1.0 / args.batch_size) * criterion_L1(fea_amp, img_match_fea_amp)
                # loss_dcs_fea += args.beta * (1.0 / args.batch_size) * criterion_L1(adv_fea_freq, img_match_fea_freq)

                loss = loss_kl + loss_dcs_fea + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()
                running_loss_dcs_fea += loss_dcs_fea.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f} \t loss_dcs_fea: {4:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10, running_loss_dcs_fea / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f} \t loss_dcs_fea: {4:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10, running_loss_dcs_fea / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0
                    running_loss_dcs_fea = 0

            file_name = '/netG_{}_{}_DCS_img_match_attn_mask_plus_fea_argu_gs_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'DCS_img_match_attn_mask_pro_plus_fea':
        attn_mask_dict = {
            'resnet50': {
                '24': 0.772123954,
                '99': 0.948461476,
                '245': 0.935379248,
                '344': 0.831811616,
                '471': 1.001062931,
                '555': 1.125071543,
                '661': 0.979531765,
                '701': 1.262162383,
                '802': 1.004333488,
                '919': 1.140061595
            },
            'densenet121': {
                '24': 0.800130102,
                '99': 1.044072207,
                '245': 1.005691982,
                '344': 0.913644495,
                '471': 1.025532607,
                '555': 1.102943568,
                '661': 1.055781428,
                '701': 1.025857863,
                '802': 1.011871849,
                '919': 1.014473898
            },
            'vgg19_bn': {
                '24': 0.680206847,
                '99': 0.923321712,
                '245': 0.969885579,
                '344': 0.836043709,
                '471': 1.038444439,
                '555': 1.184687741,
                '661': 1.027914921,
                '701': 1.295832651,
                '802': 0.980181108,
                '919': 1.063481293
            },
        }

        if args.model_type == 'resnet50':
            target_layer = model.module._modules.get('layer4')
        elif args.model_type == 'vgg19_bn':
            target_layer = model.module._modules.get('features')[23]
        elif args.model_type == 'densenet121':
            target_layer = model.module._modules.get('features')[8]
        else:
            assert False, 'Please provide correct target model names: {}'.format(model_names)

        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            running_loss_dcs_fea = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()
                # img_match_temp = img_match.clone()
                # img_match_temp.requires_grad = True
                #
                # img_match_temp_out = model(normalize(img_match_temp))
                # loss_ce_img_match = criterion_ce(img_match_temp_out, target_label)
                # loss_ce_img_match.backward()
                #
                # img_match_grad = img_match_temp.grad
                # img_match_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
                # img_match_grad_amp = torch.abs(img_match_grad_freq)
                # img_match_grad_amp = img_match_grad_amp.mean(dim=[0,1])
                # # normalize
                # img_match_grad_amp = (img_match_grad_amp - img_match_grad_amp.min()) / (img_match_grad_amp.max() - img_match_grad_amp.min())
                # attn_mask = img_match_grad_amp.clone().detach()

                attn_mask = torch.ones(img_match.shape[1:], device='cuda') * attn_mask_dict['{}'.format(args.model_type)]['{}'.format(args.match_target)]
                attn_mask = torch.clamp(attn_mask, 0.8, 1.2)

                netG.train()

                adv = netG(img)

                loss_dcs = torch.tensor(0.0, device='cuda')

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_amp = torch.abs(img_match_freq)
                img_match_amp = img_match_amp.mean(dim=0)
                # img_match_amp = img_match_amp.mean(dim=1)
                # plt.imshow(img_match_amp.mean(0).detach().cpu().numpy(), vmin=0, vmax=0.5)
                # plt.axis('off')
                # plt.savefig('img_match_amp_mean.png', bbox_inches='tight', pad_inches=0)
                img_match_amp = img_match_amp * attn_mask

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_amp = torch.abs(adv_freq)
                adv_amp = adv_amp.mean(dim=0)
                adv_amp = adv_amp * attn_mask

                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_amp, img_match_amp)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                h = Layer_out(target_layer)
                adv_out = model(normalize(adv))
                adv_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                img_match_out = model(normalize(img_match))
                img_match_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                # h = Layer_out(target_layer)
                #
                # adv_out = model(normalize(adv))
                # adv_fea = h.features.cuda()
                # img_match_out = model(normalize(img_match))
                # img_match_fea = h.features.detach().cuda()
                # h.remove()

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                adv_fea_freq = torch.fft.fft2(adv_fea, norm='ortho')
                adv_fea_freq = torch.fft.fftshift(adv_fea_freq)
                adv_fea_amp = torch.abs(adv_fea_freq)
                adv_fea_amp = adv_fea_amp.mean(dim=0)

                img_match_fea_freq = torch.fft.fft2(img_match_fea, norm='ortho')
                img_match_fea_freq = torch.fft.fftshift(img_match_fea_freq)
                img_match_fea_amp = torch.abs(img_match_fea_freq)
                img_match_fea_amp = img_match_fea_amp.mean(dim=0)

                loss_dcs_fea = torch.tensor(0.0, device='cuda')
                loss_dcs_fea += args.beta * (1.0 / args.batch_size) * criterion_L1(adv_fea_amp, img_match_fea_amp)
                # loss_dcs_fea += args.beta * (1.0 / args.batch_size) * criterion_L1(adv_fea_freq, img_match_fea_freq)

                loss = loss_kl + loss_dcs_fea + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()
                running_loss_dcs_fea += loss_dcs_fea.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f} \t loss_dcs_fea: {4:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10, running_loss_dcs_fea / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f} \t loss_dcs_fea: {4:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10, running_loss_dcs_fea / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0
                    running_loss_dcs_fea = 0

            file_name = '/netG_{}_{}_DCS_img_match_attn_mask_plus_fea_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'DCS_img_match_attn_mask_pro_plus_argu_gs':
        attn_mask_dict = {
            'resnet50': {
                '24': 0.772123954,
                '99': 0.948461476,
                '245': 0.935379248,
                '344': 0.831811616,
                '471': 1.001062931,
                '555': 1.125071543,
                '661': 0.979531765,
                '701': 1.262162383,
                '802': 1.004333488,
                '919': 1.140061595
            },
            'densenet121': {
                '24': 0.800130102,
                '99': 1.044072207,
                '245': 1.005691982,
                '344': 0.913644495,
                '471': 1.025532607,
                '555': 1.102943568,
                '661': 1.055781428,
                '701': 1.025857863,
                '802': 1.011871849,
                '919': 1.014473898
            },
            'vgg19_bn': {
                '24': 0.680206847,
                '99': 0.923321712,
                '245': 0.969885579,
                '344': 0.836043709,
                '471': 1.038444439,
                '555': 1.184687741,
                '661': 1.027914921,
                '701': 1.295832651,
                '802': 0.980181108,
                '919': 1.063481293
            },
        }

        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs[0].cuda()
                img_rot = rotation(img)[0]
                img_aug = imgs[1].cuda()

                # target_label = torch.LongTensor(img.size(0))
                # target_label.fill_(args.match_target)
                # target_label = target_label.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()
                # img_match_temp = img_match.clone()
                # img_match_temp.requires_grad = True
                #
                # img_match_temp_out = model(normalize(img_match_temp))
                # loss_ce_img_match = criterion_ce(img_match_temp_out, target_label)
                # loss_ce_img_match.backward()
                #
                # img_match_grad = img_match_temp.grad
                # img_match_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
                # img_match_grad_amp = torch.abs(img_match_grad_freq)
                # img_match_grad_amp = img_match_grad_amp.mean(dim=[0,1])
                # # normalize
                # img_match_grad_amp = (img_match_grad_amp - img_match_grad_amp.min()) / (img_match_grad_amp.max() - img_match_grad_amp.min())
                # attn_mask = img_match_grad_amp.clone().detach()

                attn_mask = torch.ones(img_match.shape[1:], device='cuda') * \
                            attn_mask_dict['{}'.format(args.model_type)]['{}'.format(args.match_target)]
                attn_mask = torch.clamp(attn_mask, 0.8, 1.2)

                netG.train()

                adv = netG(img)
                adv_rot = netG(img_rot)
                adv_aug = netG(img_aug)

                loss_dcs = torch.tensor(0.0, device='cuda')

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_amp = torch.abs(img_match_freq)
                img_match_amp = img_match_amp.mean(dim=0)
                # img_match_amp = img_match_amp.mean(dim=1)
                # plt.imshow(img_match_amp.mean(0).detach().cpu().numpy(), vmin=0, vmax=0.5)
                # plt.axis('off')
                # plt.savefig('img_match_amp_mean.png', bbox_inches='tight', pad_inches=0)
                img_match_amp = img_match_amp * attn_mask

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_amp = torch.abs(adv_freq)
                adv_amp = adv_amp.mean(dim=0)
                adv_amp = adv_amp * attn_mask

                adv_rot_freq = torch.fft.fft2(adv_rot, norm='ortho')
                adv_rot_freq = torch.fft.fftshift(adv_rot_freq)
                adv_rot_amp = torch.abs(adv_rot_freq)
                adv_rot_amp = adv_rot_amp.mean(dim=0)
                adv_rot_amp = adv_rot_amp * attn_mask

                adv_aug_freq = torch.fft.fft2(adv_aug, norm='ortho')
                adv_aug_freq = torch.fft.fftshift(adv_aug_freq)
                adv_aug_amp = torch.abs(adv_aug_freq)
                adv_aug_amp = adv_aug_amp.mean(dim=0)
                adv_aug_amp = adv_aug_amp * attn_mask

                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_amp, img_match_amp)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_rot_amp, img_match_amp)
                loss_dcs += args.alpha * (1.0 / args.batch_size) * criterion_L1(adv_aug_amp, img_match_amp)

                # gs
                adv = kernel(adv)
                adv_rot = kernel(adv_rot)
                adv_aug = kernel(adv_aug)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)
                adv_rot = torch.min(torch.max(adv_rot, img_rot - eps), img_rot + eps)
                adv_rot = torch.clamp(adv_rot, 0.0, 1.0)
                adv_aug = torch.min(torch.max(adv_aug, img_aug - eps), img_aug + eps)
                adv_aug = torch.clamp(adv_aug, 0.0, 1.0)


                adv_out = model(normalize(adv))
                adv_rot_out = model(normalize(adv_rot))
                adv_aug_out = model(normalize(adv_aug))
                img_match_out = model(normalize(img_match))


                loss_kl = 0.0
                for out in [adv_out, adv_rot_out, adv_aug_out]:
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(out, dim=1),
                                                                      F.softmax(img_match_out, dim=1))
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                      F.softmax(out, dim=1))

                loss = loss_kl  + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()

                if i % 10 == 9:
                    print(
                        'Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f}'.format(
                            epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    logging.info(
                        'Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f}'.format(
                            epoch, i, running_loss_kl / 10, running_loss_dcs / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0

            file_name = '/netG_{}_{}_DCS_img_match_attn_mask_plus_argu_gs_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'DCS_img_match_attn_mask_pro_plus_fea_argu_gs':
        attn_mask_dict = {
            'resnet50': {
                '24': 0.772123954,
                '99': 0.948461476,
                '245': 0.935379248,
                '344': 0.831811616,
                '471': 1.001062931,
                '555': 1.125071543,
                '661': 0.979531765,
                '701': 1.262162383,
                '802': 1.004333488,
                '919': 1.140061595
            },
            'densenet121': {
                '24': 0.800130102,
                '99': 1.044072207,
                '245': 1.005691982,
                '344': 0.913644495,
                '471': 1.025532607,
                '555': 1.102943568,
                '661': 1.055781428,
                '701': 1.025857863,
                '802': 1.011871849,
                '919': 1.014473898
            },
            'vgg19_bn': {
                '24': 0.680206847,
                '99': 0.923321712,
                '245': 0.969885579,
                '344': 0.836043709,
                '471': 1.038444439,
                '555': 1.184687741,
                '661': 1.027914921,
                '701': 1.295832651,
                '802': 0.980181108,
                '919': 1.063481293
            },
        }

        if args.model_type == 'resnet50':
            target_layer = model.module._modules.get('layer4')
        elif args.model_type == 'vgg19_bn':
            target_layer = model.module._modules.get('features')[23]
        elif args.model_type == 'densenet121':
            target_layer = model.module._modules.get('features')[8]
        else:
            assert False, 'Please provide correct target model names: {}'.format(model_names)

        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_dcs = 0
            running_loss_dcs_fea = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs[0].cuda()
                img_rot = rotation(img)[0]
                img_aug = imgs[1].cuda()

                # target_label = torch.LongTensor(img.size(0))
                # target_label.fill_(args.match_target)
                # target_label = target_label.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()
                # img_match_temp = img_match.clone()
                # img_match_temp.requires_grad = True
                #
                # img_match_temp_out = model(normalize(img_match_temp))
                # loss_ce_img_match = criterion_ce(img_match_temp_out, target_label)
                # loss_ce_img_match.backward()
                #
                # img_match_grad = img_match_temp.grad
                # img_match_grad_freq = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
                # img_match_grad_amp = torch.abs(img_match_grad_freq)
                # img_match_grad_amp = img_match_grad_amp.mean(dim=[0,1])
                # # normalize
                # img_match_grad_amp = (img_match_grad_amp - img_match_grad_amp.min()) / (img_match_grad_amp.max() - img_match_grad_amp.min())
                # attn_mask = img_match_grad_amp.clone().detach()

                attn_mask = torch.ones(img_match.shape[1:], device='cuda') * attn_mask_dict['{}'.format(args.model_type)]['{}'.format(args.match_target)]
                attn_mask = torch.clamp(attn_mask, 0.8, 1.2)

                netG.train()

                adv = netG(img)
                adv_rot = netG(img_rot)
                adv_aug = netG(img_aug)

                loss_dcs = torch.tensor(0.0, device='cuda')

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_amp = torch.abs(img_match_freq)
                img_match_amp = img_match_amp.mean(dim=0)
                # img_match_amp = img_match_amp.mean(dim=1)
                # plt.imshow(img_match_amp.mean(0).detach().cpu().numpy(), vmin=0, vmax=0.5)
                # plt.axis('off')
                # plt.savefig('img_match_amp_mean.png', bbox_inches='tight', pad_inches=0)
                img_match_amp = img_match_amp * attn_mask

                adv_freq = torch.fft.fft2(adv, norm='ortho')
                adv_freq = torch.fft.fftshift(adv_freq)
                adv_amp = torch.abs(adv_freq)
                adv_amp = adv_amp.mean(dim=0)
                adv_amp = adv_amp * attn_mask

                adv_rot_freq = torch.fft.fft2(adv_rot, norm='ortho')
                adv_rot_freq = torch.fft.fftshift(adv_rot_freq)
                adv_rot_amp = torch.abs(adv_rot_freq)
                adv_rot_amp = adv_rot_amp.mean(dim=0)
                adv_rot_amp = adv_rot_amp * attn_mask

                adv_aug_freq = torch.fft.fft2(adv_aug, norm='ortho')
                adv_aug_freq = torch.fft.fftshift(adv_aug_freq)
                adv_aug_amp = torch.abs(adv_aug_freq)
                adv_aug_amp = adv_aug_amp.mean(dim=0)
                adv_aug_amp = adv_aug_amp * attn_mask


                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                loss_dcs += args.alpha / 3 * (1.0 / args.batch_size) * criterion_L1(adv_amp, img_match_amp)
                loss_dcs += args.alpha / 3 * (1.0 / args.batch_size) * criterion_L1(adv_rot_amp, img_match_amp)
                loss_dcs += args.alpha / 3 * (1.0 / args.batch_size) * criterion_L1(adv_aug_amp, img_match_amp)

                # gs
                adv = kernel(adv)
                adv_rot = kernel(adv_rot)
                adv_aug = kernel(adv_aug)

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)
                adv_rot = torch.min(torch.max(adv_rot, img_rot - eps), img_rot + eps)
                adv_rot = torch.clamp(adv_rot, 0.0, 1.0)
                adv_aug = torch.min(torch.max(adv_aug, img_aug - eps), img_aug + eps)
                adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

                h = Layer_out(target_layer)
                adv_out = model(normalize(adv))
                adv_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                adv_rot_out = model(normalize(adv_rot))
                adv_rot_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                adv_aug_out = model(normalize(adv_aug))
                adv_aug_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                h = Layer_out(target_layer)
                img_match_out = model(normalize(img_match))
                img_match_fea = torch.cat(h.features, dim=0).cuda()
                h.remove()

                # h = Layer_out(target_layer)
                #
                # adv_out = model(normalize(adv))
                # adv_fea = h.features.cuda()
                # img_match_out = model(normalize(img_match))
                # img_match_fea = h.features.detach().cuda()
                # h.remove()

                loss_kl = 0.0
                for out in [adv_out, adv_rot_out, adv_aug_out]:
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(out, dim=1),
                                                                      F.softmax(img_match_out, dim=1))
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                      F.softmax(out, dim=1))

                adv_fea_freq = torch.fft.fft2(adv_fea, norm='ortho')
                adv_fea_freq = torch.fft.fftshift(adv_fea_freq)
                adv_fea_amp = torch.abs(adv_fea_freq)
                adv_fea_amp = adv_fea_amp.mean(dim=0)

                adv_rot_fea_freq = torch.fft.fft2(adv_rot_fea, norm='ortho')
                adv_rot_fea_freq = torch.fft.fftshift(adv_rot_fea_freq)
                adv_rot_fea_amp = torch.abs(adv_rot_fea_freq)
                adv_rot_fea_amp = adv_rot_fea_amp.mean(dim=0)

                adv_aug_fea_freq = torch.fft.fft2(adv_aug_fea, norm='ortho')
                adv_aug_fea_freq = torch.fft.fftshift(adv_aug_fea_freq)
                adv_aug_fea_amp = torch.abs(adv_aug_fea_freq)
                adv_aug_fea_amp = adv_aug_fea_amp.mean(dim=0)

                img_match_fea_freq = torch.fft.fft2(img_match_fea, norm='ortho')
                img_match_fea_freq = torch.fft.fftshift(img_match_fea_freq)
                img_match_fea_amp = torch.abs(img_match_fea_freq)
                img_match_fea_amp = img_match_fea_amp.mean(dim=0)

                loss_dcs_fea = torch.tensor(0.0, device='cuda')
                for fea_amp in [adv_fea_amp, adv_rot_fea_amp, adv_aug_fea_amp]:
                    loss_dcs_fea += args.beta * (1.0 / args.batch_size) * criterion_L1(fea_amp, img_match_fea_amp)
                # loss_dcs_fea += args.beta * (1.0 / args.batch_size) * criterion_L1(adv_fea_freq, img_match_fea_freq)

                loss = loss_kl + loss_dcs_fea + loss_dcs
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                running_loss_dcs += loss_dcs.item()
                running_loss_dcs_fea += loss_dcs_fea.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f} \t loss_dcs_fea: {4:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10, running_loss_dcs_fea / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_dcs:{3:.5f} \t loss_dcs_fea: {4:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_dcs / 10, running_loss_dcs_fea / 10))
                    running_loss_kl = 0
                    running_loss_dcs = 0
                    running_loss_dcs_fea = 0

            file_name = '/netG_{}_{}_DCS_img_match_attn_mask_plus_fea_argu_gs_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))


if __name__ == '__main__':
    main()
