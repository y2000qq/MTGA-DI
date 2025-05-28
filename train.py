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
parser.add_argument('--src', default='IN_50k_new', help='Source Domain: imagenet, imagenet_10c, IN_50k, comic_books, etc')
parser.add_argument('--match_target', type=int, default=24, help='Target Domain samples')
parser.add_argument('--model_type', type=str, default='resnet50', help='Model under attack (discrimnator)')
parser.add_argument('--gs', action='store_true', help='Apply gaussian smoothing')
parser.add_argument('--save_dir', type=str, default='pretrained_generators', help='Directory to save generators')
parser.add_argument('--method', default='none', help='TTP, TTAA etc')

parser.add_argument('--match_dataset', default='imagenet', help='Target domain')
parser.add_argument('--batch_size', type=int, default=20, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget during training, eps')

parser.add_argument('--re_noise', action='store_true', help='reweight noise use frequency information')
parser.add_argument('--triplet_loss', action='store_true', help='Use triplet loss')
parser.add_argument('--margin', type=float, default=0, help='Margin for triplet loss')
parser.add_argument('--tar_lfc', action='store_true', help='Use low frequency component of target image')
parser.add_argument('--hfl', action='store_true', help='calculate High frequency loss')
parser.add_argument('--ffl', action='store_true', help='calculate focal frequency loss')

parser.add_argument('--test', action='store_true', help='Test')
# todo
parser.add_argument('--resume', action='store_true', help='Resume training')

args = parser.parse_args()
print(args)

if args.test:
    logfile = os.path.join('train_loss', 'test.log')
elif args.re_noise:
    logfile = os.path.join('train_loss', 'trainloss_{}_t{}_re_noise_plus.log'.format(args.model_type, args.match_target))
elif args.tar_lfc:
    logfile = os.path.join('train_loss', 'trainloss_{}_t{}_tar_lfc.log'.format(args.model_type, args.match_target))
elif args.method == 'TTP' and args.src == 'comic_books':
    logfile = os.path.join('train_loss', 'trainloss_comic_books_{}_ttp_t{}.log'.format(args.model_type, args.match_target))
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
elif args.src == 'Paintings':
    source_path = '/datasets/Paintings/train'
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
    if hostname in ['user-Precision-7920-Tower', 'dell-Precision-7920-Tower', 'dell-Precision-7960-Tower']:  # 3091 or 3090
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
    if args.re_noise:
        if args.model_type == 'inception_v3':
            netG = GeneratorResnet_R(inception=True)
        else:
            netG = GeneratorResnet_R()
    else:
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
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
        return t

    if args.method == 'TTP' or args.method == 'argu':
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

    if args.gs:
        kernel_size = 3
        pad = 2
        sigma = 1
        kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()

    criterion_kl = nn.KLDivLoss(reduction='sum')
    criterion_bce = nn.BCELoss(reduction='sum')
    criterion_ce = nn.CrossEntropyLoss(reduction='sum')
    criterion_L1 = nn.L1Loss(reduction='sum')

    # 'TTP' method
    if args.method == 'TTP':
        for epoch in range(args.epochs):
            running_loss = 0
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
                optimG.zero_grad()

                # Unconstrained Adversaries
                adv = netG(img)
                adv_rot = netG(img_rot)
                adv_aug = netG(img_aug)

                # Smoothing
                if args.gs:
                    adv = kernel(adv)
                    adv_rot = kernel(adv_rot)
                    adv_aug = kernel(adv_aug)

                # Projection
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

                # Loss
                loss_kl = 0.0
                loss_sim = 0.0

                for out in [adv_out, adv_rot_out, adv_aug_out]:
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(out, dim=1),
                                                                      F.softmax(img_match_out, dim=1))
                    # KL divergence is not symmetric
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                      F.softmax(out, dim=1))

                # Neighbourhood similarity
                St = torch.matmul(img_match_out, img_match_out.t())
                norm = torch.matmul(torch.linalg.norm(img_match_out, dim=1, ord=2),
                                    torch.linalg.norm(img_match_out, dim=1, ord=2).t())
                St = St / norm
                for out in [adv_rot_out, adv_aug_out]:
                    Ss = torch.matmul(adv_out, out.t())
                    norm = torch.matmul(torch.linalg.norm(adv_out, dim=1, ord=2),
                                        torch.linalg.norm(out, dim=1, ord=2).t())
                    Ss = Ss / norm
                    loss_sim += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(Ss, dim=1),
                                                                       F.softmax(St, dim=1))
                    loss_sim += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(St, dim=1),
                                                                       F.softmax(Ss, dim=1))

                loss = loss_kl + loss_sim
                loss.backward()
                optimG.step()
                running_loss += loss.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 10))
                    running_loss = 0

            file_name = '/netG_{}_{}_ttp_t{}.pth'
            torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'argu':
        for epoch in range(args.epochs):
            running_loss = 0
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
                optimG.zero_grad()

                # Unconstrained Adversaries
                adv = netG(img)
                adv_rot = netG(img_rot)
                adv_aug = netG(img_aug)

                # Projection
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

                # Loss
                loss_kl = 0.0

                for out in [adv_out, adv_rot_out, adv_aug_out]:
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(out, dim=1),
                                                                      F.softmax(img_match_out, dim=1))
                    # KL divergence is not symmetric
                    loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                      F.softmax(out, dim=1))

                loss = loss_kl
                loss.backward()
                optimG.step()
                running_loss += loss.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 10))
                    running_loss = 0

            file_name = '/netG_{}_{}_argu_t{}.pth'
            torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    # todo : 'TTAA' method 目前不收敛
    elif args.method == 'TTAA':
        if args.model_type == 'resnet50':
            target_layer = model.module._modules.get('maxpool')         # 4: 56*56
        elif args.model_type == 'vgg19_bn':
            target_layer = model.module._modules.get('features')[12]    # layer 26 in vgg19_bn corresponds to layer 17 in vgg19: 256*56*56
        elif args.model_type == 'densenet121':
            target_layer = model.module._modules.get('features')[5]     # layer 5
        else:
            assert False, 'Please provide correct target model names: {}'.format(model_names)

        netD = Feature_Discriminator(128)
        netD = nn.DataParallel(netD).cuda()
        optimD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

        for epoch in range(args.epochs):
            running_loss = 0
            disc_loss = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]

                img_match = img_match.cuda()

                netG.train()
                netD.train()
                optimG.zero_grad()

                noise = netG(img)

                # Random Perturbation Dropping
                mask = get_RPD_mask(noise)

                noise = noise * mask
                adv = img + noise

                # Projection
                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                h = Layer_out(target_layer) # hook

                adv_out = model(normalize(adv))  # (batch_size, num_classes)
                adv_fea = h.features
                img_match_out = model(normalize(img_match.clone().detach()))
                img_match_fea = h.features.detach()
                h.remove()

                # train Generator
                loss_kl = 0.0
                loss_fea = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                # feature loss
                adv_fea_out = netD(adv_fea)
                loss_fea += criterion_bce(adv_fea_out, torch.ones_like(adv_fea_out).detach().cuda())

                loss = loss_kl + loss_fea
                loss.backward()
                optimG.step()
                running_loss += loss.item()

                # train Discriminatord
                adv_fea = adv_fea.detach()
                adv_fea_out = netD(adv_fea)
                optimD.zero_grad()
                loss_D = (0.5 * criterion_bce(netD(img_match_fea), torch.ones_like(adv_fea_out).detach().cuda()) +
                          0.5 * criterion_bce(adv_fea_out, torch.zeros_like(adv_fea_out).detach().cuda()))
                loss_D.backward()
                optimD.step()
                disc_loss += loss_D.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f} \t loss_D: {3:.5f}'.format(epoch, i, running_loss / 10, disc_loss / 10))
                    running_loss = 0
                    disc_loss = 0

            if epoch % 5 == 4:
                file_name = '/netG_{}_{}_ttaa_t{}.pth'
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    elif args.method == 'none':
        for epoch in range(args.epochs):
            running_loss = 0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()
                optimG.zero_grad()

                # Unconstrained Adversaries
                adv = netG(img)

                # Smoothing
                if args.gs:
                    adv = kernel(adv)

                # Projection
                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_out = model(normalize(adv))  # (batch_size, num_classes)
                img_match_out = model(normalize(img_match))

                # Loss
                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                loss = loss_kl
                loss.backward()
                optimG.step()
                running_loss += loss.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 10))
                    running_loss = 0

            file_name = '/netG_{}_{}_none_t{}.pth'
            torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    # My method
    else:
        for epoch in range(args.epochs):
            running_loss_kl = 0
            running_loss_hf = 0
            running_loss_kl_n = 0
            running_loss_fre = 0
            margin = 0.0
            for i, (imgs, _) in enumerate(train_loader):
                img = imgs.cuda()

                try:
                    img_match = next(dataiter)[0]
                except StopIteration:
                    dataiter = iter(train_loader_match)
                    img_match = next(dataiter)[0]
                img_match = img_match.cuda()

                netG.train()

                if args.re_noise:
                    adv, r_adv_hc = netG(img)   # 到底生成noise还是adv待定：输出noise很难收敛
                else:
                    adv = netG(img)
                # adv's hc
                # adv_hc, _ = get_hc(adv)
                # # noise's lc
                # noise_lc = noise - noise_hc

                # adv_lfn = img + noise_lc # get adv with low frequency noise
                # # Projection
                # adv_lfn = torch.min(torch.max(adv_lfn, img - eps), img + eps)
                # adv_lfn = torch.clamp(adv_lfn, 0.0, 1.0)
                #
                # adv_lfn_out = model(normalize(adv_lfn))  # (batch_size, num_classes)
                # img_match_out = model(normalize(img_match))

                # # Loss on low frequency noise
                # loss_kl_lfn = 0.0
                # loss_kl_lfn += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_lfn_out, dim=1),
                #                                                   F.softmax(img_match_out, dim=1))
                # loss_kl_lfn += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                #                                                   F.softmax(adv_lfn_out, dim=1))
                #
                # if args.triplet_loss:
                #     # todo
                #     pass

                loss_fre = torch.tensor(0.0).cuda()
                # high frequency loss(1.0 存在一定问题，但是有一定效果)
                # if args.freq_loss:
                #     loss_fre += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_hc, dim=1),    # 使得每个像素点在三个通道上的值合为 1
                #                                                        F.softmax(img_match_hc, dim=1))
                #     loss_fre += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_hc, dim=1),
                #                                                        F.softmax(adv_hc, dim=1))
                # high frequency loss(2.0)
                if args.hfl:
                    # spatial loss(HFC): L1 loss
                    adv_hc, adv_hf = get_hc(adv)
                    img_match_hc, img_match_hf = get_hc(img_match)

                    adv_hc = torch.mean(adv_hc, dim=1, keepdim=True)
                    img_match_hc = torch.mean(img_match_hc, dim=1, keepdim=True)
                    loss_fre_sp = (1.0 / args.batch_size) * criterion_L1(adv_hc, img_match_hc)
                    # frequency loss(HFC)
                    tmp = (adv_hf - img_match_hf) ** 2
                    loss_fre_fr = tmp[..., 0] + tmp[..., 1]
                    loss_fre_fr = (1.0 / args.batch_size) * torch.sum(loss_fre_fr)
                    loss_fre += 0.005 * (loss_fre_sp + loss_fre_fr)
                # elif args.ffl:
                #     FFL = FocalFrequencyLoss()
                #     adv_hc, _ = get_hc(adv)
                #     img_match_hc, _ = get_hc(img_match)
                #     loss_fre += FFL(adv_hc, img_match_hc)
                elif args.gs:
                    adv = kernel(adv)


                # todo: high frequency noise discriminator
                # hfn_out = model(normalize(noise_hc))
                # img_match_hfc_out = model(normalize(img_match_hc))
                # loss_kl_hfn += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(hfn_out, dim=1),
                #                                                     F.softmax(img_match_hfc_out, dim=1))
                # loss_kl_hfn += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_hfc_out, dim=1),
                #                                                     F.softmax(hfn_out, dim=1))


                # Projection
                # adv = img + noise

                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_out = model(normalize(adv))
                if args.tar_lfc:
                    img_match_lc, _ = get_lc(img_match)
                    img_match_out = model(normalize(img_match_lc))
                else:
                    img_match_out = model(normalize(img_match))

                loss_kl = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                loss_kl_n = torch.tensor(0.0).cuda()
                # triplet loss
                if args.triplet_loss:
                    margin = args.margin
                    img_out = model(normalize(img))
                    loss_kl_n += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                        F.softmax(img_out, dim=1))
                    loss_kl_n += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_out, dim=1),
                                                                        F.softmax(adv_out, dim=1))

                loss_hf = torch.tensor(0.0).cuda()
                if args.re_noise:
                    r_adv_hc_out = model(normalize(r_adv_hc))
                    # 1. KL divergence
                    img_match_hc, _ = get_hc(img_match)
                    img_match_hc_out = model(normalize(img_match_hc))

                    loss_hf += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(r_adv_hc_out, dim=1),
                                                                      F.softmax(img_match_hc_out, dim=1))
                    loss_hf += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_hc_out, dim=1),
                                                                      F.softmax(r_adv_hc_out, dim=1))

                    # 2. CE loss———poor performance
                    # label = F.one_hot(torch.tensor(args.match_target), num_classes=1000).float().cuda()
                    # label = label.unsqueeze(0).repeat(args.batch_size, 1)
                    # loss_hf += 0.1 * criterion_ce(r_adv_hc_out, label)

                loss_trip = loss_kl - loss_kl_n + margin
                loss_trip = torch.max(loss_trip, torch.zeros_like(loss_trip))

                loss = loss_trip + loss_hf + loss_fre
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                running_loss_kl += loss_kl.item()
                # running_loss_hf += loss_hf.item()
                # running_loss_kl_n += loss_kl_n.item()
                running_loss_fre += loss_fre.item()

                # if i % 10 == 9:
                #     print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_hf: {3:.5f} \t loss_kl_n: {4:.5f} \t margin: {5}'.format(
                #         epoch, i, running_loss_kl / 10, running_loss_hf / 10, running_loss_kl_n / 10, margin))
                #     logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_hf: {3:.5f} \t loss_kl_n: {4:.5f} \t margin: {5}'.format(
                #         epoch, i, running_loss_kl / 10, running_loss_hf / 10, running_loss_kl_n / 10, margin))
                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_fre: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_fre / 10))
                    logging.info('Epoch: {0} \t Batch: {1} \t loss_kl: {2:.5f} \t loss_fre: {3:.5f}'.format(
                        epoch, i, running_loss_kl / 10, running_loss_fre / 10))
                    running_loss_kl = 0
                    # running_loss_hf = 0
                    # running_loss_kl_n = 0
                    running_loss_fre = 0

            # file_name_dict = {
            #     (True, False): '/netG_{}_{}_re_t{}.pth',
            #     (True, True): '/netG_{}_{}_re_trip_t{}.pth',
            #     (False, True): '/netG_{}_{}_trip_t{}.pth',
            #     (False, False): '/netG_{}_{}_t{}.pth',
            # }
            #
            # file_name = file_name_dict[(args.re_noise, args.triplet_loss)]

            file_name = '/netG_{}_{}_ffl_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))


if __name__ == '__main__':
    main()


