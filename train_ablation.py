import argparse
import os
import numpy as np
import socket
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
from gaussian_smoothing import *

from utils import *

parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
parser.add_argument('--src', default='IN_50k_new', help='Source Domain: imagenet, imagenet_10c, IN_50k, etc')
parser.add_argument('--match_target', type=int, default=24, help='Target Domain samples')
parser.add_argument('--match_dataset', default='imagenet', help='Target domain')
parser.add_argument('--batch_size', type=int, default=20, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget during training, eps')
parser.add_argument('--model_type', type=str, default='resnet50', help='Model under attack (discrimnator)')
parser.add_argument('--gs', action='store_true', help='Apply gaussian smoothing')
parser.add_argument('--save_dir', type=str, default='pretrained_generators', help='Directory to save generators')
# parser.add_argument('--freq_netG', action='store_true', help='Use freq Unet')
#
# parser.add_argument('--freq_loss', action='store_true', help='Add High frequency loss')
# parser.add_argument('--orig_loss', action='store_true', help='Add original image loss')
# parser.add_argument('--lfc_loss', action='store_true', help='Add low frequency loss')
#
# parser.add_argument('--alpha_fl', type=float, default=0.1, help='Weight for frequency loss')

parser.add_argument('--method', default='none', help='TTP, TTAA etc')

args = parser.parse_args()
print(args)

hostname = socket.gethostname()

if args.src == 'comic books':
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
else:
    assert False, 'Please provide correct source dataset names: {}'.format(args.src)

if args.match_dataset == 'imagenet':
    if hostname in ['user-Precision-7920-Tower', 'dell-Precision-7920-Tower']:  # 3091 or 3090
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
    if args.freq_netG:
        if args.model_type == 'inception_v3':
            netG = GeneratorUnet_Fre(inception=True)
        else:
            netG = GeneratorUnet_Fre()
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

    if args.method == 'TTP':
        train_set = torchvision.datasets.ImageFolder(source_path, TwoCropTransform(train_transform, img_size))
    else:
        # 这里就用最普通的变换，不用两个CropTransform
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

    if len(train_set_match) < 1300:
        train_set_match.samples = train_set_match.samples[0:1000]
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
    criterion_ce = nn.CrossEntropyLoss()

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

    # 'TTAA' method
    elif args.method == 'TTAA':
        # todo: TTAA
        print('TTAA')

    # My method
    else:
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
                if args.freq_loss:
                    img_match_hc = get_hc(img_match)
                elif args.lfc_loss:
                    img_match_lc = get_lc(img_match)

                netG.train()
                optimG.zero_grad()

                # Unconstrained Adversaries
                if args.freq_netG:
                    adv = netG(img, img_match)
                else:
                    adv = netG(img)

                if args.freq_loss:
                    adv_hc = get_hc(adv)
                elif args.lfc_loss:
                    adv_lc = get_lc(adv)

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
                loss_fre = 0.0
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_out, dim=1),
                                                                  F.softmax(img_match_out, dim=1))
                loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                                  F.softmax(adv_out, dim=1))

                # high frequency loss
                if args.freq_loss:
                    loss_fre += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_hc, dim=1),
                                                                       F.softmax(img_match_hc, dim=1))
                    loss_fre += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_hc, dim=1),
                                                                       F.softmax(adv_hc, dim=1))

                # original image loss
                elif args.orig_loss:
                    loss_fre += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv, dim=1),
                                                                       F.softmax(img_match, dim=1))
                    loss_fre += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match, dim=1),
                                                                       F.softmax(adv, dim=1))
                # low frequency loss
                elif args.lfc_loss:
                    loss_fre += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_lc, dim=1),
                                                                       F.softmax(img_match_lc, dim=1))
                    loss_fre += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_lc, dim=1),
                                                                       F.softmax(adv_lc, dim=1))

                loss = loss_kl + args.alpha_fl * loss_fre
                loss.backward()
                optimG.step()
                running_loss += loss.item()

                if i % 10 == 9:
                    print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 10))
                    running_loss = 0

            file_name_dict = {
                (True, True): '/netG_{}_{}_fn_fl_t{}.pth',
                (True, False): '/netG_{}_{}_fl_t{}.pth',
                (False, True): '/netG_{}_{}_fn_t{}.pth',
                (False, False): '/netG_{}_{}_t{}.pth'
            }

            file_name = file_name_dict[(args.freq_loss, args.freq_netG)]

            torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))


if __name__ == '__main__':
    main()
