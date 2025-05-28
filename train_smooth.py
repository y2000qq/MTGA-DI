import argparse
import os
import logging

import numpy as np
import socket

from focal_frequency_loss import FocalFrequencyLoss
from matplotlib import pyplot as plt

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch import combinations
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable as V
from tqdm import tqdm

# from pytorch_wavelets import DWTForward, DWTInverse

from generators import *
from discriminator import *
from gaussian_smoothing import *
from high_frequency_discriminator import High_Frequency_Discriminator
from reweight_net import ReweightNet
from shapley import *

from utils import *


parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
parser.add_argument('--src', default='IN_50k_new', help='Source Domain: imagenet, imagenet_10c, IN_50k, etc')
parser.add_argument('--match_target', type=int, default=24, help='Target Domain samples')
parser.add_argument('--model_type', type=str, default='resnet50', help='Model under attack (discrimnator)')
parser.add_argument('--gs', action='store_true', help='Apply gaussian smoothing')
parser.add_argument('--class_freq_mask', action='store_true', help='Apply class frequency mask')
parser.add_argument('--method', type=str, default='none',help='none, argu, single')
parser.add_argument('--save_dir', type=str, default='pretrained_generators', help='Directory to save generators')

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
elif args.class_freq_mask:
    logfile = os.path.join('train_loss', 'class_freq_mask_{}_t{}.log'.format(args.model_type, args.match_target))
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

    if args.method == 'argu':
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
    criterion_L2 = nn.MSELoss(reduction='sum')

    # todo
    if args.method == 'argu':   # argument but img_match
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

                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_rot_freq, img_rot_freq)
                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_aug_freq, img_aug_freq)

                # like img_match

                # DCS_img_match
                img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                img_match_freq = torch.fft.fftshift(img_match_freq)
                img_match_freq = torch.abs(img_match_freq)
                img_match_freq = img_match_freq.mean(dim=0)

                loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq)
                loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_rot_freq, img_match_freq)
                loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_aug_freq, img_match_freq)

                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L2(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)
                elif args.class_freq_mask:
                    pass

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

            file_name = '/netG_{}_{}_DCS_img_match_argu_cfm_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(),
                               args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(),
                           args.save_dir + file_name.format(args.model_type, epoch, args.match_target))

    else:
        if args.class_freq_mask:
            with torch.no_grad():
                # img_match_freq_weight = torch.zeros((3, img_size, img_size)).cuda()
                # for i, (img_match, label) in enumerate(train_loader_match):
                #     img_match = img_match.cuda()
                #     label = label.cuda()
                #     img_match_freq = torch.fft.fftshift(torch.fft.fft2(img_match, norm='ortho'))
                #     img_match_freq.requires_grad = True
                #     # img_match_freq = torch.fft.ifftshift(img_match_freq)
                #     img_match = torch.fft.ifft2(torch.fft.ifftshift(img_match_freq), norm='ortho').real
                #     output = model(normalize(img_match))
                #     loss = nn.CrossEntropyLoss()(output, label)
                #     img_match_freq_grad = torch.autograd.grad(loss, img_match_freq)[0]
                #     img_match_freq_grad = torch.abs(img_match_freq_grad)
                #     img_match_freq_weight += torch.sum(img_match_freq_grad, dim=0)
                # img_match_freq_weight = torch.log(img_match_freq_weight+1)

                # img_match_freq_weight = torch.sigmoid(img_match_freq_weight)

                # for i, (img_match, label) in enumerate(train_loader_match):
                #     img_match = img_match.cuda()
                #     label = label.cuda()
                #     output = model(normalize(img_match))
                #     loss = nn.CrossEntropyLoss()(output, label)
                #     img_match_grad = torch.autograd.grad(loss, img_match)[0]
                #     img_match_freq_grad = torch.fft.fftshift(torch.fft.fft2(img_match_grad, norm='ortho'))
                #     img_match_freq_weight += torch.sum(img_match_freq_grad, dim=0)
                # img_match_freq_weight_flatten = img_match_freq_weight.flatten()
                # sorted_tensor, indices = torch.sort(img_match_freq_weight_flatten, descending=True)
                # threshold = sorted_tensor[int(len(sorted_tensor) * 0.99)]
                # img_match_freq_weight[img_match_freq_weight < threshold] = 0
                #
                # plt.imshow(torch.mean(img_match_freq_weight, dim=0).detach().cpu().numpy())
                # plt.colorbar()
                # plt.show()
                # plt.imshow((torch.mean(img_match_freq_weight, dim=0) > 0.2).detach().cpu().numpy())
                # plt.colorbar()
                # plt.show()
                # img_match_freq_weight = img_match_freq_weight / len(train_loader_match)

                ### shapley method ###
                # img_match_freq_cls = torch.zeros((3, img_size, img_size)).cuda()

                # mask_players
                avg_energy_fr = torch.zeros(img_size, img_size, device='cuda')
                for i, (img_match, _) in enumerate(train_loader_match):
                    img_match = img_match.cuda()
                    avg_energy_fr += get_mean_energy_dataset(img_match)
                avg_energy_fr /= len(train_loader_match)
                emp_dist = avg_energy_fr
                percent_range = None
                # percent_range = [0, 7, 16, 26, 38, 51, 66, 83, 102, 127, 224]
                # percent_range = [0, 20, 51, 92, 210]
                if percent_range is None:
                    percent_range = get_percentage_masks_relevance(emp_dist, 0.1)
                    logging.info('percent_range:{}'.format(percent_range))
                mask_players = []
                for i in range(len(percent_range)-1):
                    mask_players.append(1 - get_mask(emp_dist, percent_range[i], percent_range[i + 1]))

                shapley_values = None
                # shapley_values = torch.tensor([8382.949, 4237.4526, 4382.732, 4275.8066, 3294.2754, 2249.6772, 1663.04,
                #                                1519.0676, 1401.532, 894.4356], device='cuda')   # 245
                # shapley_values = torch.tensor([9056.482,6023.7725,6145.063,5034.214,3519.6013,2733.2007,
                #                                2131.4336,1469.4655,724.2866,229.44354], device='cuda')  # 802
                shapley_mask = torch.zeros((img_size, img_size), device='cuda')
                if shapley_values is None:
                    # shapley_values = {i: 0.0 for i in range(len(mask_players))}
                    shapley_values = torch.zeros(len(mask_players)).cuda()
                    for i, (img_match, _) in enumerate(train_loader_match):
                        print('batch:{}'.format(i))
                        # print(shapley_values.cpu().numpy())
                        # print((shapley_values / shapley_values.sum()).cpu().numpy())
                        # 累计占比
                        # print((shapley_values.cumsum(dim=0) / shapley_values.sum()).cpu().numpy())

                        if i == 5:
                            break

                        img_match = img_match.cuda()
                        img_match_freq = torch.fft.fftshift(torch.fft.fft2(img_match, norm='ortho'))

                        for player_id, mask_player in enumerate(mask_players):
                            other_players = torch.tensor([k for k in range(len(mask_players)) if k != player_id], device='cuda')
                            for j in range(len(other_players)+1):   # j: C(j, len(other_players))
                                if j == 0:
                                    mask_l = mask_player.clone()
                                    mask_r = torch.zeros((img_size, img_size), device='cuda')
                                    img_match_freq_masked_l = img_match_freq * mask_l
                                    img_match_freq_masked_r = img_match_freq * mask_r
                                    img_match_masked_l = torch.fft.ifft2(torch.fft.ifftshift(img_match_freq_masked_l),
                                                                         norm='ortho').real
                                    img_match_masked_r = torch.fft.ifft2(torch.fft.ifftshift(img_match_freq_masked_r),
                                                                         norm='ortho').real
                                    output_l = model(normalize(img_match_masked_l))
                                    output_r = model(normalize(img_match_masked_r))
                                    prob_t_l = F.softmax(output_l, dim=1)[:, args.match_target]
                                    prob_t_r = F.softmax(output_r, dim=1)[:, args.match_target]
                                    shapley_values[player_id] += (prob_t_l - prob_t_r).sum().item()
                                else:
                                    for combination in combinations(other_players, j):
                                        mask_l = mask_player.clone()
                                        mask_r = torch.zeros((img_size, img_size), device='cuda')
                                        for id in combination:
                                            mask_l += mask_players[id.item()]
                                            mask_r += mask_players[id.item()]
                                        img_match_freq_masked_l = img_match_freq * mask_l
                                        img_match_freq_masked_r = img_match_freq * mask_r
                                        img_match_masked_l = torch.fft.ifft2(torch.fft.ifftshift(img_match_freq_masked_l),
                                                                             norm='ortho').real
                                        img_match_masked_r = torch.fft.ifft2(torch.fft.ifftshift(img_match_freq_masked_r),
                                                                             norm='ortho').real
                                        output_l = model(normalize(img_match_masked_l))
                                        output_r = model(normalize(img_match_masked_r))
                                        prob_t_l = F.softmax(output_l, dim=1)[:, args.match_target]
                                        prob_t_r = F.softmax(output_r, dim=1)[:, args.match_target]
                                        shapley_values[player_id] += (prob_t_l - prob_t_r).sum().item()

                                    # mask_combinations = []
                                    # for combination in combinations(other_players, j):
                                    #     mask = mask_player.clone()
                                    #     for id in combination:
                                    #         mask += mask_players[id.item()]
                                    #     mask_combinations.append(mask)
                                    #
                                    # img_match_freq_masked = img_match_freq.unsqueeze(0).repeat(len(mask_combinations), 1, 1,
                                    #                                                            1)
                                    # for i, mask in enumerate(mask_combinations):
                                    #     img_match_freq_masked[i] *= mask.unsqueeze(0)
                                    #
                                    # img_match_masked = torch.fft.ifft2(
                                    #     torch.fft.ifftshift(img_match_freq_masked, dim=(-2, -1)), dim=(-2, -1),
                                    #     norm='ortho').real
                                    #
                                    # outputs = model(normalize(img_match_masked))
                                    # prob_t = F.softmax(outputs, dim=2)[:, :, args.match_target]
                                    #
                                    # baseline_prob_t = prob_t[:1].repeat(len(mask_combinations), 1)
                                    # prob_t_diff = prob_t - baseline_prob_t
                                    #
                                    # for i, combination in enumerate(combinations(other_players, j)):
                                    #     for id in combination:
                                    #         shapley_values[id.item()] += prob_t_diff[i].sum().item()
                    #
                    # for player_id, mask_player in enumerate(mask_players):
                    #     other_players = torch.tensor([k for k in range(len(mask_players)) if k != player_id])
                    #     for j in tqdm(range(len(other_players)+1)):   # j: C(j, len(other_players))
                    #         if j == 0:
                    #             mask_l = mask_player.clone()
                    #             mask_r = torch.zeros((img_size, img_size)).cuda()
                    #             for i, (img_match, _) in enumerate(train_loader_match):
                    #                 img_match = img_match.cuda()
                    #                 img_match_freq = torch.fft.fftshift(torch.fft.fft2(img_match, norm='ortho'))
                    #                 img_match_freq_masked_l = img_match_freq * mask_l
                    #                 img_match_freq_masked_r = img_match_freq * mask_r
                    #                 img_match_masked_l = torch.fft.ifft2(torch.fft.ifftshift(img_match_freq_masked_l),
                    #                                                      norm='ortho').real
                    #                 img_match_masked_r = torch.fft.ifft2(torch.fft.ifftshift(img_match_freq_masked_r),
                    #                                                      norm='ortho').real
                    #                 output_l = model(normalize(img_match_masked_l))
                    #                 output_r = model(normalize(img_match_masked_r))
                    #                 prob_t_l = F.softmax(output_l, dim=1)[:, args.match_target]
                    #                 prob_t_r = F.softmax(output_r, dim=1)[:, args.match_target]
                    #                 shapley_values[player_id] += (prob_t_l - prob_t_r).mean().item()
                    #         else:
                    #             for combination in combinations(other_players, j):
                    #                 mask_l = mask_player.clone()
                    #                 mask_r = torch.zeros((img_size, img_size)).cuda()
                    #                 for id in combination:
                    #                     mask_l += mask_players[id.item()]
                    #                     mask_r += mask_players[id.item()]
                    #                 for i, (img_match, _) in enumerate(train_loader_match):
                    #                     img_match = img_match.cuda()
                    #                     img_match_freq = torch.fft.fftshift(torch.fft.fft2(img_match, norm='ortho'))
                    #                     img_match_freq_masked_l = img_match_freq * mask_l
                    #                     img_match_freq_masked_r = img_match_freq * mask_r
                    #                     img_match_masked_l = torch.fft.ifft2(torch.fft.ifftshift(img_match_freq_masked_l),
                    #                                                          norm='ortho').real
                    #                     img_match_masked_r = torch.fft.ifft2(torch.fft.ifftshift(img_match_freq_masked_r),
                    #                                                          norm='ortho').real
                    #                     output_l = model(normalize(img_match_masked_l))
                    #                     output_r = model(normalize(img_match_masked_r))
                    #                     prob_t_l = F.softmax(output_l, dim=1)[:, args.match_target]
                    #                     prob_t_r = F.softmax(output_r, dim=1)[:, args.match_target]
                    #                     shapley_values[player_id] += (prob_t_l - prob_t_r).mean().item()
                    #
                    # shapley_values /= len(train_loader_match) * 2**len(mask_players)

                    # shapley method with Monte Carlo
                    logging.info('shapley_values: {}'.format(shapley_values.cpu().numpy()))

                    # shapley_weight = shapley_values / torch.max(torch.abs(shapley_values))
                    # logging.info('shapley_weight: {}'.format(shapley_weight.cpu().numpy()))
                    # for i, mask_player in enumerate(mask_players):
                    #     shapley_mask += mask_player * shapley_weight[i].item()

                cum_weights = shapley_values.cumsum(dim=0) / shapley_values.sum()
                for i, mask_player in enumerate(mask_players):
                    if cum_weights[i].item() < 0.98:
                        shapley_mask += mask_player


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
                #
                # adv_freq = torch.fft.fft2(adv, norm='ortho')
                # adv_freq = torch.fft.fftshift(adv_freq)
                # adv_freq = torch.abs(adv_freq)
                # adv_freq = adv_freq.mean(dim=0)
                #
                # # like img
                #
                # # DCS_img
                # # img_freq = torch.fft.fft2(img, norm='ortho')
                # # img_freq = torch.fft.fftshift(img_freq)
                # # img_freq = torch.abs(img_freq)
                # # img_freq = img_freq.mean(dim=0)
                #
                # # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_freq)
                #
                # # like img_match
                #
                # # DCS_img_match
                # img_match_freq = torch.fft.fft2(img_match, norm='ortho')
                # img_match_freq = torch.fft.fftshift(img_match_freq)
                # img_match_freq = torch.abs(img_match_freq)
                # img_match_freq = img_match_freq.mean(dim=0)
                #
                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L1(adv_freq, img_match_freq)

                # loss_dcs += 0.01 * (1.0 / args.batch_size) * criterion_L2(adv_freq, img_match_freq)

                if args.gs:
                    adv = kernel(adv)
                elif args.class_freq_mask:
                    adv_freq = torch.fft.fftshift(torch.fft.fft2(adv, norm='ortho'))
                    adv_freq = adv_freq * shapley_mask
                    adv = torch.fft.ifft2(torch.fft.ifftshift(adv_freq), norm='ortho').real

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

            file_name = '/netG_{}_{}_cfm_t{}.pth'

            if args.epochs != 20:
                if epoch % 10 == 9:
                    torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))
            else:
                torch.save(netG.state_dict(), args.save_dir + file_name.format(args.model_type, epoch, args.match_target))


if __name__ == '__main__':
    main()


