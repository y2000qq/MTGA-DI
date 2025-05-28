"""
Attacks based on gradients
"""

import argparse
import os

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchvision.models as models
import timm

import torchattacks

# Purifier
from NRP import *

import logging
import socket
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='Targeted Transferable Perturbations')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for evaluation')
parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget')
parser.add_argument('--num_targets', type=int, default=10, help='10 or 100 targets evaluation')
parser.add_argument('--source_domain', type=str, default='IN', help='Source Domain (TTP): Natural Images (IN) or comic_books')

parser.add_argument('--target_model', type=str, default='vgg19_bn', help='Black-Box(unknown) model: SIN, Augmix etc')
parser.add_argument('--source_model', type=str, default='resnet50', help='TTP Discriminator: \
{resnet18, resnet50, resnet101, resnet152, dense121, dense161, dense169, dense201,\
 vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,\
 ens_vgg16_vgg19_vgg11_vgg13_all_bn,\
 ens_res18_res50_res101_res152\
 ens_dense121_161_169_201}')
parser.add_argument('--method', type=str, default='MI', help='MI,DI,TI,SI,TI_DI,SI_DI')

parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
parser.add_argument('--n_iter', type=int, default=10, help='Number of iterations for the attack')
args = parser.parse_args()
print(args)

if args.source_domain == 'IN':
    hostname = socket.gethostname()
    if hostname == 'user-Precision-7920-Tower' or hostname == 'dell-Precision-7960-Tower':  # 3091 or A6000
        test_dir = '/datasets/Imagenet2012/val'
    elif hostname == 'ubuntu':  # 503
        test_dir = '/datasets/ILSVRC2012/val2'
    elif hostname == 'R2S1-gpu':  # 5014
        test_dir = '/datasets/ImageNet2012/val'
    elif hostname == 'LAPTOP-01RUAH3M':  # Laptop
        test_dir = 'E:/datasets/ILSVRC2012/val'
elif args.source_domain == 'comic_books':
    test_dir = '/datasets/comic books/test'

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Set-up log file
os.makedirs('Gradient_Attacks', exist_ok=True)
logfile = 'Gradient_Attacks/{}_10_target_eval_eps_{}_{}_to_{}.log'.format(args.metohd, args.eps, args.source_model, args.target_model)

logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=logfile)

eps = args.eps / 255.0
momentum = args.momentum
n_iter = args.n_iter
alpha = eps / n_iter

# Load Targeted Model
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

if args.target_model in model_names:
    model_t = models.__dict__[args.target_model](pretrained=True)
elif args.target_model == 'inception_v4':
    model_t = timm.create_model('inception_v4', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_v4.tf_in1k/model.safetensors')
elif args.target_model == 'incres_v2':
    model_t = timm.create_model('inception_resnet_v2', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_resnet_v2.tf_in1k/model.safetensors')
elif args.target_model == 'vit':
    model_t = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg_in1k/model.safetensors')
# adversarial training
elif args.target_model == "ens_adv_inception_resnet_v2":
    model_t = timm.create_model("ens_adv_inception_resnet_v2", pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_resnet_v2.tf_ens_adv_in1k/model.safetensors')
elif args.target_model == "adv_inc_v3":
    model_t = timm.create_model('inception_v3.tf_adv_in1k', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_v3.tf_adv_in1k/model.safetensors')

#
elif args.target_model == 'SIN':
    model_t = torchvision.models.resnet50(pretrained=False)
    model_t = torch.nn.DataParallel(model_t)
    checkpoint = torch.load('pretrained_models/resnet50_train_60_epochs-c8e5653e.pth.tar')
    model_t.load_state_dict(checkpoint["state_dict"])
elif args.target_model == 'SIN_IN':
    model_t = torchvision.models.resnet50(pretrained=False)
    model_t = torch.nn.DataParallel(model_t)
    checkpoint = torch.load('pretrained_models/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar')
    model_t.load_state_dict(checkpoint["state_dict"])
elif args.target_model == 'SIN_IN_FIN':
    model_t = torchvision.models.resnet50(pretrained=False)
    model_t = torch.nn.DataParallel(model_t)
    checkpoint = torch.load('pretrained_models/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar')
    model_t.load_state_dict(checkpoint["state_dict"])
elif args.target_model == 'Augmix':
    model_t = torchvision.models.resnet50(pretrained=False)
    model_t = torch.nn.DataParallel(model_t)
    checkpoint = torch.load('pretrained_models/checkpoint.pth.tar')
    model_t.load_state_dict(checkpoint["state_dict"])
else:
    assert (args.target_model in model_names), 'Please provide correct target model names: {}'.format(model_names)

model_t = model_t.to(device)
model_t.eval()

####################
# Data
####################
# Input dimensions
scale_size = 256
img_size = 224

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

if args.method == 'GAP':
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
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t

if args.num_targets==10:
    targets = [24,99,245,344,471,555,661,701,802,919]
elif args.num_targets==100:
    targets = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919, 3, 16, 36, 48, 52, 69, 71, 85, 107, 114, 130, 138, 142, 151, 162, 178, 189, 193, 207, 212, 228, 240, 260, 261, 276, 285, 291, 309, 317, 328, 340, 358, 366, 374, 390, 393, 404, 420, 430, 438, 442, 453, 464, 485, 491, 506, 513, 523, 538, 546, 569, 580, 582, 599, 605, 611, 629, 638, 646, 652, 678, 689, 707, 717, 724, 735, 748, 756, 766, 779, 786, 791, 813, 827, 836, 849, 859, 866, 879, 885, 893, 901, 929, 932, 946, 958, 963, 980, 984, 992]
else:
    raise ValueError('Please provide correct number of targets: 10 or 100')

total_acc = 0
total_samples = 0
logger.info('Target model: {}'.format(args.target_model))
for idx, target in enumerate(targets):
    logger.info('{:<10} {:<10} {:<10} {:<10}'.format('Epsilon', 'Target', 'Acc.', 'Distance'))

    test_set = datasets.ImageFolder(test_dir, data_transform)

    if args.source_domain == 'IN':
        # Remove samples that belong to the target attack label.
        source_samples = []
        for img_name, label in test_set.samples:
            if label != target:
                source_samples.append((img_name, label))
        test_set.samples = source_samples

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)
    test_size = len(test_set)
    print('Test data size:', test_size)

    # Reset Metrics
    acc=0
    distance = 0
    print('Target model: {}, Target: {}'.format(args.target_model, target))

    for i, (img, _) in enumerate(test_loader):
        # print('Target: {}, Batch: {}/{}'.format(target, i+1, len(test_loader)))
        img = img.to(device)

        target_label = torch.LongTensor(img.size(0))
        target_label.fill_(target)
        target_label = target_label.to(device)

        adv = img.clone().detach()
        momentum = torch.zeros_like(img).detach().to(device)

        for _ in range(n_iter):
            adv.requires_grad = True
            # DI-FGSM
            if 'DI' in args.method:
                out = model_t(DI(normalize(adv)))
            else:
                out = model_t(normalize(adv))

            loss = F.cross_entropy(out, target_label)
            grad = torch.autograd.grad(loss, adv, retain_graph=True, create_graph=False)[0]

            if 'TI' in args.method:
                grad = F.conv2d(grad, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

            if 'MI' in args.method:
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * decay
                momentum = grad

            adv = adv.detach() + alpha * torch.sign(grad)
            delta = torch.clamp(adv - img, min=-eps, max=eps)
            adv = torch.clamp(img + delta, min=0, max=1).detach()


        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        out = model_t(normalize(adv.clone().detach()))
        acc += torch.sum(out.argmax(dim=-1) == target_label).item()

        distance += (img - adv).max() * 255

    total_acc+=acc
    total_samples+=test_size
    logger.info('{:<10} {:<10} {:<10.4f} {:<10.4f}'.format(int(eps * 255), target, acc / test_size, distance / (i + 1)))

logger.info('Average Target Transferability')
logger.info('{:<10} {:<10.4f} {:<10.4f}'.format(int(eps * 255), total_acc / total_samples, distance / (i + 1)))
logger.info('-'*100)