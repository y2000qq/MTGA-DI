import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from generators import *
# from utils import *

parser = argparse.ArgumentParser(description='Cross Data Transferability')
parser.add_argument('--train_dir', default='imagenet', help='paintings, comics, imagenet')
parser.add_argument('--batch_size', type=int, default=20, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--model_type', type=str, default='resnet50',
                    help='Model against GAN is trained: resnet50, vgg19_bn, densenet121')
parser.add_argument('--target', type=int, default=24, help='target')
parser.add_argument('--save_dir', type=str, default='CDA', help='Directory to save models')
args = parser.parse_args()
print(args)

if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

# Normalize (0-1)
eps = args.eps / 255

# GPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

####################
# Model
####################
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
if args.model_type in ['densenet121', 'vgg19_bn', 'resnet50']:
    scale_size = 256
    img_size = 224
else:
    scale_size = 300
    img_size = 299

# Generator
if args.model_type == 'inception_v3':
    netG = GeneratorResnet(inception=True)
else:
    netG = GeneratorResnet()
netG = nn.DataParallel(netG).cuda()

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t

source_path = '/datasets/ImageNet_50k_990c'
train_set = torchvision.datasets.ImageFolder(source_path, data_transform)
train_size = len(train_set)
if train_size % args.batch_size != 0:
    train_size = (train_size // args.batch_size) * args.batch_size
    train_set.samples = train_set.samples[0:train_size]
    train_set.targets = train_set.targets[0:train_size]

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)
print('Training data size:', train_size)


# Loss
criterion = nn.CrossEntropyLoss()

####################
# Set-up noise if required
####################

# Training
print('Label: {} \t Model: {} \t Distribution: {} \t Saving instances: {}'.format(args.target,
                                                                                  args.model_type,
                                                                                  args.train_dir,
                                                                                  args.epochs))
for epoch in range(args.epochs):
    running_loss = 0
    for i, (img, _) in enumerate(train_loader):
        img = img.cuda()

        # whatever the model think about the input
        label = model(normalize(img.clone().detach())).argmax(dim=-1).detach()

        targte_label = torch.LongTensor(img.size(0))
        targte_label.fill_(args.target)
        targte_label = targte_label.cuda()

        netG.train()
        optimG.zero_grad()

        adv = netG(img)

        # Projection
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        if args.target == -1:
            # Gradient accent (Untargetted Attack)
            adv_out = model(normalize(adv))
            img_out = model(normalize(img))

            loss = -criterion(adv_out - img_out, label)

        else:
            # Gradient decent (Targetted Attack)
            # loss = criterion(model(normalize(adv)), targte_label)
            loss = criterion(model(normalize(adv)), targte_label) + criterion(model(normalize(img)), label)
        loss.backward()
        optimG.step()

        if i % 10 == 9:
            print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 100))
            running_loss = 0
        running_loss += abs(loss.item())

    torch.save(netG.state_dict(), args.save_dir + '/netG_{}_{}_rl_t{}.pth'.format(args.model_type, epoch, args.target))
