import argparse
import os
import datetime
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from feature_discriminator import FeatureDiscriminator
from label_discriminator import *
from torchvision.models import vgg16, resnet50, vgg19_bn, densenet121
from generators import *
from gaussian_smoothing import *
import socket

from utils import get_RPD_mask, Layer_out

parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
parser.add_argument('--src', default='IN_50k_new', help='Source Domain: imagenet, imagenet_10c, IN_50k, comic_books, etc')
parser.add_argument('--match_dataset', default='imagenet', help='Target domain')

parser.add_argument('--match_target', type=int, default=24, help='target class(of ImageNet)')
parser.add_argument('--feature_layer', type=int, default=-1, help='Extract feature of the label discriminator')
parser.add_argument('--batch_size', type=int, default=50, help='Number of training samples/batch, 20 or 50')
parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget during training, eps')
parser.add_argument('--model_type', type=str, default='resnet50', help='Model under attack (discrimnator)')
parser.add_argument('--save_dir', type=str, default='TTAA', help='Directory to save generators and AuxNet')
args = parser.parse_args()
print(args)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

eps = args.eps / 255

# GPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

# Input dimensions
if args.model_type == 'inception_v3':
    scale_size = 300
    img_size = 299
else:
    scale_size = 256
    img_size = 224

# Data
transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()])

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
    if hostname in ['user-Precision-7920-Tower', 'dell-Precision-7920-Tower', 'dell-Precision-7960-Tower']:  # 3091 or 3090 or A6000
        match_dir = '/datasets/Imagenet2012/train'
    elif hostname == 'ubuntu':  # 503
        match_dir = '/datasets/ILSVRC2012/train'
    elif hostname == 'R2S1-gpu':  # 5014
        match_dir = '/datasets/ImageNet2012/train'
    elif hostname == 'dell-PowerEdge-T640': # 4090
        match_dir = '/data/Imagenet2012/train'
    else:
        assert False, 'Please provide correct target dataset names: {}'.format(args.match_dataset)

train_set = torchvision.datasets.ImageFolder(
    root=source_path,
    transform=transform)
train_size = len(train_set)
if train_size % args.batch_size != 0:
    train_size = (train_size // args.batch_size) * args.batch_size
    train_set.samples = train_set.samples[0:train_size]
    train_set.targets = train_set.targets[0:train_size]
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True)
# test_set = torchvision.datasets.ImageFolder(
#     root=os.path.join(args.src_dir, 'test'),
#     transform=transform)
# test_loader = torch.utils.data.DataLoader(
#     test_set,
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=False)
print('Training data size:', train_size)
target_set = torchvision.datasets.ImageFolder(
    root=match_dir,
    transform=transform)
target_set.samples = [target_set.samples[i] for i in range(len(target_set.targets))
                           if target_set.targets[i] == args.match_target]
target_set.targets = [target_set.targets[i] for i in range(len(target_set.targets))
                           if target_set.targets[i] == args.match_target]
target_set_size = len(target_set)
if target_set_size % args.batch_size != 0:
    target_set_size = (target_set_size // args.batch_size) * args.batch_size
    target_set.samples = target_set.samples[0:target_set_size]
    target_set.targets = target_set.targets[0:target_set_size]
target_loader = torch.utils.data.DataLoader(
    target_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True)
dataiter = iter(target_loader)
print('Training (Match) data size:', target_set_size)
# Generator
if args.model_type == 'inception_v3':
    netG = GeneratorResnet(inception=True)
else:
    netG = GeneratorResnet()
netG = nn.DataParallel(netG).cuda()

# Label_Discriminator & Feature_Extractor
# if args.model_type == 'vgg19_bn':
#     l_d = Vgg19_bn()
#     feature_layer = 24
#     # Feature Discriminator
#     f_d = FeatureDiscriminator(256)
#     f_d = nn.DataParallel(f_d).cuda()
# elif args.model_type == 'vgg19':
#     l_d = Vgg19()
#     feature_layer = 16
#     # Feature Discriminator
#     f_d = FeatureDiscriminator(256)
#     f_d = nn.DataParallel(f_d).cuda()
# elif args.model_type == 'densenet121':
#     l_d = Densenet121()
#     feature_layer = 4
#     # Feature Discriminator
#     f_d = FeatureDiscriminator(256)
#     f_d = nn.DataParallel(f_d).cuda()
# elif args.model_type == 'resnet50':
#     l_d = Resnet50()
#     feature_layer = 3
#     # Feature Discriminator
#     f_d = FeatureDiscriminator(64)
#     f_d = nn.DataParallel(f_d).cuda()
# elif args.model_type == 'vgg16':
#     l_d = Vgg16()
#     feature_layer = 5
#     # Feature Discriminator
#     f_d = FeatureDiscriminator(128)
#     f_d = nn.DataParallel(f_d).cuda()
# # l_d = eval(args.model_type)()
# l_d = nn.DataParallel(l_d).cuda().eval()
# if args.feature_layer != -1:
#     feature_layer = args.feature_layer

# Target model
# t_model = resnet50(pretrained=True).cuda().eval()

# Optimizer

# Label Discriminator
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

if args.model_type in model_names:
    l_d = models.__dict__[args.model_type](pretrained=True)
else:
    assert (args.model_type in model_names), 'Please provide correct target model names: {}'.format(model_names)

l_d = nn.DataParallel(l_d).cuda().eval()

if args.model_type == 'resnet50':
    target_layer = l_d.module._modules.get('maxpool')  # 4: 56*56
    f_d = FeatureDiscriminator(64)
    f_d = nn.DataParallel(f_d).cuda()
elif args.model_type == 'vgg19_bn':
    target_layer = l_d.module._modules.get('features')[12] # layer 26 in vgg19_bn corresponds to layer 17 in vgg19: 256*56*56
    f_d = FeatureDiscriminator(256)
    f_d = nn.DataParallel(f_d).cuda()
elif args.model_type == 'densenet121':
    target_layer = l_d.module._modules.get('features')[5]  # layer 5
    f_d = FeatureDiscriminator(256)
    f_d = nn.DataParallel(f_d).cuda()
else:
    assert False, 'Please provide correct target model names: {}'.format(model_names)

optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimD = optim.Adam(f_d.parameters(), lr=args.lr, betas=(0.5, 0.999))

BCE = nn.BCELoss().cuda()
CE = nn.CrossEntropyLoss().cuda()
KL = nn.KLDivLoss().cuda()

# ----------
#  Training
# ----------
for epoch in range(args.epochs):
    D_batch_real_loss = 0
    D_batch_fake_loss = 0
    G_batch_class_loss = 0
    G_batch_feature_loss = 0
    for i, (img, _) in enumerate(train_loader):
        netG.train()
        f_d.train()

        img = img.cuda()
        try:
            t_img = next(dataiter)[0]
        except StopIteration:
            dataiter = iter(target_loader)
            t_img = next(dataiter)[0]
        t_img = t_img.cuda()

        # t_outputs, t_features = l_d(normalize(t_img.clone().detach()))
        # t_feature = t_features[feature_layer]
        # t_feature_similarity = f_d(t_feature)
        h = Layer_out(target_layer)
        t_outputs = l_d(normalize(t_img.clone().detach()))
        t_feature = torch.cat(h.features,dim=0)
        t_feature_similarity = f_d(t_feature)
        h.remove()

        adv = netG(img.clone())

        # Random Perturbation Dropping
        noise = adv - img
        mask = get_RPD_mask(noise)
        adv = img + mask * noise

        # Projection
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)
        adv = adv.cuda()

        # outputs, s_features = l_d(normalize(adv.clone().detach()))
        # s_feature = s_features[feature_layer]
        # s_feature_similarity = f_d(s_feature)
        h = Layer_out(target_layer)
        outputs = l_d(normalize(adv.clone().detach()))
        s_feature = h.features
        s_feature_similarity = f_d(s_feature)
        h.remove()

        # Update the feature discriminator
        D_fake_loss = BCE(s_feature_similarity, torch.zeros(s_feature_similarity.shape).detach().cuda())
        D_real_loss = BCE(t_feature_similarity, torch.ones(t_feature_similarity.shape).detach().cuda())
        D_loss = 0.5 * D_fake_loss + 0.5 * D_real_loss

        optimD.zero_grad()
        D_loss.backward(retain_graph=True)
        optimD.step()

        # Update the generator
        # s_pred, s_features = l_d(normalize(adv.clone().detach()))
        # s_feature = s_features[feature_layer]
        # s_feature_similarity = f_d(s_feature)
        # t_pred, _ = l_d(normalize(t_img.clone().detach()))
        h = Layer_out(target_layer)
        s_pred = l_d(normalize(adv.clone().detach()))
        s_feature = h.features
        s_feature_similarity = f_d(s_feature)
        h.remove()
        t_pred = l_d(normalize(t_img.clone().detach()))

        G_feature_loss = BCE(s_feature_similarity, torch.ones(s_feature_similarity.shape).detach().cuda())
        G_class_loss = KL(F.log_softmax(s_pred, dim=1), F.softmax(t_pred, dim=1)) + KL(F.log_softmax(t_pred, dim=1), F.softmax(s_pred, dim=1))
        G_loss = G_feature_loss + G_class_loss

        optimG.zero_grad()
        G_loss.backward()
        optimG.step()

        D_batch_real_loss += D_real_loss.item()
        D_batch_fake_loss += D_fake_loss.item()
        G_batch_class_loss += G_class_loss.item()
        G_batch_feature_loss += G_feature_loss.item()

        if (i + 1) % 100 == 0:
            print(
                'Epoch: {0}/{1} \t Batch: {2}/{3} \t Generator class loss: {4:.3f} \t feature loss: {5:.3f} \t Discriminator real loss: {6:.3f} \t fake loss: {7:.3f}' \
                .format(epoch, args.epochs, i + 1, len(train_loader), G_batch_class_loss, G_batch_feature_loss,
                        D_batch_real_loss, D_batch_fake_loss))
            G_batch_class_loss = 0
            G_batch_feature_loss = 0
            D_batch_real_loss = 0
            D_batch_fake_loss = 0

    # with torch.no_grad():
    #     target_fooling_rate = 0
    #     for i, (img, label) in enumerate(test_loader):
    #         img = img.cuda()
    #         label = label.cuda()
    #
    #         adv = netG(img.clone())
    #         # Projection
    #         adv = torch.min(torch.max(adv, img - eps), img + eps)
    #         adv = torch.clamp(adv, 0.0, 1.0)
    #
    #         outputs = t_model(normalize(adv))
    #         pred = outputs.argmax(dim=1)
    #
    #         target_fooling_rate += torch.sum(pred == args.match_target)
    #         # res.extend(list(pred.cpu().numpy()))
    #     print(
    #         '【Epoch: {0}/{1}】\t【TARGET Transfer Fooling Rate: {2}】'.format(
    #             epoch, args.epochs,
    #             target_fooling_rate / len(test_set)))
    #     # cnt = Counter(res)
    #     # print('val:', cnt)
    if args.epochs != 20:
        if epoch % 10 == 9:
            torch.save(netG.state_dict(),
                       args.save_dir + '/netG_{}_ttaa_{}_t{}.pth'.format(args.model_type, epoch, args.match_target))
    else:
        torch.save(netG.state_dict(),
                   args.save_dir + '/netG_{}_ttaa_{}_t{}.pth'.format(args.model_type, epoch, args.match_target))
        # torch.save(f_d, args.save_dir + '/f_d_{}_{}_{}.pth'.format(args.model_type, epoch, args.match_target))
