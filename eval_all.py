"""
Evaluation for 10/100-targets allsource setting as discussed in our paper.
For each target, we have 49500 samples of the other classes.
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
from tqdm import tqdm

from GAP.material.models.generators import ResnetGenerator
from generators import GeneratorResnet, GeneratorResnet_R, ConGeneratorResnet, Generator, ConGeneratorResnet_adv
from gaussian_smoothing import *

# Purifier
from NRP import *
from art.defences.preprocessor import FeatureSqueezing, JpegCompression, SpatialSmoothing
from res152_wide import get_model

import logging
import socket
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Targeted Transferable Perturbations')
# parser.add_argument('--test_dir', default='../../../data/IN/val')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for evaluation')
parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget')
parser.add_argument('--num_targets', type=int, default=10, help='10 or 100 targets evaluation')
parser.add_argument('--source_domain', type=str, default='IN', help='Source Domain (TTP): Natural Images (IN) or comic_books')

parser.add_argument('--hfl', action='store_true', help='Add frequency loss')
parser.add_argument('--re_noise', action='store_true', help='reweight noise use frequency information')
parser.add_argument('--gs', action='store_true', help='Gaussian Smoothing')

parser.add_argument('--alpha', type=float, default=0.0005, help='alpha')

parser.add_argument('--target_model', type=str, default='vgg19_bn', help='Black-Box(unknown) model: SIN, Augmix etc')
parser.add_argument('--source_model', type=str, default='resnet50', help='TTP Discriminator: \
{resnet18, resnet50, resnet101, resnet152, dense121, dense161, dense169, dense201,\
 vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,\
 ens_vgg16_vgg19_vgg11_vgg13_all_bn,\
 ens_res18_res50_res101_res152\
 ens_dense121_161_169_201}')

# For purification (https://github.com/Muzammal-Naseer/NRP)
parser.add_argument('--NRP', action='store_true', help='Apply Neural Purification to reduce adversarial effect')
# JPEG Compression
parser.add_argument('--JPEG', action='store_true', help='Apply JPEG compression to reduce adversarial effect')
parser.add_argument('--jpeg_quality', type=int, default=75, help='JPEG compression quality')
# bit-depth reduction
parser.add_argument('--BR', action='store_true', help='Apply bit-depth reduction to reduce adversarial effect')
parser.add_argument('--bit_depth', type=int, default=3, help='Bit-depth reduction to reduce adversarial effect')
# Median Smoothing
parser.add_argument('--MS', action='store_true', help='Apply Median Filtering to reduce adversarial effect')
parser.add_argument('--window_size', type=int, default=3, help='Window size for Median Filtering')

# too strong
parser.add_argument('--RS', action='store_true', help='Apply Random Smoothing to reduce adversarial effect')
parser.add_argument('--noise', type=float, default=0.00, help='Noise level for Random Smoothing')

parser.add_argument('--test', action='store_true', help='Test')
parser.add_argument('--none', action='store_true', help='none')
parser.add_argument('--method', type=str, default='DCS_img', help='TTP method: {DCS_img_match, DCS_img}')
args = parser.parse_args()
print(args)

if args.source_domain == 'IN':
    hostname = socket.gethostname()
    if hostname == 'user-Precision-7920-Tower' or hostname == 'dell-Precision-7960-Tower' or hostname == 'dell-Precision-7920-Tower':  # 3091 or A6000
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
os.makedirs('10T_subsrc', exist_ok=True)
if args.hfl:
    logfile = '10T_subsrc/none_10_target_eval_eps_{}_{}_to_{}_hfl_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.gs:
    logfile = '10T_subsrc/none_10_target_eval_eps_{}_{}_to_{}_gs_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.test:
    logfile ='10T_subsrc/test.log'
elif args.none:
    logfile = '10T_subsrc/none/none_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'GAP':
    logfile = '10T_subsrc/GAP_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'CDA':
    logfile = '10T_subsrc/CDA_10_target_eval_eps_{}_{}_to_{}_NRP_{}_JPEG_{}_BR_{}_MS_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP, args.JPEG, args.BR, args.MS)
elif args.method == 'C-GSP':
    logfile = '10T_subsrc/C-GSP/C-GSP_10_target_eval_eps_{}_{}_to_{}_NRP_{}_BR_{}_MS_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP, args.BR, args.MS)
elif args.method == 'C-GSP_new':
    logfile = '10T_subsrc/C-GSP_new/C-GSP_new_10_target_eval_eps_{}_{}_to_{}_NRP_{}_BR_{}_MS_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP, args.BR, args.MS)
elif args.method == 'TTP':
    logfile = '10T_subsrc/TTP_10_target_eval_eps_{}_{}_to_{}_NRP_{}_JPEG_{}_BR_{}_MS_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP, args.JPEG, args.BR, args.MS)
elif args.method == 'ESMA':
    logfile = '10T_subsrc/ESMA_10_target_eval_eps_{}_{}_to_{}_NRP_{}_JPEG_{}_BR_{}_MS_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP, args.JPEG, args.BR, args.MS)
elif args.method == 'TTAA_20':
    logfile = '10T_subsrc/TTAA_20epoch_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'TTAA_60':
    logfile = '10T_subsrc/TTAA_60epoch_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'tar_ssa':
    logfile = '10T_subsrc/tar_ssa_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_argu_gs_0.05':
    logfile = '10T_subsrc/DCS_img_argu_gs_0.05_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_0.002':
    logfile = '10T_subsrc/DCS_img_match_0.002_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match':
    logfile = '10T_subsrc/DCS_img_match_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_0.05':
    logfile = '10T_subsrc/DCS_img_match_0.05_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_0.1':
    logfile = '10T_subsrc/DCS_img_match_0.1_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_0.2':
    logfile = '10T_subsrc/DCS_img_match_0.2_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_1.0':
    logfile = '10T_subsrc/DCS_img_match_1.0_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_argu_gs_0.01':
    logfile = '10T_subsrc/DCS_img_match_argu_gs_0.01_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_argu_gs_0.05':
    logfile = '10T_subsrc/DCS_img_match_argu_gs_0.05_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_argu_gs_sim_0.01':
    logfile = '10T_subsrc/DCS_img_match_argu_gs_sim_0.01_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_fea':
    logfile = '10T_subsrc/DCS_img_match_fea_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_fea_0.05':
    logfile = '10T_subsrc/DCS_img_match_fea_0.05_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_plus_fea':
    logfile = '10T_subsrc/DCS_img_match_plus_fea_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_plus_fea_0.05':
    logfile = '10T_subsrc/DCS_img_match_plus_fea_0.05_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_gs':
    logfile = '10T_subsrc/DCS_img_gs_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_single':
    logfile = '10T_subsrc/DCS_img_match_single_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_plus_fea_argu_gs':
    logfile = '10T_subsrc/DCS_img_match_plus_fea_argu_gs_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'img_match_attn_mask_ssa':
    logfile = '10T_subsrc/img_match_attn_mask_ssa_0.05_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'attn_mask':
    logfile = '10T_subsrc/attn_mask_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'attn_mask_0.05':
    logfile = '10T_subsrc/attn_mask_0.05_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'attn_mask_pro':
    logfile = '10T_subsrc/attn_mask_pro_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'attn_mask_pro_plus':
    logfile = '10T_subsrc/attn_mask_pro_plus_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'attn_amp':
    logfile = '10T_subsrc/attn_amp_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_attn_mask_pro_plus_fea':
    logfile = '10T_subsrc/DCS_img_match_attn_mask_pro_plus_fea_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_attn_mask_pro_plus_argu_gs':
    logfile = '10T_subsrc/DCS_img_match_attn_mask_pro_plus_argu_gs_3_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_attn_mask_pro_plus_fea_argu_gs':
    logfile = '10T_subsrc/DCS_img_match_attn_mask_pro_plus_fea_argu_gs_10_target_eval_eps_{}_{}_to_{}_NRP_{}_JPEG_{}_BR_{}_MS_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP, args.JPEG, args.BR, args.MS)

elif args.method == 'comic_books_none':
    logfile = '10T_subsrc/comic_books_none_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'comic_books_TTP':
    logfile = '10T_subsrc/comic_books_TTP_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'comic_books_DCS_img_match':
    logfile = '10T_subsrc/comic_books_DCS_img_match_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'comic_books_DCS_img':
    logfile = '10T_subsrc/comic_books_DCS_img_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)

# layer
elif args.method == 'DCS_img_match_fea_0.05_layer1':
    logfile = '10T_subsrc/layer/DCS_img_match_fea_0.05_layer1_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_fea_0.05_layer2':
    logfile = '10T_subsrc/layer/DCS_img_match_fea_0.05_layer2_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_fea_0.05_layer3':
    logfile = '10T_subsrc/layer/DCS_img_match_fea_0.05_layer3_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_fea_0.05_layer4':
    logfile = '10T_subsrc/layer/DCS_img_match_fea_0.05_layer4_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)

# denseblock
elif args.method == 'DCS_img_match_fea_0.05_denseblock1':
    logfile = '10T_subsrc/denseblock/DCS_img_match_fea_0.05_denseblock1_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_fea_0.05_denseblock2':
    logfile = '10T_subsrc/denseblock/DCS_img_match_fea_0.05_denseblock2_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_fea_0.05_denseblock3':
    logfile = '10T_subsrc/denseblock/DCS_img_match_fea_0.05_denseblock3_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)
elif args.method == 'DCS_img_match_fea_0.05_denseblock4':
    logfile = '10T_subsrc/denseblock/DCS_img_match_fea_0.05_denseblock4_10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)

else:
    logfile = '10T_subsrc/10_target_eval_eps_{}_{}_to_{}_NRP_{}.log'.format(args.eps, args.source_model, args.target_model, args.NRP)

logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=logfile)

eps = args.eps/255.0

# Set-up Kernel
kernel_size = 3
pad = 2
sigma = 1
kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()

# Load Targeted Model
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
if args.target_model == 'inception_v3':
    model = timm.create_model('inception_v3', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_v3.tf_in1k/model.safetensors')
elif args.target_model in model_names:
    model = models.__dict__[args.target_model](pretrained=True)
elif args.target_model == 'inception_v4':
    model = timm.create_model('inception_v4', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_v4.tf_in1k/model.safetensors')
elif args.target_model == 'incres_v2':
    model = timm.create_model('inception_resnet_v2', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_resnet_v2.tf_in1k/model.safetensors')
elif args.target_model == 'vit':    # in21k_ft1k
    model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors')
elif args.target_model == 'swin':   # in1k
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, pretrained_cfg_overlay=dict(file='/home/sjq/.cache/huggingface/hub/models--timm--swin_base_patch4_window7_224.ms_in22k_ft_in1k/model.safetensors'))
# adversarial training
elif args.target_model == "ens_adv_inception_resnet_v2":
    model = timm.create_model("ens_adv_inception_resnet_v2", pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_resnet_v2.tf_ens_adv_in1k/model.safetensors')
elif args.target_model == "adv_inc_v3":
    model = timm.create_model('inception_v3.tf_adv_in1k', pretrained=False, checkpoint_path='/home/sjq/.cache/huggingface/hub/models--timm--inception_v3.tf_adv_in1k/model.safetensors')

#
elif args.target_model == 'SIN':
    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/resnet50_train_60_epochs-c8e5653e.pth.tar')
    model.load_state_dict(checkpoint["state_dict"])
elif args.target_model == 'SIN_IN':
    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar')
    model.load_state_dict(checkpoint["state_dict"])
elif args.target_model == 'SIN_IN_FIN':
    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar')
    model.load_state_dict(checkpoint["state_dict"])
elif args.target_model == 'Augmix':
    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/checkpoint.pth.tar')
    model.load_state_dict(checkpoint["state_dict"])
else:
    assert (args.target_model in model_names), 'Please provide correct target model names: {}'.format(model_names)

model = model.to(device)
model.eval()

# Input preprocessing Defenses
if args.NRP:
    purifier = NRP(3, 3, 64, 23)
    purifier.load_state_dict(torch.load('defense_models/NRP/NRP.pth'))
    purifier = purifier.to(device)
if args.JPEG:
    jpeg = JpegCompression(clip_values=(0, 1), quality=args.jpeg_quality, channels_first=True, apply_fit=False, apply_predict=True, verbose=False)
if args.BR:
    br = FeatureSqueezing(clip_values=(0, 1), bit_depth=args.bit_depth, apply_fit=False, apply_predict=True)
if args.MS:
    ms = SpatialSmoothing(clip_values=(0, 1), apply_fit=False, apply_predict=True, channels_first=True, window_size=args.window_size)
# if args.RS:
#     rs = models.resnet50(pretrained=False)
#     checkpoint = {}
#     if args.noise == 0.00:
#         ckpt = torch.load('/home/sjq/Base_Generate_Target_Attack/defense_models/RS/resnet50/noise_0.00/checkpoint.pth.tar')
#         checkpoint = {k.replace('1.module.', ''): v for k, v in ckpt['state_dict'].items()}
#         rs.load_state_dict(checkpoint)
#     elif args.noise == 0.25:
#         rs.load_state_dict(torch.load('/home/sjq/Base_Generate_Target_Attack/defense_models/RS/resnet50/noise_0.25/checkpoint.pth.tar'))
#     elif args.noise == 0.50:
#         rs.load_state_dict(torch.load('/home/sjq/Base_Generate_Target_Attack/defense_models/RS/resnet50/noise_0.50/checkpoint.pth.tar'))
#     elif args.noise == 1.00:
#         rs.load_state_dict(torch.load('/home/sjq/Base_Generate_Target_Attack/defense_models/RS/resnet50/noise_1.00/checkpoint.pth.tar'))
#     rs = rs.to(device)
# if args.HGD:
#     config, resmodel = get_model()
#     net = resmodel.net
#     checkpoint = torch.load('denoise_res_015.ckpt')
#     if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#         resmodel.load_state_dict(checkpoint['state_dict'])
#     else:
#         resmodel.load_state_dict(checkpoint)
#     resmodel = resmodel.to(device)
#     resmodel.eval()


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
    targets = [99,245,344,471,555,661,701,802]
    # targets = [344,701]
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

    if args.re_noise:
        netG = GeneratorResnet_R()
    elif 'GAP' in args.method:
        netG = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=[0])
    elif args.method == 'C-GSP':
        netG = ConGeneratorResnet(nz=16, layer=1)
    elif args.method == 'C-GSP_new':
        netG = ConGeneratorResnet_adv(nz=16, layer=1)
    elif args.method == 'ESMA':
        netG = Generator(num_target=len(targets), ch=32, ch_mult=[1,2,3,4], num_res_blocks=1)
    else:
        netG = GeneratorResnet()

    # if args.fl:
    #     checkpoint={}
    #     ckpt = torch.load('pretrained_generators/My/fl_0.1/netG_{}_19_fl_t{}.pth'.format(args.source_model, target))
    # elif args.fn:
    #     checkpoint={}
    #     ckpt = torch.load('pretrained_generators/My/fn/netG_{}_19_fn_t{}.pth'.format(args.source_model, target))
    #     prototypes = torch.load('pretrained_generators/My/fn/prototype_{}_19_t{}.pth'.format(args.source_model, target))
    #     netG.prototype_1 = prototypes['prototype_1']
    #     netG.prototype_2 = prototypes['prototype_2']
    #     netG.prototype_3 = prototypes['prototype_3']
    # elif args.none:
    #     checkpoint={}
    #     ckpt = torch.load('pretrained_generators/none_0.2_10/{0}/{1}/netG_{0}_19_t{1}.pth'.format(args.source_model, target))
    # else:
    #     checkpoint={}
    #     # ckpt = torch.load('pretrained_generators/none/netG_{0}_19_t{1}.pth'.format(args.source_model, target))
    #     ckpt = torch.load('pretrained_generators/TTP_new_0.2_10/{0}/{1}/netG_{0}_19_ttp_t{1}.pth'.format(args.source_model, target))
    #     # ckpt = torch.load('TTP_models/netG_{}_IN_19_{}.pth'.format(args.source_model, target))
    #     # print('TTP')

    if args.hfl:
        checkpoint={}
        ckpt = torch.load('pretrained_generators/My/hfl/resnet50/{1}/netG_{0}_19_hfl_t{1}.pth'.format(args.source_model, target))
    elif args.none:
        checkpoint={}
        # ckpt = torch.load('pretrained_generators/none_0.2_10/{0}/{1}/netG_{0}_19_t{1}.pth'.format(args.source_model, target))
        ckpt = torch.load('pretrained_generators/none_0.2_10/{0}/{1}/netG_{0}_19_t{1}.pth'.format(args.source_model, target))
    elif args.gs:
        checkpoint={}
        ckpt = torch.load('pretrained_generators/TTP_new_0.2_10/gs/resnet50/{1}/netG_{0}_19_gs_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'GAP':
        checkpoint={}
        ckpt = torch.load('GAP/checkpoint/resnet50/{1}/netG_{0}_epoch_20_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'CDA':
        checkpoint={}
        ckpt = torch.load('CDA/{0}/{1}/netG_{0}_19_rl_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'C-GSP':
        checkpoint={}
        ckpt = torch.load('C-GSP/checkpoints/{0}/model-{0}-AE-epoch9.pth'.format(args.source_model))
    elif args.method == 'C-GSP_new':
        checkpoint={}
        ckpt = torch.load('C-GSP/new_checkpoints/model-{0}-AE-{1}-epoch7.pth'.format(args.source_model, args.alpha))
    elif args.method == 'TTP':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/TTP_new_0.2_10/{0}/{1}/netG_{0}_19_ttp_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'ESMA':
        checkpoint={}
        ckpt = torch.load('ESMA/ESMA_Checkpoints/ckpt_299_{0}_.pt'.format(args.source_model))
    elif args.method == 'TTAA_20':
        checkpoint={}
        ckpt = torch.load('TTAA/resnet50/{1}/netG_{0}_ttaa_19_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'TTAA_60':
        checkpoint={}
        ckpt = torch.load('TTAA/resnet50/{1}/netG_{0}_ttaa_59_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'tar_ssa':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/My/tar_ssa/resnet50/{1}/netG_{0}_19_tssa_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_argu_gs_0.05':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_argu_gs_0.05/resnet50/{1}/netG_{0}_19_DCS_argu_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_0.002':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_0.002/{0}/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match/{0}/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_0.05':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_0.05/{0}/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_0.1':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_0.1/{0}/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_0.2':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_0.2/{0}/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_1.0':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_1.0/{0}/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_argu_gs_0.01':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_argu_gs/resnet50/{1}/netG_{0}_19_img_match_argu_gs_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_argu_gs_0.05':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_argu_gs_0.05/{0}/{1}/netG_{0}_19_img_match_argu_gs_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_argu_gs_sim_0.01':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/My/DCS_img_match_argu_gs_sim_0.01/resnet50/{1}/netG_{0}_19_img_match_argu_gs_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_fea':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_fea/resnet50/{1}/netG_{0}_19_DCS_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_fea_0.05':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_fea_0.05/{0}/{1}/netG_{0}_19_DCS_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_plus_fea':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_plus_fea/resnet50/{1}/netG_{0}_19_DCS_img_match_plus_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_plus_fea_0.05':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/My/DCS_img_match_plus_fea_0.05/resnet50/{1}/netG_{0}_19_DCS_img_match_plus_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_gs':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/My/DCS_img_gs/resnet50/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_single':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/My/DCS_img_match_single/resnet50/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'img_match_attn_mask_ssa':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/img_match_attn_mask_ssa/resnet50/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'attn_mask':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/img_match_attn_mask/resnet50/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'attn_mask_0.05':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/img_match_attn_mask_0.05/{0}/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'attn_mask_pro':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/img_match_attn_mask_pro/{0}/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'attn_mask_pro_plus':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/img_match_attn_mask_pro_plus/{0}/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'attn_amp':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/amp_attn/resnet50/{1}/netG_{0}_19_amp_attn_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_plus_fea_argu_gs':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_plus_fea_argu_gs/{0}/{1}/netG_{0}_19_DCS_img_match_plus_fea_argu_gs_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_attn_mask_pro_plus_fea':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_attn_mask_pro_plus_fea/{0}/{1}/netG_{0}_19_DCS_img_match_attn_mask_plus_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_attn_mask_pro_plus_argu_gs':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_attn_mask_pro_plus_argu_gs_3/{0}/{1}/netG_{0}_19_DCS_img_match_attn_mask_plus_argu_gs_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_attn_mask_pro_plus_fea_argu_gs':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_attn_mask_pro_plus_fea_argu_gs_3/{0}/{1}/netG_{0}_19_DCS_img_match_attn_mask_plus_fea_argu_gs_t{1}.pth'.format(args.source_model, target))

    elif args.method == 'comic_books_none':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/comic_books/none/resnet50/{1}/netG_{0}_19_none_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'comic_books_TTP':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/comic_books/TTP/resnet50/{1}/netG_{0}_19_ttp_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'comic_books_DCS_img_match':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/comic_books/DCS_img_match/resnet50/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'comic_books_DCS_img':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/comic_books/DCS_img/resnet50/{1}/netG_{0}_19_DCS_t{1}.pth'.format(args.source_model, target))

    # layer
    elif args.method == 'DCS_img_match_fea_0.05_layer1':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_fea_0.05_layer1/resnet50/{1}/netG_resnet50_19_DCS_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_fea_0.05_layer2':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_fea_0.05_layer2/resnet50/{1}/netG_resnet50_19_DCS_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_fea_0.05_layer3':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_fea_0.05_layer3/resnet50/{1}/netG_resnet50_19_DCS_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_fea_0.05_layer4':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_fea_0.05_layer4/resnet50/{1}/netG_resnet50_19_DCS_fea_t{1}.pth'.format(args.source_model, target))

    # denseblock
    elif args.method == 'DCS_img_match_fea_0.05_denseblock1':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_fea_0.05_denseblock1/densenet121/{1}/netG_densenet121_19_DCS_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_fea_0.05_denseblock2':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_fea_0.05_denseblock2/densenet121/{1}/netG_densenet121_19_DCS_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_fea_0.05_denseblock3':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_fea_0.05_denseblock3/densenet121/{1}/netG_densenet121_19_DCS_fea_t{1}.pth'.format(args.source_model, target))
    elif args.method == 'DCS_img_match_fea_0.05_denseblock4':
        checkpoint={}
        ckpt = torch.load('pretrained_generators/DCS_img_match_fea_0.05_denseblock4/densenet121/{1}/netG_densenet121_19_DCS_fea_t{1}.pth'.format(args.source_model, target))

    else:
        checkpoint={}
        # ckpt = torch.load('pretrained_generators/none/netG_{0}_19_t{1}.pth'.format(args.source_model, target))
        # ckpt = torch.load('pretrained_generators/TTP_new_0.2_10/{0}/{1}/netG_{0}_19_ttp_t{1}.pth'.format(args.source_model, target))
        # ckpt = torch.load('TTP_models/netG_{}_IN_19_{}.pth'.format(args.source_model, target))
        # print('TTP')

    if list(ckpt.keys())[0].startswith('module'):
        for k in ckpt.keys():
            checkpoint[k[7:]]=ckpt[k]
    else:
        checkpoint = ckpt
    netG.load_state_dict(checkpoint)
    netG = netG.to(device)
    netG.eval()

    # Reset Metrics
    acc=0
    distance = 0
    if args.RS:
        print('Target RS, noise: {}'.format(args.noise))
    else:
        print('Target model: {}, Target: {}'.format(args.target_model, target))

    if 'GAP' in args.method:
        def normalize_and_scale(delta_im, mode='train'):
            # if args.target_model == 'inception_v3':
            #     delta_im = nn.ConstantPad2d((0, -1, -1, 0), 0)(delta_im)  # crop slightly to match inception

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

        for i, (img, label) in enumerate(test_loader):
            # if (i+1) % 10 == 0:
            #     print('At Batch:', i+1)
            img, label = img.cuda(), label.cuda()

            target_label = torch.LongTensor(img.size(0))
            target_label.fill_(target)
            target_label = target_label.cuda()

            delta_im = netG(img).detach()
            delta_im = normalize_and_scale(delta_im, 'test')

            recons = torch.add(img, delta_im[0:img.size(0)])

            for cii in range(3):
                recons[:, cii, :, :] = recons[:, cii, :, :].clone().clamp(img[:, cii, :, :].min(),
                                                                          img[:, cii, :, :].max())

            out = model(recons)
            acc += torch.sum(out.argmax(dim=-1) == target_label).item()

            distance += (img - recons).max() * 255

    else:
        for i, (img, label) in tqdm(enumerate(test_loader)):
            # print('Target: {}, Batch: {}/{}'.format(target, i+1, len(test_loader)))
            img, label = img.to(device), label.to(device)

            target_label = torch.LongTensor(img.size(0))
            target_label.fill_(target)
            target_label = target_label.to(device)

            if 'TTP' in args.method or 'gs' in args.method:
                adv = kernel(netG(img)).detach()
                # adv = netG(img).detach()
            elif args.method == 'C-GSP':
                target_one_hot = torch.zeros(img.size(0), 1000, device='cuda').scatter_(1, target_label.unsqueeze(1), 1)
                noise = netG(img, target_one_hot, eps=eps).detach()
                adv = img + noise
            elif args.method == 'C-GSP_new':
                target_one_hot = torch.zeros(img.size(0), 1000, device='cuda').scatter_(1, target_label.unsqueeze(1), 1)
                adv = netG(img, target_one_hot, eps=eps).detach()
                adv = kernel(adv).detach()
            elif args.method == 'ESMA':
                target_idx = targets.index(target) * torch.ones(img.size(0), dtype=torch.long).to(device) # 0~99
                adv = netG(img, target_idx).detach()
                adv = kernel(adv).detach()
            else:
                # adv = kernel(netG(img)).detach()
                adv = netG(img).detach()
            adv = torch.min(torch.max(adv, img - eps), img + eps)
            adv = torch.clamp(adv, 0.0, 1.0)

            if args.NRP:
                # Purify Adversary
                adv = purifier(adv).detach()

            elif args.JPEG:
                adv = adv.detach().cpu().numpy()
                adv = jpeg(adv)[0]
                adv = torch.from_numpy(adv).to(device)

            elif args.BR:
                adv = adv.detach().cpu().numpy()
                adv = br(adv)[0]
                adv = torch.from_numpy(adv).to(device)

            elif args.MS:
                adv = adv.detach().cpu().numpy()
                adv = ms(adv)[0]
                adv = torch.from_numpy(adv).to(device)


            out = model(normalize(adv.clone().detach()))
            acc += torch.sum(out.argmax(dim=-1) == target_label).item()

            distance +=(img - adv).max() * 255

    total_acc+=acc
    total_samples+=test_size
    logger.info('{:<10} {:<10} {:<10.4f} {:<10.4f}'.format(int(eps * 255), target, acc / test_size, distance / (i + 1)))

logger.info('Average Target Transferability')
logger.info('{:<10} {:<10.4f} {:<10.4f}'.format(int(eps * 255), total_acc / total_samples, distance / (i + 1)))
logger.info('-'*100)