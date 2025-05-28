# refer to https://github.com/Abello966/FrequencyBiasExperiments
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms, datasets
import argparse


def get_mean_energy_dataset(Xfr):
    image = transforms.Grayscale(num_output_channels=1)(Xfr)
    freq = torch.fft.fft2(image, norm="ortho")
    energy = torch.abs(freq)
    energy.squeeze_()
    energy[:, 0, 0] = 0 # 零频率分量置0
    avg_energy_fr = torch.mean(energy, dim=0)
    avg_energy_fr = torch.fft.fftshift(avg_energy_fr)

    return avg_energy_fr


def get_percentage_masks_relevance(relevance, percent):
    range_result = [0]  # range_result on ImageNet valid: [0,7,16,26,38,51,66,83,102,127,224]
    last_result = 0
    for i in range(1, 100, int(percent * 100)):
        next_result = last_result
        mask = get_mask(relevance, float('inf'), float('inf'))

        while percent_cut_relevance(mask, relevance) < percent and next_result < (relevance.shape[0]):
            # print(percent_cut_relevance(mask, relevance))
            last_pct = percent_cut_relevance(mask, relevance)
            next_result = next_result + 1
            mask = get_mask(relevance, last_result, next_result)
            if abs(percent_cut_relevance(mask, relevance) - percent) > abs(last_pct - percent):
                next_result -= 1
                break

        range_result.append(next_result)
        last_result = next_result
    return range_result


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

    height, width = image.shape[0:2]
    mask = torch.ones((height, width), dtype=torch.uint8).to(image.device)
    center_height, is_even_height = divmod(height, 2)
    center_width, is_even_width = divmod(width, 2)

    if is_even_height == 0:
        center_height -= 0.5
    if is_even_width == 0:
        center_width -= 0.5

    y, x = torch.meshgrid(torch.arange(-center_height, height - center_height, device=image.device),
                            torch.arange(-center_width, width - center_width, device=image.device))
    mask_area = (torch.sqrt(x**2 + y**2) > radius1) & (torch.sqrt(x**2 + y**2) <= radius2)
    mask[mask_area] = 0
    return mask


def percent_cut_relevance(mask, relevance):
    return (torch.sum((mask == 0) * relevance) / torch.sum(relevance)).item()


def main():
    parser = argparse.ArgumentParser(description='Get mean energy of dataset')
    # parser.add_argument('--data_dir', type=str, default='E:/datasets/ILSVRC2012', help='dataset path')
    parser.add_argument('--data_dir', type=str, default='/datasets/Imagenet2012', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    image_datasets = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8, pin_memory=True)

    avg_energy_fr = torch.zeros(224, 224).to(device)

    for i, (img, _) in enumerate(dataloaders):
        img = img.to(device)
        avg_energy_fr += get_mean_energy_dataset(img)
        print(i)
    avg_energy_fr /= len(dataloaders)

    emp_dist = avg_energy_fr
    # emp_dist = torch.rand(224, 224).to(device)

    percent_range = get_percentage_masks_relevance(emp_dist, 0.1)




if __name__ == "__main__":
    main()
