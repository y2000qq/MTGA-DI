import os
import random
import shutil
import socket
from collections import Counter

import torchvision


# seed
random.seed(0)


# 指定ImageNet训练集文件夹路径和输出文件夹路径
hostname = socket.gethostname()
if hostname in ['user-Precision-7920-Tower', 'dell-Precision-7960-Tower']:  # 3091 or A6000
    imagenet_folder = '/datasets/Imagenet2012/val'
elif hostname == 'ubuntu':  # 503
    imagenet_folder = '/datasets/ILSVRC2012/val'
elif hostname == 'R2S1-gpu':  # 5014
    imagenet_folder = '/datasets/ImageNet2012/val'

output_folder = '/datasets/ImageNet_100_for_plot'
os.makedirs(output_folder, exist_ok=True)

# for test
# test = torchvision.datasets.ImageFolder(output_folder)

imagenet_set = torchvision.datasets.ImageFolder(imagenet_folder)

# excluded_class = [24, 245, 471, 555, 661, 701, 802, 919]
# excluded_class_n = []
# target_dict = { 3 : 'n01491361',
#                16 : 'n01560419',
#                24 : 'n01622779',
#                36 : 'n01667778',
#                48 : 'n01695060',
#                52 : 'n01728572',
#                69 : 'n01768244',
#                71 : 'n01770393',
#                85 : 'n01806567',
#                99 : 'n01855672',
#                107 : 'n01910747',
#                114 : 'n01945685',
#                130 : 'n02007558',
#                138 : 'n02018795',
#                142 : 'n02033041',
#                151 : 'n02085620',
#                162 : 'n02088364',
#                178 : 'n02092339',
#                189 : 'n02095570',
#                193 : 'n02096294',
#                207 : 'n02099601',
#                212 : 'n02100735',
#                228 : 'n02105505',
#                240 : 'n02107908',
#                245 : 'n02108915',
#                260 : 'n02112137',
#                261 : 'n02112350',
#                276 : 'n02117135',
#                285 : 'n02124075',
#                291 : 'n02129165',
#                309 : 'n02206856',
#                317 : 'n02259212',
#                328 : 'n02319095',
#                340 : 'n02391049',
#                344 : 'n02398521',
#                358 : 'n02443114',
#                366 : 'n02480855',
#                374 : 'n02488291',
#                390 : 'n02526121',
#                393 : 'n02607072',
#                404 : 'n02690373',
#                420 : 'n02787622',
#                430 : 'n02802426',
#                438 : 'n02815834',
#                442 : 'n02825657',
#                453 : 'n02870880',
#                464 : 'n02910353',
#                471 : 'n02950826',
#                485 : 'n02988304',
#                491 : 'n03000684',
#                506 : 'n03065424',
#                513 : 'n03110669',
#                523 : 'n03141823',
#                538 : 'n03220513',
#                546 : 'n03272010',
#                555 : 'n03345487',
#                569 : 'n03417042',
#                580 : 'n03457902',
#                582 : 'n03461385',
#                599 : 'n03530642',
#                605 : 'n03584254',
#                611 : 'n03598930',
#                629 : 'n03676483',
#                638 : 'n03710637',
#                646 : 'n03733281',
#                652 : 'n03763968',
#                661 : 'n03777568',
#                678 : 'n03814639',
#                689 : 'n03866082',
#                701 : 'n03888257',
#                707 : 'n03902125',
#                717 : 'n03930630',
#                724 : 'n03947888',
#                735 : 'n03980874',
#                748 : 'n04026417',
#                756 : 'n04049303',
#                766 : 'n04111531',
#                779 : 'n04146614',
#                786 : 'n04179913',
#                791 : 'n04204347',
#                802 : 'n04252077',
#                813 : 'n04270147',
#                827 : 'n04330267',
#                836 : 'n04355933',
#                849 : 'n04398044',
#                859 : 'n04442312',
#                866 : 'n04465501',
#                879 : 'n04507155',
#                885 : 'n04525038',
#                893 : 'n04548362',
#                901 : 'n04579145',
#                919 : 'n06794110',
#                929 : 'n07615774',
#                932 : 'n07695742',
#                946 : 'n07730033',
#                958 : 'n07802026',
#                963 : 'n07873807',
#                980 : 'n09472597',
#                984 : 'n11879895',
#                992 : 'n12998815'
# }
# for i in excluded_class:
#     excluded_class_n.append(target_dict[i])
#
# # imagenet_set.classes排除掉excluded_class_n
# imagenet_set.classes = [i for i in imagenet_set.classes if i not in excluded_class_n]
# 随机选择50,000次类别

k = 100
selected_classes = random.choices(imagenet_set.classes, k=k)
counts = Counter(selected_classes)

for class_label, count in counts.items():
    class_label = class_label.strip()

    # 创建类别文件夹
    class_folder = os.path.join(output_folder, class_label)
    os.makedirs(class_folder, exist_ok=True)

    # 获取该类别下的所有图片文件
    class_image_folder = os.path.join(imagenet_folder, class_label)
    images = os.listdir(class_image_folder)

    # 抽取该类别下的count张图片
    random.seed(0)
    selected_images = random.sample(images, k=count)

    for selected_image in selected_images:
        source_image_path = os.path.join(class_image_folder, selected_image)
        destination_image_path = os.path.join(class_folder, selected_image)
        shutil.copy(source_image_path, destination_image_path)