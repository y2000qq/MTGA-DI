import torch
from torch import nn
import torch.nn.functional as F

ngf = 64

# class Feature_Discriminator(nn.Module):
#     def __init__(self):
#         super(Feature_Discriminator, self).__init__()
#         # self.input_channels = input_channels
#
#         self.model = nn.Sequential(
#             nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(in_features=ngf * 8, out_features=1)
#         # self.fc = nn.Linear(in_features=ngf * 8 * 28 * 28, out_features=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.model(x)
#         x = self.global_avg_pooling(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         x = self.fc(x)
#         x = self.sigmoid(x)
#         return x

# class Feature_Discriminator(nn.Module):
#     def __init__(self):
#         super(Feature_Discriminator, self).__init__()
#         # self.input_channels = input_channels
#
#         self.model = nn.Sequential(
#             nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, bias=False),    # 56——>28
#             nn.BatchNorm2d(ngf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1, bias=False),    # 28——>14
#             nn.BatchNorm2d(ngf * 16),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(ngf * 16, ngf * 32, kernel_size=3, stride=2, padding=1, bias=False),    # 14——>7
#             nn.BatchNorm2d(ngf * 32),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(ngf * 32, 1, kernel_size=7, stride=1, padding=0, bias=False),  # 7——>1
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         x = self.model(x)
#
#         return x

# official


class Feature_Discriminator(nn.Module):
    def __init__(self, n_feature):
        super(Feature_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_feature, out_channels=128, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512*7*7, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # (batch_size,128, x/2, x/2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (batch_size,128, x/2, x/2)
        x = self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

if __name__ == '__main__':
    input_channels = 3
    model = Feature_Discriminator(input_channels)
    x = torch.randn(1, input_channels, 224, 224)
    y = model(x)
    print(model)