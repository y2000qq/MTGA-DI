import torch.nn as nn


class ReweightNet(nn.Module):
    def __init__(self, size, num_channel=64, inception=False):
        super(ReweightNet, self).__init__()
        self.inception = inception
        C, H, W = size
        self.rewe_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)


    def forward(self, img_hc):
        hc_map = self.rewe_net(img_hc)
        re_img_hc = img_hc * hc_map
        if self.inception:
            re_img_hc = self.crop(re_img_hc)

        return re_img_hc