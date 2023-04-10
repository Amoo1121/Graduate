import torch.nn as nn


class AENetwork(nn.Module):

    def __init__(self):
        super(AENetwork, self).__init__()
        base_channel = 64  # 基本通道数
        input_channel = 3  # 输入数据通道数
        potential_feature = 100  # 低维空间潜在特征
        # self.encoder1 = nn.Sequential(
        #     nn.Conv2d(input_channel, base_channel, 4, 2, 1, bias=False),
        # )
        # self.encoder2 = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.Conv2d(base_channel, base_channel << 1, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(base_channel << 1),  # 128
        # )
        # self.encoder3 = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.Conv2d(base_channel << 1, base_channel << 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(base_channel << 2),
        # )
        # self.encoder4 = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.Conv2d(base_channel << 2, potential_feature, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(potential_feature),
        # )
        #
        # self.decoder1 = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(potential_feature, base_channel << 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(base_channel << 2),
        # )
        # self.decoder2 = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(base_channel << 2, base_channel << 1, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(base_channel << 1),
        # )
        # self.decoder3 = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(base_channel << 1, base_channel, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(base_channel),
        # )
        # self.decoder4 = nn.Sequential(
        #     nn.ConvTranspose2d(base_channel, input_channel, 4, 2, 1, bias=False),
        #     nn.Tanh(),
        # )
$network

    def forward(self, x):
        # out = self.encoder1(x)
        # out = self.encoder2(out)
        # out = self.encoder3(out)
        # out = self.encoder4(out)
        # out = self.decoder1(out)
        # out = self.decoder2(out)
        # out = self.decoder3(out)
        # rec_img = self.decoder4(out)
        # return rec_img
