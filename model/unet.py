import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)

        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(32, 2, 3, stride=1, padding=1)

        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim=1)
        )

    def padd(self, x, out):
        x_D = x.shape[2]
        out_D = out.shape[2]
        if x_D != out_D:
            size = x_D - out_D
            pad_dims = (
                0, 0,
                0, 0,
                size, 0,
                0, 0,
                0, 0
            )
            out = F.pad(out, pad_dims, "constant")

            print('padding after:---' + str(out.shape))
        return out

    def forward(self, x):

        out = self.encoder1(x)
        out = F.max_pool3d(out, 2, 2)
        out = F.relu(out)
        t1 = out

        # print(out.shape)
        # 128 128
        out = self.encoder2(out)
        out = F.max_pool3d(out, 2, 2)
        out = F.relu(out)
        t2 = out


        # print(out.shape)
        # 64
        out = self.encoder3(out)
        out = F.max_pool3d(out, 2, 2)
        out = F.relu(out)
        t3 = out
        # 32


        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))
        print('out---')
        print(out.shape)
        # 16 16

        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))
        # print(t3.shape)
        # print(out.shape)
        # out = self.padd(t3, out)
        out = torch.add(out, t3)
        print(out.shape)

        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), align_corners=True, mode='trilinear'))
        # print(t2.shape)
        # print(out.shape)
        # out = self.padd(t2, out)
        out = torch.add(out, t2)
        print(out.shape)

        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))
        # print(t1.shape)
        print(out.shape)
        # out = self.padd(t1, out)
        out = torch.add(out, t1)
        # print(out.shape)
        print('aaaa')

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear'))
        output4 = self.map4(out)
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        # if self.training is True:
        #      return output1, output2, output3 #, output4
        # else:
        #     return output3
        return out