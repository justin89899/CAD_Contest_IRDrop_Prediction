""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

from .unet_parts_n_v2 import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 32)
        # self.down1 = Down(32, 64)
        # self.down2 = Down(64, 128)
        # self.down3 = Down(128, 128)
        # print("Small model")
        # self.up2 = Up(256, 64, bilinear)
        # self.up3 = Up(128, 32, bilinear)
        # self.up4 = Up(64, 32, bilinear)
        # self.outc = OutConv(32, n_classes)

        # self.inc = DoubleConv(n_channels, 16, 7)
        # self.down1 = Down(16, 32, 7)
        # self.down2 = Down(32, 64, 7)
        # self.down3 = Down(64, 128, 7)
        # self.down4 = Down(128, 128, 7)
        # print("Big model")
        # self.up1 = Up(256, 64, 7, bilinear)
        # self.up2 = Up(128, 32, 7, bilinear)
        # self.up3 = Up(64, 16, 7, bilinear)
        # self.up4 = Up(32, 16, 7, bilinear)
        # self.outc = OutConv(16, n_classes)

        # self.inc = DoubleConv(n_channels, 16, 5)
        # self.down1 = Down(16, 32, 5)
        # self.down2 = Down(32, 64, 5)
        # self.down3 = Down(64, 128, 5)
        # self.down4 = Down(128, 128, 5)
        # print("Big model")
        # self.up1 = Up(256, 64, 5, bilinear)
        # self.up2 = Up(128, 32, 5, bilinear)
        # self.up3 = Up(64, 16, 5, bilinear)
        # self.up4 = Up(32, 16, 5, bilinear)
        # self.outc = OutConv(16, n_classes)

        self.inc = DoubleConv(n_channels, 16, 5)
        self.down1 = Down(16, 32, 5)
        self.down2 = Down(32, 64, 5)
        self.down3 = Down(64, 128, 5)
        self.down4 = Down(128, 128, 5)
        self.up1 = Up(256, 64, 5, bilinear)
        self.up2 = Up(128, 32, 5, bilinear)
        self.up3 = Up(64, 16, 5, bilinear)
        self.up4 = Up_no_concat(16, 16, 5, bilinear)
        self.outc = OutConv(16, n_classes)

        # self.inc = DoubleConv(n_channels, 16, 7)
        # self.down1 = Down(16, 32, 7)
        # self.down2 = Down(32, 64, 7)
        # self.down3 = Down(64, 64, 7)
        # print("Big model")
        # self.up1 = Up(128, 32, 7, bilinear)
        # self.up2 = Up(64, 16, 7, bilinear)
        # self.up3 = Up(32, 8, 7, bilinear)
        # self.outc = OutConv(8, n_classes)

        # self.inc = DoubleConv(n_channels, 32)
        # self.down1 = Down(32, 64)
        # self.down2 = Down(64, 128)
        # self.down3 = Down(128, 256)
        # self.down4 = Down(256, 256)
        # print("Big model")
        # self.up1 = Up(512, 128, bilinear)
        # self.up2 = Up(256, 64, bilinear)
        # self.up3 = Up(128, 32, bilinear)
        # self.up4 = Up(64, 32, bilinear)
        # self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x = self.up1(x4, x3)
        # x = self.up2(x, x2)
        # x = self.up3(x, x1)
        # logits = self.outc(x)
        # return logits
    
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    # print(net)
