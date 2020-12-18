import torch
from monai import networks
from torch import nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.enc0 = networks.blocks.Convolution(dimensions=2,
                                                in_channels=in_channels,
                                                out_channels=out_channels)
        self.enc1 = networks.blocks.Convolution(dimensions=2,
                                                in_channels=out_channels,
                                                out_channels=out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x1 = self.enc0(x)
        x1 = self.enc1(x1)
        return x1


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = networks.blocks.Convolution(dimensions=2,
                                              in_channels=in_channels,
                                              out_channels=in_channels // 2,
                                              is_transposed=True,
                                              kernel_size=3,
                                              strides=2)
        self.dec0 = networks.blocks.Convolution(dimensions=2,
                                                in_channels=in_channels,
                                                out_channels=out_channels)
        self.dec1 = networks.blocks.Convolution(dimensions=2,
                                                in_channels=out_channels,
                                                out_channels=out_channels)

    def forward(self, x, e):
        x1 = self.up(x)
        x1 = torch.cat((x1, e), axis=1)
        x1 = self.dec0(x1)
        x1 = self.dec1(x1)
        return x1


class PixelShuffleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PixelShuffleDecoderBlock, self).__init__()
        self.up = networks.blocks.SubpixelUpsample(dimensions=2,
                                                   in_channels=in_channels)
        self.halve = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels // 2,
                               kernel_size=1)
        self.dec0 = networks.blocks.Convolution(dimensions=2,
                                                in_channels=in_channels,
                                                out_channels=out_channels)
        self.dec1 = networks.blocks.Convolution(dimensions=2,
                                                in_channels=out_channels,
                                                out_channels=out_channels)

    def forward(self, x, e):
        x1 = self.halve(self.up(x))
        x1 = torch.cat((x1, e), axis=1)
        x1 = self.dec0(x1)
        x1 = self.dec1(x1)
        return x1


class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        # 3x3 Conv + 3x3 Conv
        self.convs = nn.ModuleList([
            nn.Sequential(
                networks.blocks.Convolution(dimensions=2,
                                            in_channels=in_ch,
                                            out_channels=out_ch * 2),
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1))
            for in_ch, out_ch in zip(input_channels, output_channels)
        ])

    def forward(self, xs: list, last_layer):
        hcs = [
            F.interpolate(c(x),
                          scale_factor=2**(len(self.convs) - i),
                          align_corners=True,
                          mode='bilinear')
            for i, (c, x) in enumerate(zip(self.convs, xs))
        ]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class SubPixelNetWithFPN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.enc0 = EncoderBlock(in_channels=3, out_channels=64)
        self.m0 = nn.MaxPool2d(kernel_size=(2, 2))
        self.enc1 = EncoderBlock(in_channels=64, out_channels=128)
        self.m1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.enc2 = EncoderBlock(in_channels=128, out_channels=256)
        self.m2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.aspp = networks.blocks.SimpleASPP(spatial_dims=2,
                                               in_channels=256,
                                               conv_out_channels=128)

        self.dec2 = PixelShuffleDecoderBlock(in_channels=512, out_channels=256)
        self.dec1 = PixelShuffleDecoderBlock(in_channels=256, out_channels=128)
        self.dec0 = PixelShuffleDecoderBlock(in_channels=128, out_channels=64)
        self.fpn = FPN([512, 256, 128], [16] * 3)
        self.drop = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(in_channels=3 * 16 + 64,
                              out_channels=1,
                              kernel_size=(1, 1))

    def forward(self, x):
        e0 = self.enc0(x)  # 64x256x256
        f0 = self.m0(e0)  # 64x128x128
        e1 = self.enc1(f0)  # 128x128x128
        f1 = self.m1(e1)  # 128x64x64
        e2 = self.enc2(f1)  # 256x64x64
        f2 = self.m2(e2)  # 256x32x32
        bottom = self.aspp(f2)  #512x32x32
        d2 = self.dec2(bottom, e2)  #256x64x64
        d1 = self.dec1(d2, e1)  #128x128x128
        d0 = self.dec0(d1, e0)  #64x256x256
        final = self.fpn([bottom, d2, d1], d0)
        return self.head(self.drop(final))


class SubPixelNet(nn.Module):
    def __init__(self):
        super(SubPixelNet, self).__init__()
        self.enc0 = EncoderBlock(in_channels=3, out_channels=64)
        self.m0 = nn.MaxPool2d(kernel_size=(2, 2))
        self.enc1 = EncoderBlock(in_channels=64, out_channels=128)
        self.m1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.enc2 = EncoderBlock(in_channels=128, out_channels=256)
        self.m2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.aspp = networks.blocks.SimpleASPP(spatial_dims=2,
                                               in_channels=256,
                                               conv_out_channels=128)

        self.dec2 = PixelShuffleDecoderBlock(in_channels=512, out_channels=256)
        self.dec1 = PixelShuffleDecoderBlock(in_channels=256, out_channels=128)
        self.dec0 = PixelShuffleDecoderBlock(in_channels=128, out_channels=64)
        self.head = nn.Conv2d(in_channels=64,
                              out_channels=1,
                              kernel_size=(1, 1))

    def forward(self, x):
        e0 = self.enc0(x)  # 64x256x256
        f0 = self.m0(e0)  # 64x128x128
        e1 = self.enc1(f0)  # 128x128x128
        f1 = self.m1(e1)  # 128x64x64
        e2 = self.enc2(f1)  # 256x64x64
        f2 = self.m2(e2)  # 256x32x32
        bottom = self.aspp(f2)  #512x32x32
        d2 = self.dec2(bottom, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        return self.head(d0)


class ASPPUnet(nn.Module):
    def __init__(self):
        super(ASPPUnet, self).__init__()
        self.enc0 = EncoderBlock(in_channels=3, out_channels=64)
        self.m0 = nn.MaxPool2d(kernel_size=(2, 2))
        self.enc1 = EncoderBlock(in_channels=64, out_channels=128)
        self.m1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.enc2 = EncoderBlock(in_channels=128, out_channels=256)
        self.m2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.aspp = networks.blocks.SimpleASPP(spatial_dims=2,
                                               in_channels=256,
                                               conv_out_channels=128)

        self.dec2 = DecoderBlock(in_channels=512, out_channels=256)
        self.dec1 = DecoderBlock(in_channels=256, out_channels=128)
        self.dec0 = DecoderBlock(in_channels=128, out_channels=64)
        self.head = nn.Conv2d(in_channels=64,
                              out_channels=1,
                              kernel_size=(1, 1))

    def forward(self, x):
        e0 = self.enc0(x)  # 64x256x256
        f0 = self.m0(e0)  # 64x128x128
        e1 = self.enc1(f0)  # 128x128x128
        f1 = self.m1(e1)  # 128x64x64
        e2 = self.enc2(f1)  # 256x64x64
        f2 = self.m2(e2)  # 256x32x32
        bottom = self.aspp(f2)  #512x32x32
        d2 = self.dec2(bottom, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        return self.head(d0)


class BasicUnet(nn.Module):
    def __init__(self):
        super(BasicUnet, self).__init__()
        self.enc0 = EncoderBlock(in_channels=3, out_channels=64)
        self.m0 = nn.MaxPool2d(kernel_size=(2, 2))
        self.enc1 = EncoderBlock(in_channels=64, out_channels=128)
        self.m1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.enc2 = EncoderBlock(in_channels=128, out_channels=256)
        self.m2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bottom = networks.blocks.Convolution(dimensions=2,
                                                  in_channels=256,
                                                  out_channels=512)
        self.dec2 = DecoderBlock(in_channels=512, out_channels=256)
        self.dec1 = DecoderBlock(in_channels=256, out_channels=128)
        self.dec0 = DecoderBlock(in_channels=128, out_channels=64)
        self.head = nn.Conv2d(in_channels=64,
                              out_channels=1,
                              kernel_size=(1, 1))

    def forward(self, x):
        e0 = self.enc0(x)  # 64x256x256
        f0 = self.m0(e0)  # 64x128x128
        e1 = self.enc1(f0)  # 128x128x128
        f1 = self.m1(e1)  # 128x64x64
        e2 = self.enc2(f1)  # 256x64x64
        f2 = self.m2(e2)  # 256x32x32
        bottom = self.bottom(f2)  #512x32x32
        d2 = self.dec2(bottom, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        return self.head(d0)
