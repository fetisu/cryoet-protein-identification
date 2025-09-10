# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ConvNeXtBlock3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconvx3 = nn.Conv3d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.pwconv1 = nn.Conv3d(dim, dim * 4, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(dim * 4, dim, kernel_size=1, bias=False)

    def forward(self, x):
        input = x
        x = self.dwconvx3(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + x
        return x


class Stem3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()
        self.stem = nn.Conv3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self._norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def forward(self, x):
        x = self.stem(x)
        x = self._norm(x)
        return x


class ConvNeXt3DEncoderUNetStyle(nn.Module):
    def __init__(self, in_ch=1, dims=[64, 128, 256, 512], depths=[3, 3, 3, 3]):
        super().__init__()
        self.stem = Stem3D(in_ch, dims[0])

        self.downsample_layers = nn.ModuleList(
            [
                nn.Identity(),
                nn.Sequential(
                    nn.GroupNorm(num_groups=1, num_channels=dims[0]),
                    nn.Conv3d(
                        dims[0],
                        dims[1],
                        kernel_size=(2, 2, 2),
                        stride=(2, 2, 2),
                    ),
                ),
                nn.Sequential(
                    nn.GroupNorm(num_groups=1, num_channels=dims[1]),
                    nn.Conv3d(
                        dims[1],
                        dims[2],
                        kernel_size=(2, 2, 2),
                        stride=(2, 2, 2),
                    ),
                ),
                nn.Sequential(
                    nn.GroupNorm(num_groups=1, num_channels=dims[2]),
                    nn.Conv3d(
                        dims[2],
                        dims[3],
                        kernel_size=(2, 2, 2),
                        stride=(2, 2, 2),
                    ),
                ),
            ]
        )

        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(ConvNeXtBlock3D(dim=dims[i]))
            cur += depths[i]
            self.stages.append(nn.Sequential(*blocks))
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dims[-1])

    def forward(self, x):
        f0 = self.stem(x)

        # stage0
        # no downsampling
        f0 = self.stages[0](f0)
        # stage1
        f1 = self.downsample_layers[1](f0)
        f1 = self.stages[1](f1)
        # stage2
        f2 = self.downsample_layers[2](f1)
        f2 = self.stages[2](f2)
        # stage3
        f3 = self.downsample_layers[3](f2)
        f3 = self.stages[3](f3)
        f3 = self.norm(f3)

        return [x, f0, f1, f2, f3]


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=in_channels,
        )
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(
            in_channels, in_channels * 3, kernel_size=1, bias=False
        )
        self.conv3 = nn.Conv3d(
            in_channels * 3, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        return x


class Decoder3D(nn.Module):
    def __init__(
        self,
        encoder_dims=[64, 128, 256, 512],
        decoder_dims=[32, 64, 128, 256],
        out_channels=1,
    ):
        super().__init__()

        # 1) f3 -> f2
        self.up3 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=True
        )
        self.dec3 = ConvBlock3D(
            in_channels=encoder_dims[3] + encoder_dims[2],
            out_channels=decoder_dims[2],
        )
        # 2) f2 -> f1
        self.up2 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=True
        )
        self.dec2 = ConvBlock3D(
            in_channels=decoder_dims[2] + encoder_dims[1],
            out_channels=decoder_dims[1],
        )
        # 3) f1 -> f0
        self.up1 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=True
        )
        self.dec1 = ConvBlock3D(
            in_channels=decoder_dims[1] + encoder_dims[0],
            out_channels=decoder_dims[0],
        )
        # 4) f0 -> x
        self.up0 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=True
        )
        self.dec0 = ConvBlock3D(
            in_channels=decoder_dims[0], out_channels=decoder_dims[0]
        )
        self.final_conv = nn.Conv3d(
            decoder_dims[0], out_channels, kernel_size=1
        )

    def forward(self, features):
        x, f0, f1, f2, f3 = features

        # --- 1) f3 -> f2 ---
        d3 = self.up3(f3)  # Upsample
        d3 = torch.cat([d3, f2], dim=1)  # skip connect
        d3 = self.dec3(d3)  # conv block

        # --- 2) f2 -> f1 ---
        d2 = self.up2(d3)
        d2 = torch.cat([d2, f1], dim=1)
        d2 = self.dec2(d2)

        # --- 3) f1 -> f0 ---
        d1 = self.up1(d2)
        d1 = torch.cat([d1, f0], dim=1)
        d1 = self.dec1(d1)

        # --- 4) f0 -> x ---
        d0 = self.up0(d1)
        d0 = self.dec0(d0)

        out = self.final_conv(d0)
        return out


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = ConvNeXt3DEncoderUNetStyle(in_channels)
        self.decoder = Decoder3D(out_channels=out_channels)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out
