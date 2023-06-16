import torch.nn as nn
from modules import DoubleConv, Down, Up, Out


class UNet3D(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet3D, self).__init__()

        # Encoder path
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = Down(32, 64)
        self.enc3 = Down(64, 128)
        self.enc4 = Down(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        # Decoder path
        self.dec4 = Up(512, 256)
        self.dec3 = Up(256, 128)
        self.dec2 = Up(128, 64)
        self.dec1 = Up(64, 32)

        # Output
        self.out = Out(32, n_classes)

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        bottleneck = self.bottleneck(enc4)

        # Decoder path
        dec4 = self.dec4(bottleneck, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        out = self.out(dec1)

        return out
