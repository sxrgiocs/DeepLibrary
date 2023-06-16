import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, pad=1, activation='l'):
        super(SingleConv, self).__init__()
        self.activation = activation

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=k_size,
                                stride=stride,
                                padding=pad)

        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.batch_norm(x)

        if self.activation == 'l':
            x = F.leaky_relu(x)
        elif self.activation == 'g':
            x = F.gelu(x)
        elif self.activation == 'e':
            x = F.elu(x)
        else:
            x = F.relu(x)

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, pad=1):
        super(DoubleConv, self).__init__()

        self.conv1 = SingleConv(in_channels, out_channels, k_size, stride, pad)
        self.conv2 = SingleConv(out_channels, out_channels, k_size, stride, pad)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.pool(x)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2)

        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.shape[2] - x1.shape[2]
        diffY = x2.shape[3] - x1.shape[3]
        diffX = x2.shape[4] - x1.shape[4]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2))

        x = torch.cat([x2, x1], dim=1)

        return self.double_conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)

        # If binary segmentation apply sigmoid activation
        if x.size(1) == 2:
            x = torch.sigmoid(x)
        # If multi-class segmentation apply softmax activation
        else:
            x = F.softmax(x, dim=1)

        return x
