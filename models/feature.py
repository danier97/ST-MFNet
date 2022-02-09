import torch
import torch.nn as nn



class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(SEBlock, self).__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNextBlock(nn.Module):
    def __init__(self, down, cin, cout, ks, stride=1, groups=32, base_width=4, norm_layer=None):
        super(ResNextBlock, self).__init__()
        if norm_layer is None or norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'identity':
            norm_layer = Identity
        width = int(cout * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(cin, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        if down:
            self.conv2 = nn.Conv2d(width, width, kernel_size=ks, stride=stride, 
                                   padding=(ks-1)//2, groups=groups, bias=False)
        else:
            self.conv2 = nn.ConvTranspose2d(width, width, kernel_size=ks, stride=stride, 
                                            padding=(ks-stride)//2, groups=groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, cout, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != cout:
            if down:
                self.downsample = nn.Sequential(
                    nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
                    norm_layer(cout),
                )
            else:
                self.downsample = nn.Sequential(
                    # ks = stride here s.t. resolution can be kept 
                    nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=stride, bias=False),
                    norm_layer(cout),
                )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiScaleResNextBlock(nn.Module):
    def __init__(self, down, cin, cout, ks_s, ks_l, stride, norm_layer):
        super(MultiScaleResNextBlock, self).__init__()
        self.resnext_small = ResNextBlock(down, cin, cout//2, ks_s, stride, norm_layer=norm_layer)
        self.resnext_large = ResNextBlock(down, cin, cout//2, ks_l, stride, norm_layer=norm_layer)
        self.attention = SEBlock(cout)


    def forward(self, tensorCombine):
        out_small = self.resnext_small(tensorCombine)
        out_large = self.resnext_large(tensorCombine)
        out = torch.cat([out_small, out_large], 1)
        out = self.attention(out)
        return out


class UMultiScaleResNext(nn.Module):
    def __init__(self, channels=[64,128,256,512], norm_layer='batch', inplanes=6, **kwargs):
        super(UMultiScaleResNext, self).__init__()
        self.conv1 = MultiScaleResNextBlock(True, inplanes, channels[0], ks_s=3, ks_l=7, stride=2, norm_layer=norm_layer)
        self.conv2 = MultiScaleResNextBlock(True, channels[0], channels[1], ks_s=3, ks_l=7, stride=2, norm_layer=norm_layer)
        self.conv3 = MultiScaleResNextBlock(True, channels[1], channels[2], ks_s=3, ks_l=5, stride=2, norm_layer=norm_layer)
        self.conv4 = MultiScaleResNextBlock(True, channels[2], channels[3], ks_s=3, ks_l=5, stride=2, norm_layer=norm_layer)
        
        self.deconv4 = MultiScaleResNextBlock(True, channels[3], channels[3], ks_s=3, ks_l=5, stride=1, norm_layer=norm_layer)
        self.deconv3 = MultiScaleResNextBlock(False, channels[3], channels[2], ks_s=4, ks_l=6, stride=2, norm_layer=norm_layer)
        self.deconv2 = MultiScaleResNextBlock(False, channels[2], channels[1], ks_s=4, ks_l=8, stride=2, norm_layer=norm_layer)
        self.deconv1 = MultiScaleResNextBlock(False, channels[1], channels[0], ks_s=4, ks_l=8, stride=2, norm_layer=norm_layer)


    def forward(self, im0, im2):
        tensorJoin = torch.cat([im0, im2], 1) # (B,6,H,W)

        tensorConv1 = self.conv1(tensorJoin)
        tensorConv2 = self.conv2(tensorConv1)
        tensorConv3 = self.conv3(tensorConv2)
        tensorConv4 = self.conv4(tensorConv3)

        tensorDeconv4 = self.deconv4(tensorConv4)
        tensorDeconv3 = self.deconv3(tensorDeconv4 + tensorConv4)
        tensorDeconv2 = self.deconv2(tensorDeconv3 + tensorConv3)
        tensorDeconv1 = self.deconv1(tensorDeconv2 + tensorConv2)

        return tensorDeconv1