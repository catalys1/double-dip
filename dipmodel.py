import torch
import torch.nn.functional as F

# LeakyReLU activations
# Strided convolution downsampling
# Bilinear interpolation upsampling
# Reflection padding in convolutions

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, torch.nn.Conv2d): torch.nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, torch.nn.Linear): m.weight.data.normal_(0, 0.01)
    for x in m.children(): init_cnn(x)


def group_norm(channels):
    ng = channels // 16 if not channels % 16 else channels // 4
    return torch.nn.GroupNorm(max(1, ng), channels)


def batch_norm(channels):
    return torch.nn.BatchNorm2d(channels)
    

def conv(inc, outc, ks=3, stride=1, norm=group_norm):
    c = torch.nn.Conv2d(inc, outc, ks, padding=0, stride=stride, bias=False)
    bn = norm(outc)
    a = torch.nn.LeakyReLU(inplace=True)
    mods = [c, bn, a]
    if ks > 1:
        p = torch.nn.ReflectionPad2d([ks // 2] * 4)
        mods = [p] + mods
    return torch.nn.Sequential(*mods)


class DownBlock(torch.nn.Module):
    def __init__(self, inc, outc, ks=3, norm=group_norm):
        super().__init__()
        self.layer1 = conv(inc, outc, ks, stride=2, norm=norm)
        self.layer2 = conv(outc, outc, ks, norm=norm)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class UpBlock(torch.nn.Module):
    def __init__(self, inc, outc, ks=3, norm=group_norm):
        super().__init__()
        self.bn = norm(inc)
        self.layer1 = conv(inc, outc, ks)
        self.layer2 = conv(outc, outc, ks)

    def forward(self, x):
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


class DIP(torch.nn.Module):
    def __init__(self, inc, c_down, k_down, c_up, k_up, c_skip=None,
                 k_skip=None, norm='group', c_out=3):
        super().__init__()
        assert len(c_down) == len(k_down)
        assert len(c_up) == len(k_up)
        assert len(c_down) == len(c_up)
        if c_skip is None:
            assert k_skip is None
        if k_skip is None:
            assert c_skip is None

        nblock = len(c_down)
        skip = c_skip is not None
        c_skip = [0]*nblock if c_skip is None else c_skip
        k_skip = [0]*nblock if k_skip is None else k_skip

        assert norm in ('group', 'batch')
        norm = globals()[norm+'_norm']

        inc = [inc] + c_down[:-1]
        down = []
        up = []
        for channels, ksize, ic in zip(c_down, k_down, inc):
            block = DownBlock(ic, channels, ksize, norm=norm)
            down.append(block)
        inc = [c_down[-1]] + c_up[:-1]
        for channels, ksize, ic, sk in zip(c_up, k_up, inc, c_skip):
            block = UpBlock(ic + sk, channels, ksize, norm=norm)
            up.append(block)

        if skip:
            skips = []
            for channels, ksize, dw in zip(c_skip, k_skip, c_down):
                block = conv(dw, channels, ksize, norm=norm)
                skips.append(block)

        self.down = torch.nn.ModuleList(down)
        self.up = torch.nn.ModuleList(up)
        self.skip = torch.nn.ModuleList(skips) if skip else None
        self.out_conv = torch.nn.Conv2d(c_up[-1], c_out, 1, bias=True)

        init_cnn(self)

    def forward(self, x):
        skips = []
        for i in range(len(self.down)):
            x = self.down[i](x)
            if self.skip is not None:
                skips.append(self.skip[i](x))

        for i in range(len(self.up)):
            if self.skip is not None:
                x = torch.cat((x, skips[-(i+1)]), 1)
            x = self.up[i](x)

        x = self.out_conv(x)
        x = torch.sigmoid(x)

        return x


class DoubleDIP(torch.nn.Module):
    def __init__(self, dip1, dip2, dipm):
        super().__init__()
        self.dip1 = dip1
        self.dip2 = dip2
        self.dipm = dipm

    def forward(self, z1, z2, zm):
        y1 = self.dip1(z1)
        y2 = self.dip2(z2)
        ym = self.dipm(zm)

        return y1, y2, ym


def dip(norm='group', c_out=3):
    return DIP(32, [128]*5, [3]*5, [128]*5, [3]*5, [4]*5, [1]*5,
               norm=norm, c_out=c_out)

