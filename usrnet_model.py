import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)

    def forward(self, x):
        res = self.res(x)
        return x + res


# -------------------------------------------------------
# convTranspose + relu
# -------------------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, padding=0, bias=True, mode='2R'):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)


# -------------------------------------------------------
# strideconv + relu
# -------------------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, padding=0, bias=True, mode='2R'):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)


# Prior Module
class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=(16, 32, 64, 64)):
        super(ResUNet, self).__init__()
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, padding=1, bias=False)

        # downsample
        self.m_down1 = sequential(*[ResBlock(nc[0], nc[0], bias=False, mode='CRC') for _ in range(2)],
                                  downsample_strideconv(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = sequential(*[ResBlock(nc[1], nc[1], bias=False, mode='CRC') for _ in range(2)],
                                  downsample_strideconv(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = sequential(*[ResBlock(nc[2], nc[2], bias=False, mode='CRC') for _ in range(2)],
                                  downsample_strideconv(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = sequential(*[ResBlock(nc[3], nc[3], bias=False, mode='CRC') for _ in range(2)])

        # upsample
        self.m_up3 = sequential(upsample_convtranspose(nc[3], nc[2], bias=False, mode='2'),
                                *[ResBlock(nc[2], nc[2], bias=False, mode='CRC') for _ in range(2)])
        self.m_up2 = sequential(upsample_convtranspose(nc[2], nc[1], bias=False, mode='2'),
                                *[ResBlock(nc[1], nc[1], bias=False, mode='CRC') for _ in range(2)])
        self.m_up1 = sequential(upsample_convtranspose(nc[1], nc[0], bias=False, mode='2'),
                                *[ResBlock(nc[0], nc[0], bias=False, mode='CRC') for _ in range(2)])

        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, padding=1, bias=False)

    def forward(self, x):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        return x


def splits(m, sf):
    """
    Split m into sfxsf distinct blocks
    Args:
        m: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    """
    b = torch.stack(torch.chunk(m, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


# Data Module
# Eq.7 from paper https://arxiv.org/abs/2003.10428
class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, x, fk, fkc, f2k, fkfy, alpha, sf):
        d = fkfy + torch.fft.fftn(alpha * x, dim=(-2, -1))
        t1 = torch.mean(splits(fk.mul(d), sf), dim=-1, keepdim=False)
        t2 = torch.mean(splits(f2k, sf), dim=-1, keepdim=False)
        t3 = fkc * (t1.div(t2 + alpha)).repeat(1, 1, sf, sf)
        return torch.real(torch.fft.ifftn((d - t3) / alpha, dim=(-2, -1)))


def p2o(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    """
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf


def upsample(x, sf=3):
    """
    s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    Args:
        x: tensor image, NxCxWxH
        sf: scale factor
    """
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


# Hyper-parameter Module
class HyperNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=16, h_nc=64):
        super(HyperNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, h_nc, 1),
            nn.ReLU(),
            nn.Conv2d(h_nc, h_nc, 1),
            nn.ReLU(),
            nn.Conv2d(h_nc, out_nc, 1),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


class USRNet(nn.Module):
    def __init__(self, n_iter=8, in_nc=4, out_nc=3, h_nc=64, nc=(64, 128, 256, 512)):
        super(USRNet, self).__init__()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc)
        self.D = DataNet()
        self.h = HyperNet(in_nc=2, out_nc=n_iter*2, h_nc=h_nc)
        self.n = n_iter

    def forward(self, x, k, sf, sigma):
        # [B, 3, h, w]

        h, w = x.shape[-2:]
        Fk = p2o(k, (h * sf, w * sf))
        FkC = torch.conj(Fk)
        F2k = torch.pow(torch.abs(Fk), 2)
        STy = upsample(x, sf=sf)
        FkFy = FkC * torch.fft.fftn(STy, dim=(-2, -1))

        x = F.interpolate(x, scale_factor=sf, mode='nearest')

        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))

        for i in range(self.n):
            x = self.D(x, Fk, FkC, F2k, FkFy, ab[:, i:i+1, ...], sf)
            x = self.p(torch.cat((x, ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1))

        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 24, 24)
    k = torch.randn(1, 1, 25, 25)
    sf = 4
    sigma = torch.randn(1, 1, 1, 1)
    #
    # # net = HyperNet()
    # # print(net(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1)).shape)
    #
    # # net = ResUNet()
    # # print(net(x).shape) # Concat with noise
    #
    net = USRNet()
    total_params = sum(p.numel() for p in net.parameters())
    print(f'USRNet has a total of {total_params} parameters')
    print(net(x, k, sf, sigma).shape)

    a = sequential(*[ResBlock(16, 16, bias=False, mode='CRC') for _ in range(2)],
                   downsample_strideconv(16, 32, bias=False, mode='2'))

    # print(a)
