"""Networks for inheritance models"""
#  Copyright (C) 2019  Robert J Weston, Oxford Robotics Institute
#
#  xmen
#  email:   robw@robots.ox.ac.uk
#  github: https://github.com/robw4/xmen/
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#   along with this program. If not, see <http://www.gnu.org/licenses/>.

import torch
import torch.nn as nn


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not

    Taken from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Concat(nn.Module):
    def __init__(self, modules):
        super(Concat, self).__init__()
        self.layers = torch.nn.ModuleList(modules)

    def forward(self, args):
        out = []
        for m, x in zip(list(self.layers), args):
            out.append(m(x))
        return torch.cat(out, dim=1)


class ConvDecoder(nn.Module):
    def __init__(self, cy=6, cz=100, cx=1, cf=512, hw0=(4, 4), n_lvls=3):
        """Generator starts at level 2 (4 x 4)"""
        super(ConvDecoder, self).__init__()
        layers = [
            Concat([
                nn.ConvTranspose2d(cy, cf // 2, hw0, 1, bias=False),
                nn.ConvTranspose2d(cz, cf // 2, hw0, 1, bias=False)])]
        for l in range(0, n_lvls - 1):
            layers += [
                nn.BatchNorm2d(cf // 2 ** l),
                nn.ReLU(True),
                nn.ConvTranspose2d(cf // 2 ** l, cf // 2 ** (l + 1), 4, 2, 1, bias=False)]
        layers += [
            nn.BatchNorm2d(cf // 2 ** (n_lvls - 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(cf // 2 ** (n_lvls - 1), cx, 4, 2, 1),
            nn.Tanh()]
        self.nn = nn.Sequential(*layers)

    def forward(self, y, z):
        z, y = (t.reshape([t.shape[0], t.shape[1], 1, 1]) for t in (z, y))
        return self.nn((y, z))


class ConvEncoder(nn.Module):
    def __init__(self, cx, cy, cz, cf0=512, hw0=(1, 1), n=3):
        super(ConvEncoder, self).__init__()
        layers = [
            Concat([
                nn.Conv2d(cx, cf0 // 2 ** n, 4, 2, 1, bias=True),
                nn.Conv2d(cy, cf0 // 2 ** n, 4, 2, 1, bias=True)])]
        for i, l in enumerate(reversed(range(1, n))):
            if i > 1:
                layers += [nn.BatchNorm2d(cf0 // 2 ** l)]
            layers += [
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(cf0 // 2 ** l, cf0 // 2 ** (l - 1), 4, 2, 1, bias=False)]
        layers += [
            nn.BatchNorm2d(cf0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(cf0, cz, hw0, 1)]
        self.nn = nn.Sequential(*layers)

    def forward(self, x, y):
        ones = torch.ones(
            [x.shape[0], y.shape[1], x.shape[2], x.shape[3]], device=y.device)
        # Broadcast y across hw
        y = ones * y.reshape([y.shape[0], y.shape[1], 1, 1])
        return self.nn((x, y))


def to_onehot(y, c=10):
    Y = torch.zeros([c])
    Y[y.squeeze().long()] = 1.
    return Y


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# VAE implementation
class GeneratorNet(ConvDecoder):
    pass


class PosteriorNet(ConvEncoder):
    def __init__(self, cx, cy, cz, cf, hw0, nl):
        super(PosteriorNet, self).__init__(cx, cy, 2 * cz, cf, hw0, nl)
        self.softplus = torch.nn.Softplus()

    def forward(self, x, y):
        h = super(PosteriorNet, self).forward(x, y)
        mu, sigma = torch.chunk(h, 2, 1)
        return mu, self.softplus(sigma)


class PriorNet(nn.Module):
    def __init__(self, cy, cz, cf, nl=1):
        super(PriorNet, self).__init__()
        layers = [
            nn.Linear(cy, cf, bias=False),
            nn.BatchNorm1d(cf),
            nn.ReLU()]
        for _ in range(nl):
            layers += [
                nn.Linear(cf, cf, bias=False),
                nn.BatchNorm1d(cf),
                nn.ReLU()]
        self.nn = nn.Sequential(*layers)
        self.mean = nn.Linear(cf, cz)
        self.sd = nn.Sequential(
            nn.Linear(cf, cz),
            nn.Softplus())

    def forward(self, y):
        h = self.nn(y)
        return self.mean(h), self.sd(h)


class DiscriminatorNet(torch.nn.Module):
    def __init__(self, cx, cy, cf, hw0, nl, mode='vanilla'):
        super().__init__()
        # Discriminator has only one output
        self.nn = ConvEncoder(cx, cy, 1, cf, hw0, nl)
        self.criterion = torch.nn.MSELoss() if mode == 'lsgan' else torch.nn.BCEWithLogitsLoss()

    def forward(self, x, real=None):
        score = self.nn(*x)
        if real is not None:
            target = torch.ones_like(score) if real else torch.zeros_like(score)
            return self.criterion(score, target)
        else:
            return score
