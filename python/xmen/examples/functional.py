"""Functional experiment example"""
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#   along with this program. If not, see <http://www.gnu.org/licenses/>.
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
from typing import Tuple
import xmen
import os

try:
    import torch
except ImportError:
    print('In order to run this script first add pytorch to the python path')


def get_datasets(cy, cz, b, ngpus, ncpus, ns, data_root, hw, **kwargs):
    """Returns a dictionary of iterable get_datasets for modes 'train' (label image pairs)
    and 'inference' (just inputs spanning the prediction space)"""
    from torch.utils.data import DataLoader
    from torchvision.datasets.mnist import MNIST
    from torch.distributions import Normal
    import torchvision.transforms as T

    def to_target(y):
        Y = torch.zeros([cy])
        Y[y] = 1.
        return Y.reshape([cy, 1, 1])
    # Generate test samples
    y = torch.stack([to_target(i) for i in range(cy)])
    y = y.unsqueeze(0).expand(
        [ns, cy, cy, 1, 1]).reshape(  # Expand across last dim
        [-1, cy, 1, 1])  # Batch
    z = Normal(0., 1.).sample([y.shape[0], cz, 1, 1])
    return {
        'train': DataLoader(
            MNIST(data_root, download=True,
                  transform=T.Compose(
                    [T.Resize(hw), T.CenterCrop(hw),
                     T.ToTensor(), T.Normalize([0.5], [0.5])]),
                  target_transform=to_target),
            batch_size=b * ngpus,
            shuffle=True, num_workers=ncpus),
        'inference': list(
            zip(y.unsqueeze(0), z.unsqueeze(0)))}  # Turn into batches


def dcgan(
    x=xmen.X,   # experiment instance
    *,
    b: int = 128,  # the batch size per gpu
    hw0: Tuple[int, int] = (4, 4),  # the height and width of the image
    nl: int = 4,  # the number of levels in the discriminator.
    data_root: str = os.getenv("HOME") + '/data/mnist',  # @p the root data directory
    cx: int = 1,
    cy: int = 10,  # the dimensionality of the conditioning vector
    cf: int = 512,  # the number of features after the first conv in the discriminator
    cz: int = 100,  # the dimensionality of the noise vector
    ncpus: int = 8,  # the number of threads to use for data loading
    ngpus: int = 1,  # the number of gpus to run the model on
    epochs: int = 20,  # no. of epochs to train for
    lr: float = 0.0002,  # learning rate
    betas: Tuple[float, float] = (0.5, 0.999),  # the beta parameters for the
    # monitoring parameters
    checkpoint: str = 'nn_.*@1e',  # checkpoint at this modulo string
    log: str = 'loss_.*@20s',  # log scalars
    sca: str = 'loss_.*@20s',  # tensorboard scalars
    img: str = '_x_|x$@20s',  # tensorboard images
    nimg: int = 64,  # the maximum number of images to display to tensorboard
    ns: int = 5  # the number of samples to generate at inference)
):
    from xmen.monitor import Monitor, TensorboardLogger
    from xmen.examples.models import weights_init, set_requires_grad, GeneratorNet, DiscriminatorNet
    from torch.distributions import Normal
    from torch.distributions.one_hot_categorical import OneHotCategorical
    from torch.optim import Adam
    import logging

    hw = [d * 2 ** nl for d in hw0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = logging.getLogger()
    logger.setLevel('INFO')
    # dataset
    datasets = get_datasets(
        cy, cz, b, ngpus, ncpus, ns, data_root, hw)
    # models
    nn_g = GeneratorNet(cy, cz, cx, cf, hw0, nl)
    nn_d = DiscriminatorNet(cx, cy, cf, hw0, nl)
    nn_g = nn_g.to(device).float().apply(weights_init)
    nn_d = nn_d.to(device).float().apply(weights_init)
    # distributions
    pz = Normal(torch.zeros([cz]), torch.ones([cz]))
    py = OneHotCategorical(probs=torch.ones([cy]) / cy)
    # optimisers
    op_d = Adam(nn_d.parameters(), lr=lr, betas=betas)
    op_g = Adam(nn_g.parameters(), lr=lr, betas=betas)

    # monitor
    m = Monitor(
        x.directory, checkpoint=checkpoint,
        log=log, sca=sca, img=img,
        time=('@20s', '@1e'),
        img_fn=lambda x: x[:min(nimg, x.shape[0])],
        hooks=[TensorboardLogger('image', '_xi_$@1e', nrow=10)])

    for _ in m(range(epochs)):
        # (1) train
        for x, y in m(datasets['train']):
            # process input
            x, y = x.to(device), y.to(device).float()
            b = x.shape[0]
            # discriminator step
            set_requires_grad([nn_d], True)
            op_d.zero_grad()
            z = pz.sample([b]).reshape([b, cz, 1, 1]).to(device)
            _x_ = nn_g(y, z)
            loss_d = nn_d((x, y), True) + nn_d((_x_.detach(), y.detach()), False)
            loss_d.backward()
            op_d.step()
            # generator step
            op_g.zero_grad()
            y = py.sample([b]).reshape([b, cy, 1, 1]).to(device)
            z = pz.sample([b]).reshape([b, cz, 1, 1]).to(device)
            _x_ = nn_g(y, z)
            set_requires_grad([nn_d], False)
            loss_g = nn_d((_x_, y), True)
            loss_g.backward()
            op_g.step()
        # (2) inference
        if 'inference' in datasets:
            with torch.no_grad():
                for yi, zi in datasets['inference']:
                    yi, zi = yi.to(device), zi.to(device)
                    _xi_ = nn_g(yi, zi)


def printer(a, b, c, d, e, f, g, h, i, j):
    """Print out the inputs"""
    print(a, b, c, d, e, f, g, h, i, j)
