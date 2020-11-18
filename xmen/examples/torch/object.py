"""A class implementation of dcgan"""
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
import torch
from torch.distributions import Normal
from xmen.experiment import Experiment


class Dcgan(Experiment):
    """Train a conditional GAN to predict MNIST digits.

    To viusalise the results run::

        tensorboard --logdir ...

    """
    import tempfile
    b: int = 128  # @p the batch size per gpu
    hw0: Tuple[int, int] = (4, 4)  # @p the height and width of the image
    nl: int = 4  # @p The number of levels in the discriminator.
    data_root: str = tempfile.gettempdir()  # @p the root data directory
    cx: int = 1  # @p the dimensionality of the image input
    cy: int = 10  # @p the dimensionality of the conditioning vector
    cf: int = 512  # @p the number of features after the first conv in the discriminator
    cz: int = 100  # @p the dimensionality of the noise vector
    ncpus: int = 8  # @p the number of threads to use for data loading
    ngpus: int = 1  # @p the number of gpus to run the model on
    epochs: int = 100  # @p no. of epochs to train for
    gan: str = 'lsgan'  # @p the gan type to use (one of ['vanilla', 'lsgan'])
    lr: float = 0.0002  # @p learning rate
    betas: Tuple[float, float] = (0.5, 0.999)  # @p The beta parameters for the
    # Monitoring parameters
    checkpoint: str = 'nn_.*@1e'  # @p
    log: str = 'loss_.*@20s'  # @p log scalars
    sca: str = 'loss_.*@20s'  # @p tensorboard scalars
    img: str = '_x_|x$@20s'  # @p tensorboard images
    time: str = ('@20s', '@1e')  # @p timing modulos
    nimg: int = 64  # @p The maximum number of images to display to tensorboard
    ns: int = 5  # @p The number of samples to generate at inference

    @property
    def hw(self): return [d * 2 ** self.nl for d in self.hw0]

    @property
    def device(self): return 'cuda' if torch.cuda.is_available() else 'cpu'

    def datasets(self):
        """Returns a dictionary of iterable get_datasets for modes 'train' (label image pairs)
        and 'inference' (just inputs spanning the prediction space)"""
        from torch.utils.data import DataLoader
        from torchvision.datasets.mnist import MNIST
        import torchvision.transforms as T

        def to_target(y):
            Y = torch.zeros([self.cy])
            Y[y] = 1.
            return Y.reshape([self.cy, 1, 1])
        # Generate test samples
        y = torch.stack([to_target(i) for i in range(self.cy)])
        y = y.unsqueeze(0).expand(
            [self.ns, self.cy, self.cy, 1, 1]).reshape(  # Expand across last dim
            [-1, self.cy, 1, 1])  # Batch
        z = Normal(0., 1.).sample([y.shape[0], self.cz, 1, 1])
        return {
            'train': DataLoader(
                MNIST(self.data_root, download=True,
                      transform=T.Compose(
                        [T.Resize(self.hw), T.CenterCrop(self.hw),
                         T.ToTensor(), T.Normalize([0.5], [0.5])]),
                      target_transform=to_target),
                batch_size=self.b * self.ngpus,
                shuffle=True, num_workers=self.ncpus),
            'inference': list(
                zip(y.unsqueeze(0), z.unsqueeze(0)))}  # Turn into batches

    def build(self):
        """Build generator, discriminator and optimisers."""
        from torch.optim import Adam
        from xmen.examples.models import GeneratorNet, DiscriminatorNet
        nn_g = GeneratorNet(self.cy, self.cz, self.cx, self.cf, self.hw0, self.nl)
        op_g = Adam(nn_g.parameters(), lr=self.lr, betas=self.betas)
        nn_d = DiscriminatorNet(self.cx, self.cy, self.cf, self.hw0, self.nl)
        op_d = Adam(nn_d.parameters(), lr=self.lr, betas=self.betas)
        return nn_g, nn_d, op_g, op_d

    def run(self):
        """Train the mnist dataset for a fixed number of epochs running an
        inference loop after every epoch."""
        from xmen.monitor import Monitor, TensorboardLogger
        from xmen.examples.models import weights_init, set_requires_grad
        from torch.distributions import Normal
        from torch.distributions.one_hot_categorical import OneHotCategorical
        # setup
        datasets = self.datasets()
        pz = Normal(torch.zeros([self.cz]), torch.ones([self.cz]))
        py = OneHotCategorical(probs=torch.ones([self.cy]) / self.cy)
        nn_g, nn_d, op_g, op_d = self.build()
        nn_g = nn_g.to(self.device).float().apply(weights_init)
        nn_d = nn_d.to(self.device).float().apply(weights_init)
        # training loop
        m = Monitor(
            self.directory, ckpt=self.checkpoint,
            log=self.log, sca=self.sca, img=self.img,
            time=('@20s', '@1e'),
            img_fn=lambda x: x[:min(self.nimg, x.shape[0])],
            hooks=[TensorboardLogger('image', '_xi_$@1e', nrow=10)])
        for _ in m(range(self.epochs)):
            for x, y in m(datasets['train']):
                # process input
                x, y = x.to(self.device), y.to(self.device).float()
                b = x.shape[0]
                # discriminator step
                set_requires_grad([nn_d], True)
                op_d.zero_grad()
                z = pz.sample([b]).reshape([b, self.cz, 1, 1]).to(self.device)
                _x_ = nn_g(y, z)
                loss_d = nn_d((x, y), True) + nn_d((_x_.detach(), y.detach()), False)
                loss_d.backward()
                op_d.step()
                # generator step
                op_g.zero_grad()
                y = py.sample([b]).reshape([b, self.cy, 1, 1]).to(self.device)
                z = pz.sample([b]).reshape([b, self.cz, 1, 1]).to(self.device)
                _x_ = nn_g(y, z)
                set_requires_grad([nn_d], False)
                loss_g = nn_d((_x_, y), True)
                loss_g.backward()
                op_g.step()
            # inference
            if 'inference' in datasets:
                with torch.no_grad():
                    for yi, zi in datasets['inference']:
                        yi, zi = yi.to(self.device), zi.to(self.device)
                        _xi_ = nn_g(yi, zi)