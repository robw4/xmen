"""A multi"""
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
import os

# ------------------------------------------------------
# ---- BASE EXPERIMENTS --------------------------------
# ------------------------------------------------------
class BaseGenerative(Experiment):
    """An abstract class defining parameters and some useful properties common to
     both the VAE and cGAN implementations.

    Inherited classes must overload:
    - get_datasets()
    - build()
    - run()

    Inherited classes can optionally overload:
    - parameter defaults and documentation

    Note:
        The output size is given as hw0 * 2 ** nl (eg (4, 2) * 2 ** 4 = (64, 32)
    """
    b: int = 256  # @p the batch size per gpu
    hw0: Tuple[int, int] = (4, 4)  # @p the height and width of the image
    nl: int = 4  # @p The number of levels in the discriminator.
    data_root: str = os.getenv("HOME") + '/data/mnist'  # @p the root data directory
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

    test_samples = None

    # Useful properties
    @property
    def hw(self): return [d * 2 ** self.nl for d in self.hw0]

    @property
    def device(self): return 'cuda' if torch.cuda.is_available() else 'cpu'

    def datasets(self): return NotImplementedError

    def run(self): return NotImplementedError

    def build(self): return NotImplementedError


class BaseCVae(BaseGenerative):
    """BaseCVae is an intermediary class defining the model and training loop but not the dataset.

    Inherited classes should overload:
    - dataset()
    """
    nlprior: int = 1  # @p The number of hidden layers in the prior network
    w_kl: float = 1.0  # @p The weighting on the KL divergence term
    ns: int = 5  # @p The number of samples to generate at inference
    predictive: str = 'prior'  # @p Use either the 'prior' or 'posterior' predictive at inference
    b = 32
    log = 'loss.*|log_px|kl_qz_pz@20s'
    sca = 'loss.*|log_px|kl_qz_pz@20s'
    img = '_x_|x$@20s'
    time = ('@10s', '@1e')

    def build(self):
        from torch.optim import Adam
        from itertools import chain
        from xmen.examples.torch.models import GeneratorNet, PosteriorNet, PriorNet
        nn_gen = GeneratorNet(self.cy, self.cz, self.cx, self.cf, self.hw0, self.nl)
        nn_post = PosteriorNet(self.cx, self.cy, self.cz, self.cf, self.hw0, self.nl)
        nn_prior = PriorNet(self.cy, self.cz, self.cf, self.nlprior)
        opt = Adam(chain(nn_gen.parameters(), nn_post.parameters(), nn_prior.parameters()),
                   self.lr, self.betas)
        return nn_gen, nn_post, nn_prior, opt

    def run(self):
        from torch.distributions import Normal, kl_divergence
        from xmen.monitor import Monitor, TensorboardLogger
        from xmen.examples.torch.models import weights_init
        # construct model
        datasets = self.datasets()
        nn_gen, nn_post, nn_prior, opt = self.build()
        nn_gen, nn_post, nn_prior = (
            v.to(self.device).float().apply(weights_init) for v in (
            nn_gen, nn_post, nn_prior))
        if self.ngpus > 1:
            nn_gen, nn_post, nn_prior = (torch.nn.DataParallel(n) for n in (
                nn_gen, nn_post, nn_prior))
        m = Monitor(
            self.directory, checkpoint=self.checkpoint,
            log=self.log, sca=self.sca, img=self.img,
            img_fn=lambda x: x[:min(self.nimg, x.shape[0])],
            time=('@20s', '@1e'),
            hooks=[TensorboardLogger('image', '_xi_$@20s', nrow=self.ns)])
        for _ in m(range(self.epochs)):
            # Training
            for x, y in m(datasets['train']):
                x, y = x.to(self.device), y.to(self.device).float()
                qz = Normal(*nn_post(x, y))
                z = qz.sample()
                # Likelihood
                _x_ = nn_gen(y, z)
                px = Normal(_x_, 0.1)
                pz = Normal(*nn_prior(y.reshape([-1, self.cy])))
                log_px = px.log_prob(x).sum()
                kl_qz_pz = kl_divergence(qz, pz).sum()
                loss = self.w_kl * kl_qz_pz - log_px
                opt.zero_grad()
                loss.backward()
                opt.step()
            # Inference
            if 'inference' in datasets:
                with torch.no_grad():
                    for yi, zi in datasets['inference']:
                        if self.predictive == 'posterior':
                            qz = Normal(*nn_prior(yi.reshape([-1, self.cy])))
                            zi = qz.sample()
                        _xi_ = nn_gen(yi, zi)


class BaseGAN(BaseGenerative):
    """BaseGan is an intermediary class defining the model and training loop but not the dataset.

    Inherited classes should overload:
    - dataset()
    - distributions()
    """
    # overload previous parameter default (documentation is maintained)
    b = 128
    # define new parameters
    ns: int = 5  # @p The number of samples to generate at inference

    def build(self):
        """Build the model from the current configuration"""
        from torch.optim import Adam
        from xmen.examples.torch.models import GeneratorNet, DiscriminatorNet
        nn_g = GeneratorNet(self.cy, self.cz, self.cx, self.cf, self.hw0, self.nl)
        op_g = Adam(nn_g.parameters(), lr=self.lr, betas=self.betas)
        nn_d = DiscriminatorNet(self.cx, self.cy, self.cf, self.hw0, self.nl)
        op_d = Adam(nn_d.parameters(), lr=self.lr, betas=self.betas)
        return nn_g, nn_d, op_g, op_d

    def distributions(self): raise NotImplementedError

    def run(self):
        from xmen.monitor import Monitor, TensorboardLogger
        from xmen.examples.torch.models import set_requires_grad, weights_init
        # Get get_datasets
        datasets = self.datasets()
        py, pz = self.distributions()
        nn_g, nn_d, op_g, op_d = self.build()
        nn_g = nn_g.to(self.device).float().apply(weights_init)
        nn_d = nn_d.to(self.device).float().apply(weights_init)
        m = Monitor(
            self.directory, checkpoint=self.checkpoint,
            log=self.log, sca=self.sca, img=self.img,
            time=('@20s', '@1e'),
            img_fn=lambda x: x[:min(self.nimg, x.shape[0])],
            hooks=[TensorboardLogger('image', '_xi_$@1e', nrow=self.ns)])
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
            # Inference
            if 'inference' in datasets:
                with torch.no_grad():
                    for yi, zi in datasets['inference']:
                        yi, zi = yi.to(self.device), zi.to(self.device)
                        _xi_ = nn_g(yi, zi)


class BaseMnist(BaseGenerative):
    """cDCGAN Training on the MNIST get_datasets"""
    # Update defaults
    hw0, nl = (4, 4), 3  # Default size = 32 x 32
    data_root = os.getenv("HOME") + '/data/mnist'
    cx, cy, cz, cf = 1, 10, 100, 512
    ns = 10  # Number of samples to generate during inference

    def datasets(self):
        """Configure the get_datasets used for training"""
        from torch.utils.data import DataLoader
        from torchvision.datasets.mnist import MNIST
        import torchvision.transforms as T

        def to_target(y):
            Y = torch.zeros([self.cy])
            Y[y] = 1.
            return Y.reshape([self.cy, 1, 1])

        transform = T.Compose(
            [T.Resize(self.hw), T.CenterCrop(self.hw),
             T.ToTensor(), T.Normalize([0.5], [0.5])])
        y = torch.stack([to_target(i) for i in range(self.cy)])
        y = y.unsqueeze(0).expand(
            [self.ns, self.cy, self.cy, 1, 1]).reshape(  # Expand across last dim
            [-1, self.cy, 1, 1])  # Batch
        z = Normal(0., 1.).sample([y.shape[0], self.cz, 1, 1])
        return {
            'train': DataLoader(
                MNIST(self.data_root, download=True,
                      transform=transform,
                      target_transform=to_target),
                batch_size=self.b * self.ngpus,
                shuffle=True, num_workers=self.ncpus),
            'inference': list(
                zip(y.unsqueeze(0), z.unsqueeze(0)))}  # Turn into batches


# ------------------------------------------------------
# ---- RUNABLE EXPERIMENTS -----------------------------
# ------------------------------------------------------
class InheritedMnistGAN(BaseMnist, BaseGAN):
    """Train a cDCGAN on MNIST"""
    epochs = 20
    ncpus, ngpus = 0, 1
    b = 128

    def distributions(self):
        """Generate one hot and normal samples"""
        from torch.distributions import Normal
        from torch.distributions.one_hot_categorical import OneHotCategorical
        pz = Normal(torch.zeros([self.cz]), torch.ones([self.cz]))
        py = OneHotCategorical(probs=torch.ones([self.cy]) / self.cy)
        return py, pz


class InheritedMnistVae(BaseMnist, BaseCVae):
    """Train a conditional VAE on MNIST"""
    ns = 10
    w_beta = 1.5
    lr = 0.00005
    ncpus, ngpus = 0, 2

