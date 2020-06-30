import torch
import torch.nn as nn

from xmen.experiment import Experiment
Monitor = ...

class TorchExperiment(Experiment):
    from typing import Optional, Iterable, Union
    checkpoint: Optional[Union[Iterable[str], str]] = None  # @p specify checkpoints eg. 'nn.*@1e'
    keep: Optional[int] = None  # @p keep these most recent checkpoints
    log: Optional[Union[Iterable[str], str]] = None  # @p specify logging eg. 'loss.*@100s'
    sca: Optional[Union[Iterable[str], str]] = None  # @p specify tensorboard scalars eg. 'loss.*@1e'
    img: Optional[Union[Iterable[str], str]] = None  # @p specify tebnsorboard images eg. 'x.*@200s'
    time: Optional[Union[Iterable[str], str]] = ('@20s', '@1e')  # @p Time the experiment eg. '@20s'
    message: Optional[Union[Iterable[str], str]] = 'self@100s'  # @p Update experiment status in messages eg. '@20s'
    limit: Optional[str] = None  # @p Define maximum number of steps or epochs to run the experiment for

    def __init__(self, name=None, root=None, purpose='', copy=True, **kwargs):
        super(TorchExperiment, self).__init__(name=None, root=None, purpose='', copy=True, **kwargs)
        self._monitor = None
        self.hooks = []
        self.img_fn = None
        self.sca_fn = None

    def monitor(self, iter, **kwargs):
        """Monitor over the given iterator"""
        if not self._monitor:
            self.configure(**kwargs)
        return self._monitor(iter)

    def configure(self, **kwargs):
        """Update the experiment monitoring configuration.

        Args:
            **kwargs:
                checkpoint: checkpoint specified torch.Module objects
                log: log specified objects on triggers to stdout
                img: log as tensorboard images
                img_fn: Apply this function to each variable before logging as images
                sca: log as tensorboard scalars
                sca_fn: Apply this function to each variable before logging as scalars
                time: log timing statistics at these frequencies
                hooks: A list of user specified hooks conforming the the ``Hook`` api.
                message: summarise the current state of the experiment at these frequencies using
                    the xmen.Experiment api.
                limit: define maximum number of steps or epochs to run the experiment for
        """
        # User defined hooks (either) modulo or not modulo
        params = {k: v for k, v in self.__dict__.items() if k in {
            'checkpoint', 'checkpoints_to_keep', 'log', 'img',
            'sca', 'time', 'message', 'limit', 'img_fn', 'sca_fn', 'hooks'}}

        # Reconfigure the monitor
        self._monitor = Monitor(
            self.directory, **{**params, **kwargs})


class GanLoss(torch.nn.Module):
    """A helper class for calculating the GAN loss for discriminators and genenerators.

        In in its original form this is given as:

        G*, get_disc*  = min_G max_D   E_ x~p_r(x) [ log get_disc(x) ] + E_ z~p_z(z)[ log 1 - get_disc(G(x)) ]

        In step 1 the simulator is held fixed whilst we maximise with respect to the discriminator:
        get_disc*  =   max_D   log get_disc(x) +  log 1 - get_disc(G(x))
            =   min_D  -y log(chain_lengths(x)) - (1 - y ) log (1 - chain_lengths(G(z))      (where y=1 if real and 0 if fake)
            =   min_D  BCE(chain_lengths(x), 1) + BCE(chain_lengths(G(z)), 0)                (where BCE(p, y) gives either log(p) or log(1 - p)

        In step 2 the discriminator is held fixed whilst we maximise with respect to the simulator:
        G*  =   min_G   log get_disc(x) +  log 1 - get_disc(G(x))
            =   min_G   log 1 - get_disc(G(x))
            =   min_G   - BCE(get_disc(G(x), 0)

        Three modified modes are currently implemented:
            1. 'vanilla':
                Early on in training the distinguishing between real and generated examples is very easy, the get_disc
                saturates get_disc(G(x)) -> 0 and the gradient with respect to the parameters tend towards 0.
                Instead the authors propose an alternative optimisation step:
                    G*  =   max_G   log chain_lengths(G(x))
                        =   min_G   BCE(chain_lengths(G(x), 1)
            2. 'lsgan': Instead of using BCE criterion in LSGAN BCE is swapped for MSE criterion
            3. 'wsgangp': Instead of using BCE criterion in wasserstein GAN we consider:
                get_disc*  =   min_D  - get_disc(x).mean() + get_disc(G(x)).mean()  +  lambda * GP          (where .mean()  averages across batches)
                G*  =   min_G  - get_disc(G(x)).mean()
        """
    def __init__(self, mode, gp_weight=None):
        super(GanLoss, self).__init__()
        self.mode = mode
        self.gp_weight = gp_weight

    def disc(self, discriminator, real, fake):
        if not isinstance(real, (list, tuple)):
            real = [real]
        if not isinstance(fake, (list, tuple)):
            fake = [fake]
        loss = 0.5 * (self._calc_loss(discriminator(*real), True) + self._calc_loss(discriminator(*fake), False))
        if self.mode != 'wgangp':
            return loss
        else:
            gp = self.gradient_penalty(discriminator, real, fake)
            return loss + self.gp_weight * gp

    def gen(self, discriminator, fake):
        """The criterion function optimised for G with a fixed get_disc. Note the orginal paper initially proposes
        G*  = min_G  E_ x~p_r(x) [ log get_disc(x) ] + E_ z~p_z(z)[ log 1 - get_disc(G(x)) ]
            = min_G  E_ z~p_z(z)[ log 1 - get_disc(G(x)) ]

        However, early on during training the get_disc can very easily predict that the generated examples are fake
        meaning get_disc(G(x)) -> 0. In this case log(1-get_disc(G(z)) -> log(1) -> 0 which saturates, Instead they propose to
        optimise G* = max_G  E_ z~p_z(z)[ log get_disc(G(x)) ] That is maximise the probability of the get_disc judging
        the example as real.
        """
        # min_G max_D   E_ x~p_r(x) [ log get_disc(x) ] + E_ z~p_z(z)[ log 1 - get_disc(G(x)) ]
        if not isinstance(fake, (list, tuple)):
            fake = [fake]
        return self._calc_loss(discriminator(*fake), True)

    def _calc_loss(self, score, is_real):
        if self.mode in ['vanilla', 'lsgan']:
            target = torch.ones_like(score) if is_real else torch.zeros_like(score)
            criterion = torch.nn.MSELoss() if 'lsgan' else torch.nn.BCEWithLogitsLoss()
            return criterion(score, target)
        else:
            return -score.mean() if is_real else score.mean()


class Concat(nn.Module):
    def __init__(self, modules):
        super(Concat, self).__init__()
        self.layers = torch.nn.ModuleList(modules)

    def forward(self, args):
        out = []
        for m, x in zip(list(self.layers), args):
            out.append(m(x))
        return torch.cat(out, dim=1)


class Generator(nn.Module):
    def __init__(self, cx=1, cy=6, cz=100, cf=512, hw=(32, 32)):
        """Generator starts at level 2 (4 x 4)"""
        super(Generator, self).__init__()

        self.nn = nn.Sequential(
            # [b, cy_in, 1, 1] and [b, cz_in, 1, 1]
            Concat([
                nn.ConvTranspose2d(cz, cf // 2, (hw[0] // 2 ** 3, hw[1] // 2 ** 3), 1, bias=False),
                nn.ConvTranspose2d(cy, cf // 2, (hw[0] // 2 ** 3, hw[1] // 2 ** 3), 1, bias=False)]),
            # [b, cf, 4, 4]
            nn.BatchNorm2d(cf // 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(cf // 1, cf // 2, 4, 2, 1, bias=False),
            # [b, cf // 2, 8, 8]
            nn.BatchNorm2d(cf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(cf // 2, cf // 4, 4, 2, 1, bias=False),
            # [b, cf // 4, 16, 16]
            nn.BatchNorm2d(cf // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(cf // 4, cx, 4, 2, 1, bias=False),
            nn.Tanh())
        # [b, 1, 32, 32]

    def forward(self, z, y):
        z, y = (t.reshape([t.shape[0], t.shape[1], 1, 1]) for t in (z, y))
        return self.nn((z, y))


class Discriminator(nn.Module):
    def __init__(self, cx, cy, cf=512, hw=(32, 32)):
        super(Discriminator, self).__init__()
        self.nn = nn.Sequential(
            # [b, cx_in, 32, 32] and [b, cy_in, 32, 32]
            Concat([
                nn.Conv2d(cx, cf // 8, 4, 2, 1),
                nn.Conv2d(cy, cf // 8, 4, 2, 1),
            ]),
            # [b, cf // 4, 16, 16]
            nn.BatchNorm2d(cf // 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(cf // 4, cf // 2, 4, 2, 1),
            # [b, cf // 2, 8, 8]
            nn.BatchNorm2d(cf // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(cf // 2, cf // 1, 4, 2, 1),
            # [b, cf // 1, 4, 4]
            nn.BatchNorm2d(cf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(cf // 1, 1, (hw[0] // 2 ** 3, hw[1] // 2 ** 3), 1))

    def __call__(self, x, y):
        ones = torch.ones(
            [x.shape[0], y.shape[1], x.shape[2], x.shape[3]], device=y.device)
        # Broadcast y across hw
        y = ones * y.reshape([y.shape[0], y.shape[1], 1, 1])
        return self.nn((x, y))


def to_onehot(y, c=10):
    b = y.shape[0]
    Y = torch.zeros([b, c], device=y.device)
    Y[torch.arange(b), y.squeeze().long()] = 1.
    return Y


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)