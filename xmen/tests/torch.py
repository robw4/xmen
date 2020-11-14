"""A basic deep learning experiment"""
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as T
import torch
import torch.utils

from xmen.tests.utils import Generator, Discriminator, weights_init, GanLoss, to_onehot, TorchExperiment


class MNISTcDCGAN(TorchExperiment):
    """A basic gan experiment making use of a cDCGAN implementation"""
    from typing import Tuple
    # Parameters
    batch_size: int = 64  # @p the batch size per gpu
    hw: Tuple[int, int] = (32, 32)  # @p the height and width of the image
    ncpus: int = 8  # @p the number of threads to use for dataloading
    ngpus: int = 2  # @p the number of gpus to run the model on
    cx: int = 1  # @p number of input image channels
    cy: int = 1  # @p The number of channels for the conditioning information
    cf: int = 512  # @p the number of features after the first conv in the discriminator
    cz: int = 100  # @p the dimensionality of the noise vector
    epochs: int = 100  # @p no. of epochs to train for
    lr: float = 0.0002  # @p ltearning rate
    betas: Tuple[float, float] = (0.5, 0.999)  # @p beta parameters used in the Adam optimiser
    root: str = '/tmp/mnist'  # @p the root directory for the mnist dataset

    # Configure default monitoring
    checkpoint = '^nn_.*@1e'  # @p checkpoint all variables matching this regex
    log = '^loss_.*@20s'  # @p log variables to stdout
    sca = '^loss_.*@20s'  # @p log tensorboard scalars
    img = '^_x_|x$@1e'  # @p log tensorboard images
    message = '^self@20s'  # @p update message frequency
    time = ('@20s', '@1e')  # @p time the experiment

    def run(self):
        """Train the cDCGAN model on mnist"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Dataset
        transform = T.Compose(
            [T.Resize(self.hw), T.CenterCrop(self.hw),
             T.ToTensor(), T.Normalize([0.5], [0.5])])
        dataset = torch.utils.data.DataLoader(
            MNIST(self.root, download=True, transform=transform),
            batch_size=self.batch_size * self.ngpus,
            pin_memory=True,
            shuffle=True, num_workers=self.ncpus)
        # Networks
        nn_g = Generator(self.cx, self.cy, self.cz, self.cf, self.hw)
        nn_d = Discriminator(self.cx, self.cy, self.cf, self.hw)
        nn_g, nn_d = (nn.to(device).float().apply(weights_init) for nn in (nn_g, nn_d))
        # Optimiser
        opt_g = torch.optim.Adam(nn_g.parameters(), lr=self.lr, betas=self.betas)
        opt_d = torch.optim.Adam(nn_d.parameyers(), lr=self.lr, betas=self.betas)
        # Criterion
        cr_gan = GanLoss('lsgan')
        pz = torch.distributions.Normal(0., 1)

        # Update configuration options
        for _ in self.monitor(range(self.epochs)):
            for x, y in self.monitor(dataset):
                # Process
                x, y = x.to(device), y.to(device).float()
                y = to_onehot(torch.unsqueeze(y, 1), 10)
                z = pz.sample([self.batch_size, self.cz]).to(device)
                # forward
                _x_ = nn_g(z, y)
                # backward G
                opt_g.zero_grad()
                loss_g = cr_gan.gen(nn_d, (_x_, y))
                loss_g.backward()
                loss_g.step()
                # backward D
                opt_d.zero_grad()
                loss_d = cr_gan.disc(nn_d, (x, y), (_x_.detach(), y.detach()))
                loss_d.backward()
                opt_d.step()


if __name__ == '__main__':
    MNISTcDCGAN().main()
