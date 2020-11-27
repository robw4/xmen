from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
from typing import List, Any

from xmen.lightning import TensorBoardLogger, Trainer


class LitMNIST(LightningModule):
    """Example torch lighnning module taken from the docs"""

    def __init__(self):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x_in):
        batch_size, channels, width, height = x_in.size()
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x_in.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log_dict(
            {'loss': loss, 'x': x})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_epoch_end(
        self, outputs: List[Any]
    ) -> None:
        self.log('loss_val', torch.stack(outputs).mean())

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def test_epoch_end(
        self, outputs: List[Any]
    ) -> None:
        self.log('loss_val', torch.stack(outputs).mean())


def lit_experiment(
        root,
        batch_size=64,    # The batch size of the experiment
        epochs=5,  # Number of epochs to train for
):
    """Xmen meets pytorch_lightning"""
    import xmen
    import os

    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    # prepare transforms standard to MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    # data
    mnist_train = MNIST('/tmp/xmen', train=True, download=True, transform=transform)
    mnist_train = DataLoader(mnist_train, batch_size=batch_size)
    mnist_val = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    mnist_val = DataLoader(mnist_val, batch_size=batch_size)

    model = LitMNIST()
    trainer = Trainer(
        default_root_dir=root.directory,
        max_epochs=epochs,
        logger=TensorBoardLogger(
            root=root,
            log=['loss@100s', 'loss_val'],
            sca=['loss@100s', 'loss_val'],
            img='x@100s',
            time='@500s',
            msg='loss@50s'
        )
    )
    trainer.fit(model, mnist_train, mnist_val)
    trainer.test(model, mnist_val)


if __name__ == '__main__':
    from xmen.functional import functional_experiment
    Exp, _ = functional_experiment(lit_experiment)
    Exp().main()

