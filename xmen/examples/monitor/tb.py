"""defining TensorboardLogger hooks"""
from xmen.monitor import Monitor, TensorboardLogger
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T


ds = DataLoader(MNIST(os.getenv("HOME") + '/data/mnist', download=True,
      transform=T.Compose(
          [T.Resize([64, 64]), T.CenterCrop([64, 64]),
           T.ToTensor(), T.Normalize([0.5], [0.5])])), 8)

m = Monitor(
    directory='/tmp/tensorboard_25',
    hooks=[
        TensorboardLogger('scalar', '^a|^x$|^z$|^X$@10s', prefix='sca_'),
        TensorboardLogger('image', '^mnist@10s', lambda x: x[:2], prefix='img_'),
        TensorboardLogger('histogram', 'Z@1s'),
        TensorboardLogger('figure', 'fig@10s'),
        TensorboardLogger('text', 'i@5s', lambda x: f'Hello a step {x}'),
        TensorboardLogger('video', '^mnist@10s', lambda x: (x.unsqueeze(0) - x.min()) / (x.max() - x.min()))
    ]
)

# dictionary variable
X = torch.zeros([10, 3, 64, 64])
X[:, :, 10:-10, 10:-10] = 1
x = 0.

# Dictionary variables will be expanded
a = [1, 2]
z = {'x': 5, 'y': 10}

for i, (mnist, _) in m(zip(range(31), ds)):
    fig = plt.figure(figsize=[10, 5])
    plt.plot(np.linspace(0, 1000), np.cos(np.linspace(0, 1000) * i))

    Z = torch.randn([10, 3, 64, 64]) * i / 100
    x = (i - 50) ** 2
    z['i'] = i
    z['x'] += 1
    z['y'] = z['x'] ** 2