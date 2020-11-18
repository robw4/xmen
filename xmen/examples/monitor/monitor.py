from xmen.monitor import Monitor
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T

plt.style.use('ggplot')
ds = DataLoader(MNIST(os.getenv("HOME") + '/data/mnist', download=True,
      transform=T.Compose(
          [T.Resize([64, 64]), T.CenterCrop([64, 64]),
           T.ToTensor(), T.Normalize([0.5], [0.5])])), 8)

m = Monitor(
    directory='/tmp/tb_5',
    sca=['^z$|^X$@10s', '^a|^x$@5s'],
    img=['^mnist@10s', '^mnist@5s'], img_fn=[lambda x: x[:2], lambda x: x[:5]], img_pref=['2', '5'],
    hist='Z@1s',
    fig='fig@10s',
    txt='i@5s', txt_fn=lambda x: f'Hello at step {x}',
    vid='^mnist@10s', vid_fn=lambda x: (x.unsqueeze(0) - x.min()) / (x.max() - x.min())
)

# variables
x = 0.
a = [1, 2]
z = {'x': 5, 'y': 10}
for i, (mnist, _) in m(zip(range(31), ds)):
    # plot a figure
    fig = plt.figure(figsize=[10, 5])
    plt.plot(np.linspace(0, 1000), np.cos(np.linspace(0, 1000) * i))
    # random tensor
    Z = torch.randn([10, 3, 64, 64]) * i / 100
    # scalars
    x = (i - 15) ** 2
    z['i'] = i
    z['x'] += 1
    z['y'] = z['x'] ** 2