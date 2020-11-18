"""Automatic check-pointing"""
from xmen.monitor import Monitor
from torch.nn import Conv2d
from torch.optim import Adam

model = Conv2d(2, 3, 3)
model2 = Conv2d(2, 3, 3)
optimiser = Adam(model.parameters())

m = Monitor(
    '/tmp/checkpoint',
    ckpt=['^model$@5s', 'opt@1e', '^model2$@20s'],
    ckpt_keep=[5, 1, None])
for _ in m(range(10)):
    for _ in m(range(20)):
        # Do something
        ...

