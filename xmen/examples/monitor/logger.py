"""Automatic Logging"""
from xmen.monitor import Monitor

m = Monitor(
    log=['x@2s', 'y@1e'],
    log_fn=[lambda _: '|'.join(_), lambda _: _],
    log_format=['', '.5f'])

x = ['cat', 'dog']
y = 0.
for _ in m(range(3)):
    for i in m(range(5)):
        y += i


