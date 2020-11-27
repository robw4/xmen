"""using the leader argument"""
from xmen import Experiment
from xmen.monitor import TorchMonitor

ex = Experiment()
ex.register('/tmp', 'ex')
m = TorchMonitor(msg='^y$|^i$->^ex$@10s', msg_keep='min', msg_leader='^y$')

x = -50
for i in m(range(100)):
    x += 1
    y = x ** 2
    if i % 10 == 1:
        print([ex.messages.get(k, None) for k in ('i', 'y')])

