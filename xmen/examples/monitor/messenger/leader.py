"""using the leader argument"""
from xmen import Experiment
from xmen.monitor import Monitor, XmenMessenger

ex = Experiment()
ex.register('/tmp', 'ex')
m = Monitor(
    hooks=[
        XmenMessenger('^y$|^i$->^ex$@10s', keep='min', leader='^y$')])

x = -50
for i in m(range(100)):
    x += 1
    y = x ** 2
    if i % 10 == 1:
        print([ex.messages.get(k, None) for k in ('i', 'y')])