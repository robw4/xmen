"""expanding dictionary variables"""
from xmen import Experiment
from xmen.monitor import Monitor, XmenMessenger

ex = Experiment()
ex.register('/tmp', 'ex')
m = Monitor(
    hooks=[
        XmenMessenger('z->^ex$@10s', keep='min', leader='y', expand=True)])

# dictionary variable
z = {'x': 5, 'y': 10}
for i in m(range(100)):
    z['i'] = i
    z['x'] += 1
    z['y'] = z['x'] ** 2
    if i % 10 == 1:
        # messages will be logged from inside z when expand is true (else just z will be logged)
        print([ex.messages.get(k, None) for k in ('i', 'y', 'x')])
