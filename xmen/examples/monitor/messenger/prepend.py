"""using expand with prepend"""
from xmen import Experiment
from xmen.monitor import Monitor, XmenMessenger

ex = Experiment()
ex.register('/tmp', 'ex')
m = Monitor(
    hooks=[
        XmenMessenger('z->^ex$@10s', keep='min', leader='z_y', expand=True, prepend=True)])


z = {'x': 5, 'y': 10}
for i in m(range(100)):
    z['i'] = i
    z['x'] += 1
    z['y'] = z['x'] ** 2
    if i % 10 == 1:
        # messages are prepended by 'z_'
        print([ex.messages.get(k, None) for k in ('z_i', 'z_y', 'z_x')])