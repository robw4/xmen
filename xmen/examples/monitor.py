


# xmen messenger hook
from xmen import Experiment
from xmen.monitor import Monitor, XmenMessenger

# messenger = XmenMessenger('y.*->ex.*@10s')   # log all variables matching the loss to experiments matching ex
# m = Monitor(hooks=[messenger])
# y1, y2 = 0, 0
# ex1, ex2 = Experiment(), Experiment()
# ex1.register('/tmp', 'ex1')
# ex2.register('/tmp', 'ex2')
# for i in m(range(40)):
#     y1 += 1
#     y2 += 2
#     if i % 10 == 1:
#         print(', '.join(
#             [f"ex1: {k} = {ex1.messages.get(k, None)}" for k in ('y1', 'y2')] +
#             [f"ex2: {k} = {ex1.messages.get(k, None)}" for k in ('y1', 'y2')]))

# ex = Experiment()
# ex.register('/tmp', 'ex')
# m = Monitor(
#     hooks=[
#         XmenMessenger('^y$|^i$->^ex$@10s', keep='min', leader='^y$')])
#
# x = -50
# for i in m(range(100)):
#     x += 1
#     y = x ** 2
#     if i % 10 == 1:
#         print([ex.messages.get(k, None) for k in ('i', 'y')])



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
        print([ex.messages.get(k, None) for k in ('z_i', 'z_y', 'z_x')])
