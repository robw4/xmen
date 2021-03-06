"""automatic messaging"""
from xmen import Experiment
from xmen.monitor import Monitor

# link some experiments
ex1, ex2 = Experiment(), Experiment()
ex1.register('/tmp', 'ex1')
ex2.register('/tmp', 'ex2')

# open_socket monitor
m = Monitor(msg='y.*->ex.*@10s')

y1, y2 = 0, 0
for i in m(range(40)):  # monitor loop
    y1 += 1
    y2 += 2
    if i % 10 == 1:
        # messages added automatically to ex1 and ex2
        print(', '.join(
            [f"ex1: {k} = {ex1.messages.get(k, None)}" for k in ('y1', 'y2')] +
            [f"ex2: {k} = {ex1.messages.get(k, None)}" for k in ('y1', 'y2')]))

# timing information is also logged
print('\nAll Messages')
for k, v in ex1.messages.items():
    print(k, v)




