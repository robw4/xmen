from xmen.monitor import Monitor, Logger

m = Monitor(
    hooks=[
        Logger('x@2s', process_func=lambda x: '|'.join(x)),
        Logger('y@1e', format='.5f')])

x = ['cat', 'dog']
y = 0.
for _ in m(range(3)):
    for i in m(range(5)):
        y += i
