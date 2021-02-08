from xmen import Experiment
from xmen.monitor import Monitor
import os

X = Experiment(os.path.join(os.environ['HOME'], 'tmp'), 'ex1')

a, b, c, d, e, f = 0, 0, 0, 0, 0, 0


def identity(x):
    return x


def mult(x):
    return 10 * x


m = Monitor(
    log=('a|b@2s', 'b@1e', 'c@1er', "d|e@1eo", "e@1se"),
    log_fn=(mult, identity, identity, identity, identity),
    log_format='.3f',
    msg='a->X@1s',
    time=('@2s', '@1e'),
    probe='X@10s',
    limit='@20s')
for _ in m(range(2)):
    for _ in m(range(2)):
        for _ in m(range(3)):
            for _ in m(range(4)):
                for _ in m(range(5)):
                    a += 1
                b += 1
            c += 1
        d += 1
    e += 1
