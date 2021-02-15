import time
from xmen.monitor import Monitor


def timeout(root,
            sleep: float = 1.   # time to sleep for
            ):
    monitor = Monitor(msg='seconds|^a$|^b$->root@1s')
    seconds, a, b = 0, 'hello', 'world'
    for i in monitor(range(100)):
        for j in monitor(range(10000)):
            print(i + 1)
            time.sleep(1.)
            seconds += sleep
