from xmen.app._server import receiver_client
from xmen.config import GlobalExperimentManager


if __name__ == '__main__':
    import threading
    import queue

    gem = GlobalExperimentManager()
    with gem:
        thread_receive = threading.Thread(target=receiver_client, args=(gem.host, gem.port + 1))
        thread_receive.start()
        thread_receive.join()
        #
        # while True:
        #     try:
        #         experiments = get_q.get(True)
        #         print(experiments.keys())
        #
        #     except queue.Empty:
        #         pass