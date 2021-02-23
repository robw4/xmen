#! python3

import socket
import struct
import os

import ruamel.yaml
from ruamel.yaml import StringIO

from xmen.config import GlobalExperimentManager
from xmen.experiment import IncompatibleYmlException
from xmen.utils import commented_to_py

CONFIG = GlobalExperimentManager()
HOST = CONFIG.host
PORT = CONFIG.port



def sender_server(q, host, port, experiments):
    import socket, ssl

    #context = ssl.create_default_context()
    # context.load_cert_chain(
    #     certfile=os.path.join('/home/kebl4674/.ssl', 'cert.pem'),
    #     keyfile=os.path.join('/home/kebl4674/.ssl', 'key.pem'))
    server_socket = socket.socket()
    server_socket.bind(('', port))
    server_socket.listen(10)

    while True:
        try:
            experiments = q.get(False)
        except queue.Empty:
            pass

        conn, address = server_socket.accept()
        # conn = context.wrap_socket(conn, server_side=True, server_hostname='')
        send(experiments, conn)
        print(f'Sent experiments {list(updates.keys())} from {address}')
        conn.shutdown(socket.SHUT_RDWR)
        conn.close()  # close the connection


def receiver_client(host, port):
    import ssl
    # context = ssl.create_default_context()

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # with context.wrap_socket(ss, server_hostname=host) as s:
            print(f'{host} {port}')
            s.connect((host, port))
            dic = receive(s)
            print(f'Received {list(dic.keys())} in client')


def receiver_server(q, interval, host, port):
    import os
    import socket
    import time
    import ssl
    # context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(
         certfile='/etc/letsencrypt/live/xmen.rob-otics.co.uk/fullchain.pem',
         keyfile='/etc/letsencrypt/live/xmen.rob-otics.co.uk/privkey.pem'
    )
    #context.check_hostname = False

    # get the hostname
    server_socket = socket.socket()
    print('Binding on', ('', port))
    server_socket.bind(('', port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(10)
    while True:
        print('Beginning...')
        try:
            updates = {}
            last = time.time()
            while time.time() - last < interval:
                conn, address = server_socket.accept()
                conn = context.wrap_socket(conn, server_side=True)
                # try:
                #
                # except ssl.SSLError as m:
                #     if 'tlsv1 alert unknown ca' in m.reason:
                #         print(str(m))
                #         pass
                params = receive(conn)
                updates[os.path.join(params['_root'], params['_name'])] = params
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()  # close the connection
            if updates:
                print(f'Received updates from {list(updates.keys())} from {address}')
                q.put((updates, last))

        except socket.error as m:
            print('An error occured')
            print(m)
            break
        #     # if 'tlsv1 alert unknown ca':
        #     #     pass
        #     # q.put((None, m))
        #     print('An error ', m)
        #     with open('/data/engs-robot-learning/kebl4674/usup/tmp/xmen-error-log.txt', 'w') as f:
        #         f.write(str(m))
        #     break


if __name__ == '__main__':
    import threading
    import queue

    gem = GlobalExperimentManager()

    with gem:
        get_q = queue.Queue()
        send_q = queue.Queue(maxsize=1)

        thread_receive = threading.Thread(target=receiver_server, args=(get_q, 1., gem.host, 8000))
        thread_send = threading.Thread(target=sender_server, args=(send_q, gem.host, gem.port + 1, gem.experiments))
        thread_receive.start()
        thread_send.start()

        while True:
            try:
                updates, last = get_q.get(True)
                #gem.experiments.update(updates)
                send_q.put(gem.experiments)

            except queue.Empty:
                pass