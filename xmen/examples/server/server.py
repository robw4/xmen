import socket
import argparse

import threading

parser = argparse.ArgumentParser()
parser.add_argument('ports', nargs='*')


def client_task(conn):
    while True:
        # receive data stream. it won't accept data packet greater than 1024 bytes
        import struct
        # print(struct.calcsize('Q'))
        length = conn.recv(struct.calcsize('Q'))
        if length:
            length, = struct.unpack('Q', length)
            print('length = ', length)
            data = conn.recv(length).decode()
            print(data)
    conn.close()  # close the connection


def multi_client_server():
    # get the hostname
    from xmen.experiment import HOST, PORT
    # host = socket.gethostname()
    # port = 6011  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((HOST, PORT))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(10)

    threads = []
    while True:
        conn, address = server_socket.accept()
        print("Connection from: " + str(address))
        threading.Thread()
        t = threading.Thread(target=client_task, args=(conn, address))
        t.start()
        threads.append(t)


def multi_client_server_2():
    from xmen.utils import commented_to_py
    # get the hostname
    import struct
    import ruamel.yaml
    from xmen.experiment import IncompatibleYmlException
    host = socket.gethostname()
    port = 6011  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(10)
    while True:
        conn, address = server_socket.accept()
        print("Connection from: " + str(address))
        length = conn.recv(struct.calcsize('Q'))
        if length:
            length, = struct.unpack('Q', length)
            print('length = ', length)
            params = conn.recv(length).decode()

            yaml = ruamel.yaml.YAML()
            try:
                params = yaml.load(params)
            except:
                raise IncompatibleYmlException
            params = {k: commented_to_py(v) for k, v in params.items()}
            print(params)
        conn.close()  # close the connection


def updates_client(q, interval):
    from xmen.utils import commented_to_py
    import os
    import socket
    import time
    # get the hostname
    import struct
    import ruamel.yaml
    from xmen.experiment import IncompatibleYmlException
    host = '193.62.124.26'
    port = 6011  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(10)

    while True:
        updates = {}
        last = time.time()
        while time.time() - last < interval:
            conn, address = server_socket.accept()
            # print("Connection from: " + str(address))
            length = conn.recv(struct.calcsize('Q'))
            if length:
                length, = struct.unpack('Q', length)
                # print('length = ', length)
                params = conn.recv(length).decode()
                yaml = ruamel.yaml.YAML()
                try:
                    params = yaml.load(params)
                except:
                    raise IncompatibleYmlException
                params = {k: commented_to_py(v) for k, v in params.items()}
                print(params['_name'])
                # print(params)
                updates[os.path.join(params['_root'], params['_name'])] = params
            conn.close()  # close the connection
        if updates:
            q.put((updates, last))


def server_program():
    # get the hostname
    host = socket.gethostname()
    port = 7000  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(10)

    conn, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))
    while True:
        # receive data stream. it won't accept data packet greater than 1024 bytes
        data = conn.recv(1024).decode()
        if not data:
            # if data is not received break
            break
        print("from connected user: " + str(data))
        data = input(' -> ')
        conn.send(data.encode())  # send data to the client

    conn.close()  # close the connection

class Args:
    interval = 1.


if __name__ == '__main__':
    import threading
    import queue
    from xmen.app._xgent import updates_client


    q = queue.Queue()
    t = threading.Thread(target=updates_client, args=(q, Args()))
    t.start()
    while True:
        try:
            updates, last = q.get(False)
            print(updates.keys())
        except queue.Empty:
            pass

