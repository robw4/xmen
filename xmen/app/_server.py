#! python3

import socket
import struct
import os
import mysql.connector
from mysql.connector import MySQLConnection, Error
from mysql.connector import Error
from configparser import ConfigParser
import socket
import json
from xmen.utils import get_meta, get_version
from xmen.experiment import get_timestamps


import ruamel.yaml
from ruamel.yaml import StringIO

from xmen.config import GlobalExperimentManager
from xmen.utils import commented_to_py, IncompatibleYmlException

from xmen.server import receive

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


class ServerTask(object):
    def __init__(
            self,
            host='',
            port=8000,
            db_host=None,
            db_port=None,
            interval=1.,
            n_clients=100,
            config='/home/robw/config.ini',
            certfile='/etc/letsencrypt/live/xmen.rob-otics.co.uk/fullchain.pem',
            keyfile='/etc/letsencrypt/live/xmen.rob-otics.co.uk/privkey.pem'
    ):
        self.host = host
        self.port = port
        self.db_host = db_host
        self.db_port = db_port
        self.interval = interval
        self.n_clients = n_clients
        self._config = config
        self.certfile = certfile
        self.keyfile = keyfile

    def open_socket(self):
        import ssl
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(
            certfile=self.certfile,
            keyfile=self.keyfile,
        )
        # get the hostname
        s = socket.socket()
        print('Binding on', (self.host, self.port))
        s.bind((self.host, self.port))  # bind host address and port together
        return s, context

    def __call__(self):
        import socket
        import time
        from xmen.server import decode_request, ChangePassword, AddUser, Failed, send
        server_socket, context = self.open_socket()
        # configure how many client the server can listen simultaneously
        server_socket.listen(self.n_clients)
        print('Beginning...')
        while True:
            conn = None
            try:
                last = time.time()
                while time.time() - last < self.interval:
                    conn, address = server_socket.accept()
                    conn = context.wrap_socket(conn, server_side=True)
                    msg = receive(conn)
                    request = decode_request(msg)
                    response = Failed('Request not recognised')
                    if isinstance(request, ChangePassword):
                        response = self.change_password(request.user, request.password, request.new_password)
                    elif isinstance(request, AddUser):
                        response = self.register_user(request.user, request.password)
                    send(response, conn)
            except socket.error as m:
                print('An error occured')
                print(m)
                break

            finally:
                if conn is not None:
                    conn.shutdown(socket.SHUT_RDWR)
                    conn.close()  # close the connection

    @property
    def config(self, section='mysql'):
        """Return the database configuration file as a dict"""
        if not isinstance(self._config, dict):
            parser = ConfigParser()
            parser.read(self._config)
            config = {}
            if parser.has_section(section):
                items = parser.items(section)
                for item in items:
                    config[item[0]] = item[1]
            else:
                raise Exception('{0} not found in the {1} file'.format(section, self._config))
            self._config = config
        return self._config

    def database(self):
        return MySQLConnection(**self.config)

    def validate_password(self, user, password):
        from xmen.server import PasswordNotValid, PasswordValid, UserDoesNotExist
        database = self.database()
        cursor = database.cursor()
        try:
            cursor.execute(f"SELECT * FROM users WHERE user = '{user}'")
            matches = cursor.fetchall()
            if matches:
                valid = self.is_valid(password, matches[0][3])
                response = PasswordValid(user) if valid else PasswordNotValid(user)
            else:
                response = UserDoesNotExist(user)
        finally:
            cursor.close()
            database.close()
        return response

    def change_password(self, user, old, new):
        from xmen.server import PasswordValid, PasswordNotValid, PasswordChanged, Failed, FailedException
        response = self.validate_password(user, old)
        database = self.database()
        cursor = database.cursor()
        try:
            if isinstance(response, PasswordValid):
                hashed, salt = self.hash_password(new)
                cursor.execute(f"UPDATE users SET password = {str(hashed)[1:]} WHERE user = '{user}'")
                database.commit()
                response = PasswordChanged(user)
            elif isinstance(response, PasswordNotValid):
                response = Failed(f'Password is not valid for {user}')
        except Exception as m:
            response = Failed(str(m))
            pass
        finally:
            cursor.close()
            database.close()
            return response

    def hash_password(self, password):
        import sys
        path = os.path.join(os.getenv('HOME'), '.xmen')
        if not path in sys.path:
            sys.path.append(path)
        from password import hash
        return hash(password)

    def is_valid(self, password, hashed):
        import sys
        path = os.path.join(os.getenv('HOME'), '.xmen')
        if not path in sys.path:
            sys.path.append(path)
        from password import check
        return check(password, hashed)

    def register_user(self, user, password):
        from xmen.server import send, PasswordNotValid, PasswordValid, UserDoesNotExist, UserCreated, Failed, FailedException
        database = self.database()
        cursor = database.cursor()
        response = None
        try:
            response = self.validate_password(user, password)
            if isinstance(response, (PasswordValid, PasswordNotValid)):
                pass
            elif isinstance(response, UserDoesNotExist):
                print('user does not exist')
                hashed, salt = self.hash_password(password)
                cursor.execute(
                    f"INSERT INTO users(user, password, salt) VALUES('{user}',{str(hashed)[1:]},{str(salt)[1:]})")
                database.commit()
                response = UserCreated(user)
        except Exception as m:
            cursor.close()
            database.close()
            response = Failed(str(m))
        finally:
            cursor.close()
            database.close()
            return response


def receiver_server(host='', port=8000, interval=1.):
    import os
    import socket
    import time
    import ssl
    from xmen.server import REGISTER_USER
    # context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(
         certfile='/etc/letsencrypt/live/xmen.rob-otics.co.uk/fullchain.pem',
         keyfile='/etc/letsencrypt/live/xmen.rob-otics.co.uk/privkey.pem'
    )

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
                msg = receive(conn)
                if msg['request'] == REGISTER_USER:
                    try:
                        exists = add_user(msg['user'], msg['password'])
                    except Exception:
                        pass
                # if msg['request'] == 'save':
                #     params = msg['params']
                #     updates[os.path.join(params['_root'], params['_name'])] = params

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
    server_task = ServerTask()
    server_task()
    # server_task.register_user(None, 'oli', 'sssword')


    # gem = GlobalExperimentManager()
    #
    # # with gem:
    # get_q = queue.Queue()
    # send_q = queue.Queue(maxsize=1)
    #
    # thread_receive = threading.Thread(target=receiver_server, args=(get_q, 1., gem.host, 8000))
    # thread_send = threading.Thread(target=sender_server, args=(send_q, gem.host, gem.port + 1, gem.experiments))
    # thread_receive.start()
    # thread_send.start()
    #
    # while True:
    #     try:
    #         updates, last = get_q.get(True)
    #         #gem.experiments.update(updates)
    #         send_q.put(gem.experiments)
    #
    #     except queue.Empty:
    #         pass