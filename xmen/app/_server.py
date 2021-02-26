import os
import time
from mysql.connector import MySQLConnection
from configparser import ConfigParser
from xmen.server import *

HOST = ''
PORT = 8000
CONFIG = '/home/robw/config.ini'
CERTFILE = '/etc/letsencrypt/live/xmen.rob-otics.co.uk/fullchain.pem'
KEYFILE = '/etc/letsencrypt/live/xmen.rob-otics.co.uk/privkey.pem'


def server(args):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(
        certfile=args.certfile,
        keyfile=args.keyfile,
    )
    # get the hostname
    s = socket.socket()
    s.bind((args.host, args.port))  # bind host address and port together



class Server(object):
    def __init__(
            self,
            host=HOST,
            port=PORT,
            db_host=None,
            db_port=None,
            interval=1.,
            n_clients=100,
            config=CONFIG,
            certfile=CERTFILE,
            keyfile=KEYFILE
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
        """Open a socket and an ssl context"""
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
                        print('Got Change Password Request')
                        response = self.change_password(request.user, request.password, request.new_password)
                    elif isinstance(request, AddUser):
                        print('Got Add User Request')
                        response = self.register_user(request.user, request.password)
                    elif isinstance(request, RegisterExperiment):
                        print('Got Register Experiment request')
                        response = self.register_experiment(
                            request.user, request.password, request.root, request.data)
                    elif isinstance(request, UpdateExperiment):
                        print('Got Update Experiment Request')
                        response = self.update_experiment(
                            request.user, request.password, request.root, request.data, request.status)
                        print(response)
                    send(response, conn)
            except Exception as m:
                print(f'An error occured {m}')
                pass
            finally:
                if conn is not None:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                        conn.close()  # close the connection
                    except (socket.error, OSError) as m:
                        print(m)

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

    def update_experiment(self, user, password, root, data, status):
        from xmen.experiment import DELETED
        database = self.database()
        cursor = database.cursor()
        response = None
        try:
            response = self.validate_password(user, password)
            if isinstance(response, PasswordNotValid):
                return Failed(f'{response.msg}')
            elif isinstance(response, Failed):
                return response
            else:
                # assume experiments previously at the same root have since been deleted]
                cursor.execute(
                    f"UPDATE experiments SET status = '{status}', data = '{data}' WHERE root = '{root}' AND status != '{DELETED}'")
                database.commit()
                response = ExperimentUpdated(user, root)
        except Exception as m:
            cursor.close()
            database.close()
            response = Failed(str(m))
        finally:
            cursor.close()
            database.close()
            return response

    def register_experiment(self, user, password, root, data):
        from xmen.experiment import REGISTERED, DELETED
        database = self.database()
        cursor = database.cursor()
        response = None
        try:
            response = self.validate_password(user, password)
            if isinstance(response, PasswordNotValid):
                return Failed(f'{response.msg}')
            elif isinstance(response, Failed):
                return response
            else:
                # assume experiments previously at the same root have since been deleted
                cursor.execute(
                    f"UPDATE experiments SET status = '{DELETED}' WHERE root = '{root}'")
                cursor.execute(
                    f"INSERT INTO experiments(root, status, user, data) VALUES('{root}','{REGISTERED}','{user}','{data}')"
                )
                database.commit()
                response = ExperimentRegistered(user, root)
        except Exception as m:
            cursor.close()
            database.close()
            response = Failed(str(m))
        finally:
            cursor.close()
            database.close()
            return response


if __name__ == '__main__':
    server_task = Server()
    server_task()
