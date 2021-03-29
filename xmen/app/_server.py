import os
import argparse
from configparser import ConfigParser

from mysql.connector import MySQLConnection

from xmen.app._xgent import DESCRIPTION
from xmen.server import *

parser = argparse.ArgumentParser(prog='xmen-server', description=DESCRIPTION)
parser.add_argument('--host', '-H', default='', help='The host to run the xmen server on')
parser.add_argument('--port', '-P', default=8000, help='The port to run the xmen server on', type=int)
parser.add_argument('--certfile', '-C', default='/etc/letsencrypt/live/xmen.rob-otics.co.uk/fullchain.pem',
                    help='The path to the ssl certificate')
parser.add_argument('--dbconfig', '-D', default='/home/robw/config.ini')
parser.add_argument('--keyfile', '-K', default='/etc/letsencrypt/live/xmen.rob-otics.co.uk/privkey.pem')
parser.add_argument('--n_clients', '-N', default=100, help='The maximum number of client connections')


def server(args):
    from multiprocessing import Process as Thread
    print('Server using multiprocessing')
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(
        certfile=args.certfile,
        keyfile=args.keyfile)
    s = socket.socket()
    s.bind((args.host, args.port))
    s.listen(args.n_clients)
    processes = []
    try:
        while True:
            conn, address = s.accept()
            p = Thread(
                target=ServerTask(
                    args.host, args.port, args.dbconfig,
                    args.certfile, args.keyfile, args.n_clients),
                args=(conn, address))
            p.start()
            processes += [p]
    finally:
        for p in processes:
            p.kill()
            p.join()
        s.shutdown(socket.SHUT_RDWR)
        s.close()  # close the connection


class ServerTask(object):
    def __init__(
            self,
            host,
            port,
            config,
            certfile,
            keyfile,
            n_clients=100,
    ):
        self.host = host
        self.port = port
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

    def __call__(self, conn, addr):
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(
                certfile=self.certfile,
                keyfile=self.keyfile)
            conn = context.wrap_socket(conn, server_side=True)
            while True:
                # receive
                msg = receive(conn)
                if not msg:
                    # connection has been closed
                    break
                request = decode_request(msg)
                response = Failed('Request not recognised')

                print(request)
                if isinstance(request, ChangePassword):
                    response = self.change_password(request.user, request.password, request.new_password)
                elif isinstance(request, AddUser):
                    response = self.register_user(request.user, request.password)
                elif isinstance(request, LinkExperiment):
                    response = self.link_experiment(
                        request.user, request.password, request.root, request.data, request.status)
                elif isinstance(request, UpdateExperiment):
                    response = self.update_experiment(
                        request.user, request.password, request.root, request.data, request.status)
                elif isinstance(request, DeleteExperiment):
                    response = self.delete_experiment(
                        request.user, request.password, request.root)
                elif isinstance(request, GetExperiments):
                    response = self.get_experiments(request.user, request.password, request.roots,
                                                    request.status, request.updated_since, request.max_n)
                # manage response
                if isinstance(response, Failed):
                    print(response.msg)

                send(response, conn)
        except Exception as m:
            print(f'An error occured:')
            print(m)
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

    def validate_password(self, user, password, hard=True):
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

            failed = (PasswordNotValid, UserDoesNotExist) if hard else PasswordNotValid
            if isinstance(response, failed):
                response = Failed(response.msg)
                print(response.msg)
        finally:
            cursor.close()
            database.close()
        return response

    def change_password(self, user, password, new):
        response = self.validate_password(user, password)
        if isinstance(response, Failed):
            return response
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
            if len(user) < 3:
                response = Failed(msg='username must be at least 3 characters long')
            elif len(password) < 6:
                response = Failed(msg='password must be at least 7 characters long')
            else:
                response = self.validate_password(user, password, hard=False)

            if isinstance(response, UserDoesNotExist):
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
        # validate password
        response = self.validate_password(user, password)
        if isinstance(response, Failed):
            return response
        database = self.database()
        cursor = database.cursor()
        response = None
        try:
            # assume experiments previously at the same root have since been deleted]
            cursor.execute(
                f"SELECT root, data, updated, status FROM experiments WHERE root REGEXP '{root}' AND user = '{user}' ")
            matches = cursor.fetchall()
            if not matches:
                response = self.link_experiment(user, password, root, data, status)
            else:
                cursor.execute(
                    f"UPDATE experiments "
                    f"SET status = %s, data = %s, updated = CURRENT_TIMESTAMP() "
                    f"WHERE root = %s AND status != '{DELETED}' AND user = %s",
                    (status, data, root, user)
                )
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

    def delete_experiment(self, user, password, root):
        from xmen.experiment import DELETED
        response = self.validate_password(user, password)
        if isinstance(response, Failed):
            return response
        database = self.database()
        cursor = database.cursor()
        response = None
        try:
            cursor.execute(
                f"UPDATE experiments "
                f"SET status = '{DELETED}', updated = CURRENT_TIMESTAMP() "
                f"WHERE root = '{root}' AND user = '{user}' ")
            database.commit()
            response = ExperimentDeleted(user, root)
        except Exception as m:
            cursor.close()
            database.close()
            response = Failed(str(m))
        finally:
            cursor.close()
            database.close()
            return response

    def update_data(self, experiments, **updates):
        """Update data field in each experiment"""
        from xmen.utils import dic_from_yml, dic_to_yaml
        out = []
        for e in experiments:
            dic = dic_from_yml(string=e)
            dic.update(updates)
            out.append(dic_to_yaml(e))
        return out

    def get_experiments(self, user, password, roots, status, updated, max_n):
        import time
        response = self.validate_password(user, password)
        if isinstance(response, Failed):
            return response
        database = self.database()
        cursor = database.cursor()
        response = None
        try:
            request_time = time.strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                f"SELECT * FROM experiments WHERE root REGEXP '{roots}' "
                f"AND status REGEXP '{status}' AND user = '{user}' AND updated > '{updated}' "
                f"ORDER BY updated")
            matches = cursor.fetchall()
            if max_n is not None:
                matches = matches[-max_n:]
            response = GotExperiments(user, matches, roots, status, request_time)
        except Exception as m:
            cursor.close()
            database.close()
            response = Failed(str(m))
        finally:
            cursor.close()
            database.close()
            return response

    def link_experiment(self, user, password, root, data, status):
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
                # assume experiments previously at the same root have since been deleted
                cursor.execute(
                    f"UPDATE experiments SET status = '{DELETED}'"
                    f"WHERE root = '{root}'")
                cursor.execute(
                    f"INSERT INTO experiments(root, status, user, data) VALUES(%s, %s,%s,%s)",
                    (root, status, user, data)
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
    args = parser.parse_args()
    server(args)
