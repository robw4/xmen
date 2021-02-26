"""xmen server helper functions"""
import socket
import queue
from xmen.utils import dic_to_yaml

import struct
from typing import NamedTuple, Optional, Dict


class FailedException(Exception):
    def __init__(self, msg):
        self.msg = msg


# -------------------------------------
# Requests
# -------------------------------------
ADD_USER = 'register_user'
VALIDATE_PASSWORD = 'validate_password_request'
CHANGE_PASSWORD = 'change_password'
UPDATE_EXPERIMENT = 'update_experiment'
REGISTER_EXPERIMENT = 'register_experiment'


class Request(object):
    pass


class RegisterExperiment(NamedTuple, Request):
    user: str
    password: str
    root: str
    data: str
    request: str = REGISTER_EXPERIMENT


class UpdateExperiment(NamedTuple, Request):
    user: str
    password: str
    root: str
    status: str
    data: str
    request: str = UPDATE_EXPERIMENT


class ValidatePassword(NamedTuple, Request):
    user: str
    password: str
    request: str = VALIDATE_PASSWORD


class ChangePassword(NamedTuple, Request):
    user: str
    password: str
    new_password: str
    request: str = CHANGE_PASSWORD


class AddUser(NamedTuple, Request):
    user: str
    password: Optional[str]
    request: str = ADD_USER


REQUESTS = {
    ADD_USER: AddUser,
    VALIDATE_PASSWORD: ValidatePassword,
    CHANGE_PASSWORD: ChangePassword,
    REGISTER_EXPERIMENT: RegisterExperiment,
    UPDATE_EXPERIMENT: UpdateExperiment,
}


def decode_request(dic: Dict): return REQUESTS[dic['request']](**dic)


def generate_request(dic: Dict, typ: Request): return REQUESTS[dic[typ]](**dic)


# -------------------------------------
# Responses
# -------------------------------------
PASSWORD_VALID = 'password_valid'
PASSWORD_NOT_VALID = 'password_not_valid'
USER_DOES_NOT_EXIST = 'user_does_not_exist'
USER_CREATED = 'user_created'
FAILED = 'failed'
PASSWORD_CHANGED = 'password_changed'
EXPERIMENT_REGISTERED = 'experiment_registered'
EXPERIMENT_UPDATED = 'experiment_updated'


def decode_response(dic):
    return {PASSWORD_VALID: PasswordValid,
            PASSWORD_NOT_VALID: PasswordNotValid,
            USER_DOES_NOT_EXIST: UserDoesNotExist,
            USER_CREATED: UserCreated,
            PASSWORD_CHANGED: PasswordChanged,
            FAILED: Failed,
            EXPERIMENT_REGISTERED: ExperimentRegistered,
            EXPERIMENT_UPDATED: ExperimentUpdated,
    }[dic['response']](**dic)


class ExperimentUpdated(NamedTuple):
    user: str
    root: str
    response: str = EXPERIMENT_REGISTERED

    @property
    def msg(self): return f'Experiment {self.root} registered {self.user}'


class ExperimentRegistered(NamedTuple):
    user: str
    root: str
    response: str = EXPERIMENT_REGISTERED

    @property
    def msg(self): return f'Experiment {self.root} registered {self.user}'


class Failed(NamedTuple):
    msg: str = ''
    response: str = FAILED

    def __repr__(self):
        return f'FAILED: {self.msg}'


class UserCreated(NamedTuple):
    user: str
    response: str = USER_CREATED

    @property
    def msg(self): return f'User account created for {self.user}'


class PasswordChanged(NamedTuple):
    user: str
    response: str = PASSWORD_CHANGED

    @property
    def msg(self): return f'Password changed for {self.user}'


class UserDoesNotExist(NamedTuple):
    user: str
    response: str = USER_DOES_NOT_EXIST

    @property
    def msg(self): return f'User account exists for {self.user} and passwords match'


class PasswordValid(NamedTuple):
    user: str
    response: str = PASSWORD_VALID

    @property
    def msg(self): return f'User account exists for {self.user} and passwords match'


class PasswordNotValid(NamedTuple):
    user: str
    response: str = PASSWORD_NOT_VALID

    @property
    def msg(self): return f'password incorrect for {self.user}'

# -------------------------------------
# Helper functions
# -------------------------------------


def send(dic, conn):
    """Send dictionary to connected socket"""
    if hasattr(dic, '_asdict'):
        dic = dic._asdict()
    string = dic_to_yaml(dic)
    buffer = string.encode()
    conn.sendall(struct.pack('Q', len(buffer)) + buffer)


def receive(conn, timeout=None):
    """Receive yaml dictionary over connected socket"""
    import ruamel.yaml
    from xmen.utils import IncompatibleYmlException
    from xmen.utils import commented_to_py
    # conn.settimeout()
    length = conn.recv(struct.calcsize('Q'))
    dic = None
    if length:
        length, = struct.unpack('Q', length)
        buffer = conn.recv(length).decode()
        yaml = ruamel.yaml.YAML()
        try:
            dic = yaml.load(buffer)
        except:
            raise IncompatibleYmlException
        dic = {k: commented_to_py(v) for k, v in dic.items()}
    return dic

import ssl


def setup_socket():
    context = ssl.create_default_context()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return sock, context


def get_context(): return ssl.create_default_context()


def get_socket(): return socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def send_request(request):
    from xmen.config import Config
    config = Config()
    context = ssl.create_default_context()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
        with context.wrap_socket(ss, server_hostname=config.server_host) as s:
            s.connect((config.server_host, config.server_port))
            send(request, s)
            response = receive(s)
    return decode_response(response)


def send_request_task(q):
    from xmen.config import Config
    print('In send reqest task')
    config = Config()
    context = ssl.create_default_context()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
        with context.wrap_socket(ss, server_hostname=config.server_host) as s:
            s.connect((config.server_host, config.server_port))
            print('Connected')
            while True:
                try:
                    print(f'Getting request')
                    request = q.get()
                    if not request:
                        break
                    print(f'In got request')
                    send(request, s)
                    print(f'Sent request')
                    response = receive(s)
                    print('Response is ', response)
                except queue.Empty:
                    print('Empty queue')
                    pass
