"""xmen server api"""
import socket
import queue
from xmen.utils import dic_to_yaml

import struct
import ssl

from typing import NamedTuple, Optional, Dict, List


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
LINK_EXPERIMENT = 'link_experiment'
DELETE_EXPERIMENT = 'delete_experiment'
GET_EXPERIMENTS = 'get_experiments'


class Request(object):
    pass


class LinkExperiment(NamedTuple, Request):
    user: str
    password: str
    root: str
    data: str
    status: str
    request: str = LINK_EXPERIMENT

    def __repr__(self): return f'{self.__class__.__name__}<user={self.user}, root={self.root}, status={self.status}>'


class UpdateExperiment(NamedTuple, Request):
    user: str
    password: str
    root: str
    status: str
    data: str
    request: str = UPDATE_EXPERIMENT

    def __repr__(self): return f'{self.__class__.__name__}<user={self.user}, root={self.root}, status={self.status}>'


class GetExperiments(NamedTuple, Request):
    user: str
    password: str
    roots: str
    status: str
    updated_since: str = '1960-01-01 00:00:00'
    max_n: int = None
    request: str = GET_EXPERIMENTS

    def __repr__(self): return f'{self.__class__.__name__}<user={self.user}, root={self.roots}, status={self.status}, ' \
                               f'updated_since={self.updated_since}>'


class ValidatePassword(NamedTuple, Request):
    user: str
    password: str
    request: str = VALIDATE_PASSWORD

    def __repr__(self): return f'{self.__class__.__name__}<user={self.user}>'


class ChangePassword(NamedTuple, Request):
    user: str
    password: str
    new_password: str
    request: str = CHANGE_PASSWORD

    def __repr__(self): return f'{self.__class__.__name__}<user={self.user}>'


class AddUser(NamedTuple, Request):
    user: str
    password: Optional[str]
    request: str = ADD_USER

    def __repr__(self): return f'{self.__class__.__name__}<user={self.user}>'


class DeleteExperiment(NamedTuple, Request):
    user: str
    password: str
    root: str
    request: str = DELETE_EXPERIMENT

    def __repr__(self): return f'{self.__class__.__name__}<user={self.user}, root={self.root}>'


REQUESTS = {
    ADD_USER: AddUser,
    VALIDATE_PASSWORD: ValidatePassword,
    CHANGE_PASSWORD: ChangePassword,
    LINK_EXPERIMENT: LinkExperiment,
    UPDATE_EXPERIMENT: UpdateExperiment,
    DELETE_EXPERIMENT: DeleteExperiment,
    GET_EXPERIMENTS: GetExperiments,
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
EXPERIMENT_LINKED = 'experiment_registered'
EXPERIMENT_UPDATED = 'experiment_updated'
EXPERIMENT_DELETED = 'experiment_deleted'
GOT_EXPERIMENTS = 'got_experiments'


def decode_response(dic):
    return {PASSWORD_VALID: PasswordValid,
            PASSWORD_NOT_VALID: PasswordNotValid,
            USER_DOES_NOT_EXIST: UserDoesNotExist,
            USER_CREATED: UserCreated,
            PASSWORD_CHANGED: PasswordChanged,
            FAILED: Failed,
            EXPERIMENT_LINKED: ExperimentRegistered,
            EXPERIMENT_UPDATED: ExperimentUpdated,
            EXPERIMENT_DELETED: ExperimentDeleted,
            GOT_EXPERIMENTS: GotExperiments
    }[dic['response']](**dic)


class ExperimentDeleted(NamedTuple):
    user: str
    root: str
    response: str = EXPERIMENT_DELETED

    @property
    def msg(self): return f'Experiment {self.root} deleted for {self.user}'


class ExperimentUpdated(NamedTuple):
    user: str
    root: str
    response: str = EXPERIMENT_LINKED

    @property
    def msg(self): return f'Experiment {self.root} registered for {self.user}'


class ExperimentRegistered(NamedTuple):
    user: str
    root: str
    response: str = EXPERIMENT_LINKED

    @property
    def msg(self): return f'Experiment {self.root} registered for {self.user}'


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
    def msg(self): return f'User account does not exist for {self.user}'


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


class GotExperiments(NamedTuple):
    user: str
    matches: List
    roots: str
    status: str
    time: str = None
    response: str = GOT_EXPERIMENTS

    @property
    def msg(self): return f'Found {len(self.matches)} experiments ' \
                          f'matching root {self.roots} with status {self.status}'


# -------------------------------------
# Helper functions
# -------------------------------------
def send(dic, conn, typ='safe'):
    """Send dictionary to connected socket"""
    from xmen.utils import commented_to_py
    if hasattr(dic, '_asdict'):
        dic = dic._asdict()
    # data is always sent without comments
    dic = {k: commented_to_py(v) for k, v in dic.items()}
    string = dic_to_yaml(dic, typ, True)

    buffer = string.encode()
    conn.sendall(struct.pack('Q', len(buffer)) + buffer)


def receive_message(conn, length):
    buffer = b''
    while len(buffer) != length:
        _ = conn.recv(length)
        if _ == '':
            break
        buffer += _
    assert len(buffer) == length
    return buffer


def receive(conn, typ='safe', default_flow_style=False):
    """Receive dictionary over connected socket"""
    import ruamel.yaml
    from xmen.utils import IncompatibleYmlException
    from xmen.utils import commented_to_py
    length = receive_message(conn, struct.calcsize('Q'))
    dic = None
    if length:
        length, = struct.unpack('Q', length)
        buffer = receive_message(conn, length)
        yaml = ruamel.yaml.YAML(typ=typ)
        yaml.default_flow_style = default_flow_style
        try:
            dic = yaml.load(buffer)
        except Exception as m:
            raise IncompatibleYmlException
        dic = {k: commented_to_py(v) for k, v in dic.items()}
    return dic


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


def add_gpu_info(request: UpdateExperiment):
    from xmen.utils import dic_from_yml, dic_to_yaml, get_meta
    if isinstance(request, UpdateExperiment):
        meta = get_meta(get_gpu=True)
        dic = dic_from_yml(string=request.data)
        if meta.get('gpu', None):
            dic['_meta']['gpu'] = meta['gpu']
            data = dic_to_yaml(dic)
            request = UpdateExperiment(
                request.user, request.password,
                request.root, request.status, data)
        return request


def send_request_task(q_request, q_response=None, hook=None):
    from xmen.config import Config
    import time
    config = Config()
    context = ssl.create_default_context()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
        with context.wrap_socket(ss, server_hostname=config.server_host) as s:
            while True:
                try:
                    s.connect((config.server_host, config.server_port))
                except (socket.error, OSError):
                    time.sleep(1.)
                else:
                    while True:
                        try:
                            request = q_request.get()
                            if not request:
                                return
                            if hook:
                                request = hook(request)
                            send(request, s)
                            response = receive(s)
                            if q_response is not None:
                                q_response.put(response)
                        except queue.Empty:
                            pass
                        except KeyboardInterrupt:
                            pass
