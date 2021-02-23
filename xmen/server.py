"""xmen server helper functions"""
import socket
import queue
from xmen.utils import dic_to_yaml
import struct


def send(dic, conn):
    """Send dictionary to connected socket"""
    string = dic_to_yaml(dic)
    buffer = string.encode()
    try:
        conn.sendall(struct.pack('Q', len(buffer)) + buffer)
    except socket.error:
        pass


def receive(conn):
    """Receive yaml dictionary over connected socket"""
    import ruamel.yaml
    from xmen.utils import IncompatibleYmlException
    from xmen.utils import commented_to_py
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


def sender_client(host, port, q: queue.Queue):
    """Send messages from queue to host on port using SSL"""
    import ssl
    import socket
    context = ssl.create_default_context()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
        with context.wrap_socket(ss, server_hostname=host) as s:
            s.connect((host, port))
            while True:
                try:
                    dic = q.get()
                    send(dic, s)
                except q.empty():
                    pass
