import socket
from xmen.experiment import HOST, PORT


def client_program():
    from xmen.app._server import send
    import ssl
    import time
    #host = socket.gethostname()  # as both code is running on same pc
    #port = 7000  # socket server port number

    #client_socket = socket.socket()  # instantiate
    #client_socket.connect((host, port))  # connect to the server
    context = ssl.create_default_context()
    print(context.verify_mode)
    dic = {
        '_root': 'root',
        '_name': 'client',
        'a': 'Hello',
        'b': 'World'}
    HOST = 'xmen.rob-otics.co.uk'

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
        with context.wrap_socket(ss, server_hostname=HOST) as s:
            s.connect((HOST, PORT))
            while True:
                print(f'Asking for connection from {HOST} {PORT}')
                dic['time'] = float(time.time())
                send(dic, s)
                print(f'Sent dic to {HOST} {PORT}')
                time.sleep(2.)


if __name__ == '__main__':
    client_program()
