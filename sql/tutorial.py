import mysql.connector
from mysql.connector import MySQLConnection, Error
from mysql.connector import Error
from configparser import ConfigParser
import socket
import json
from xmen.utils import get_meta, get_version
from xmen.experiment import get_timestamps

HOST = socket.gethostname()

params = {
    '_root': '/some/path/to/experiment',
    '_user': 'robw',
    '_timestamps': get_timestamps(),
    '_meta': get_meta(),
    '_version': get_version(path=__file__),
}


def add_experiment(params):
    query = f"INSERT INTO experiments(root, host, user, data) " \
             "VALUES(%s,%s,%s,%s)"
    args = ('/some/path/to/experiment', HOST, 'robw', json.dumps(params))
    try:
        db_config = read_db_config()
        conn = MySQLConnection(**db_config)

        cursor = conn.cursor()
        cursor.execute(query, args)

        if cursor.lastrowid:
            print('last insert id', cursor.lastrowid)
        else:
            print('last insert id not found')
        conn.commit()
    except Error as error:
        print(error)

    finally:
        cursor.close()
        conn.close()


def read_db_config(filename='/home/robw/config.ini', section='mysql'):
    """Return the database configuration file as a dict"""
    parser = ConfigParser()
    parser.read(filename)

    config = {}
    if parser.has_section(section):
        items = parser.items(section)
        for item in items:
            config[item[0]] = item[1]
    else:
        raise Exception('{0} not found in the {1} file'.format(section, filename))
    return config


def connect():
    """ Connect to MySQL database """
    conn = None
    config = read_db_config()
    try:
        conn = mysql.connector.connect(**config)
        if conn.is_connected():
            print('Connected to MySQL database')

    except Error as e:
        print(e)

    finally:
        if conn is not None and conn.is_connected():
            conn.close()


if __name__ == '__main__':
    add_experiment(params)
