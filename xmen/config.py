#  Copyright (C) 2019  Robert J Weston, Oxford Robotics Institute
#
#  xmen
#  email:   robw@robots.ox.ac.uk
#  github: https://github.com/robw4/xmen/
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#   along with this program. If not, see <http://www.gnu.org/licenses/>.
"""Helper classes and functions for managing the global configuration and setting / interfacing with the xmen server."""

import os
import re

import xmen.manager
from xmen.utils import get_meta
from xmen.server import *
import getpass

from functools import wraps
from typing import Tuple


def connected(method):
    @wraps(method)
    def _fn(self, *args, **kwargs):
        if self.user is None:
            print('WARNING: no account is currently registered with the xmen server. '
                  'To configure run xmen config --register_user')
            return
        else:
            with self.socket as ss:
                with self.context.wrap_socket(ss, server_hostname=self.server_host) as s:
                    try:
                        s.connect((self.server_host, self.server_port))
                    except (OSError, socket.error):
                        print(f'ERROR: cannot connect to server {self.server_host}, {self.server_port}')
                        return
            # user registered and server is online
            return method(self, *args, **kwargs)
    return _fn


def _send(request):
    config = Config()
    sock = get_socket()
    context = get_context()
    with sock as ss:
        with context.wrap_socket(ss, server_hostname=config.server_host) as s:
            s.connect((config.server_host, config.server_port))
            send(request, s)
            msg = receive(s)
            return decode_response(msg)


class Config(object):
    """Manage global settings and server configuration alongside interfacing with the server."""
    def __init__(self):
        """Instantiate the global configuration. If an ``xmen.yml`` config file exists in
        ``os.path.join(os.getenv('HOME'), '.xmen')`` then it is loaded else a ``xmen.yml`` will be generated
        from defaults."""
        self.server_host = 'xmen.rob-otics.co.uk'
        self.server_port = 8000
        self.user = None
        self.password = None
        self.local_host = socket.gethostname()
        self.local_user = getpass.getuser()
        self.prompt = True
        self.save_conda = True  # Whether to save conda info to file
        self.redirect_stdout = True  # Whether to also log the stdout and stderr to a text file in each experiment dir
        self.requeue = True   # Whether to requeue experiments if SLURM is available
        self.header = ''
        self.python_experiments = {}
        self.linked = []

        # private attributes (also saved)
        self._dir = os.getenv('XMEN', os.path.join(os.getenv('HOME'), '.xmen'))

        if not os.path.exists(self._path):
            # make the directory if it doesnt exist
            if not os.path.exists(self._dir):
                os.makedirs(self._dir)
            # save the config file
            self.to_yml()
        else:
            self.from_yml()

    @property
    def _path(self): return os.path.join(self._dir, 'xmen.yml')

    def _migrate(self):
        """Generate a xmen.yml file from the now depreciated config.yml file."""
        if os.path.exists(os.path.join(self._dir, 'config.yml')):
            config = GlobalExperimentManager()
            self.python_experiments = config.python_experiments
            self.prompt = config.prompt_for_message
            self.save_conda = config.save_conda
            self.redirect_stdout = config.redirect_stdout
            self.requeue = config.requeue
            self.header = config.header
            self.link(config.paths())
            return True
        else:
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.to_yml()

    @property
    def socket(self): return get_socket()

    @property
    def context(self): return get_context()

    def filter(self, pattern=None):
        """Return a list of experiments currently linked with global configuration which match regex ``pattern``"""
        if pattern is not None:
            return [f for f in self.linked if re.match(pattern, f)]

    @property
    def settings(self):
        """Return a dictionary of all the attributes considered as configurable settings/"""
        return {
        k: v for k, v in self.__dict__.items()
        if k[0] != '_' and k not in ['linked', 'python_experiments']}

    @connected
    def send_request(self, requests, workers=8):
        """Send a list of requests to the server. When ``workers`` is greater than 0 the requests will be pooled
        between ``workers``"""
        if requests:
            if not isinstance(requests, (list, tuple)):
                requests = [requests]
            if workers:
                workers = min(len(requests), workers)
                from multiprocessing.pool import Pool
                with Pool(processes=workers) as pool:
                    responses = pool.map(_send, requests)
                return responses
            else:
                responses = []
                for r in requests:
                    responses.append(_send(r))
                return responses

    @connected
    def change_password(self, password, new_password):
        """Attempt to change the password for the current user."""
        with self.socket as ss:
            with self.context.wrap_socket(ss, server_hostname=self.server_host) as s:
                s.connect((self.server_host, self.server_port))
                send(ChangePassword(self.user, password, new_password), s)
                msg = receive(s)
                response = decode_response(msg)
                if isinstance(response, PasswordChanged):
                    self.password = password
                    self.to_yml()
                    print(response.msg)
                else:
                    print(response)

    def register_user(self, user, password):
        """Register user with with the server."""
        with self.socket as ss:
            with self.context.wrap_socket(ss, server_hostname=self.server_host) as s:
                # print('Attempting to connect to ', (self.server_host, self.server_port))
                s.connect((self.server_host, self.server_port))
                send(AddUser(user, password), s)
                dic = receive(s)
                response = decode_response(dic)
                if isinstance(response, UserCreated):
                    self.user, self.password = user, password
                    self.to_yml()
                    print(response.msg)
                elif isinstance(response, PasswordValid):
                    if user != self.user:
                        self.user, self.password = user, password
                        self.to_yml()
                        print(response.msg)
                        print(f'Switched user to {user}')
                    else:
                        print(f'Current user is {user}. No change.')
                elif isinstance(response, (PasswordNotValid, Failed)):
                    print('ERROR: ' + response.msg)
                else:
                    raise NotImplementedError

    def setup(self):
        """Run an interactive setup for user configuration."""
        print('Config Setup...')
        import ruamel.yaml
        for k, v in self.settings.items():
            if k not in ['password', 'user']:
                msg = input(f'{k} (default={v}): ')
                if msg:
                    msg = ruamel.yaml.load(msg, Loader=ruamel.yaml.SafeLoader)
                    self.__dict__[k] = msg
        msg = input('Would you like to link a user account with the xmen server? [y | n]: ')
        if msg == 'y':
            user = input('user: ')
            user = user.strip()
            from getpass import getpass
            password = getpass()
            password = password.strip()
            try:
                self.register_user(user, password)
            except FailedException as m:
                print(f'ERROR: {m.msg}')
        # finally
        if os.path.exists(os.path.join(self._dir, 'config.yml')):
            msg = input('Found old config. Would you like to move '
                        'experiments across to the new configuration? [y | n]: ')
            if msg == 'y':
                self._migrate()
        self.to_yml()

    def to_yml(self):
        """Save the current config"""
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.default_flow_style = False
        with open(self._path, 'w') as file:
            yaml.dump(self.__dict__, file)

    def from_yml(self):
        """Load the current config"""
        with open(self._path, 'r') as file:
            import ruamel.yaml
            params = ruamel.yaml.load(file, Loader=ruamel.yaml.SafeLoader)
            for k, v in params.items():
                self.__dict__[k] = v

    def load_params(self, root):
        """Load parameters for an experiment"""
        from xmen.utils import dic_from_yml
        return dic_from_yml(path=os.path.join(root, 'params.yml'))

    def link(self, roots):
        """Link an experiment instance with the global configuration"""
        if not isinstance(roots, (list, tuple)):
            roots = [roots]
        self.linked += [r for r in roots]
        requests = []
        for root in roots:
            data = open(os.path.join(root, 'params.yml'), 'r').read()
            status = self.load_params(root)['_status']
            requests.append(
                LinkExperiment(
                    user=self.user,
                    password=self.password,
                    root=f'{self.local_user}@{self.local_host}:{root}',
                    data=data,
                    status=status))
        self.send_request(requests)
        # finally update self
        self.to_yml()

    def sync(self, roots=None, max_processes=8):
        """Synchronise linked experiments with the server. If experiments are found with the server that
        exist on the current host but are not currently linked then they will be optionally relinked.
        If they are found to not exist on the current host they will be deleted from the server.
        """
        from xmen.utils import dic_from_yml
        if roots is None:
            roots = self.linked
        response = self.send_request([
            GetExperiments(
                user=self.user,
                password=self.password,
                roots=f'{self.local_user}@{self.local_host}:.*',
                max_n=100000,
                status='.*')], workers=0)[0]
        requests = []
        if response.matches:
            uid, root, status, user, added, updated, data = zip(*response.matches)
            data = [dic_from_yml(string=d) for d in data]
            # request that folders that are not currently linked with the global config are
            # deleted from the server
            relink = []
            delete = []
            for r, d, u, s in zip(root, data, updated, status):
                if not d['_root'] in self.linked and s != xmen.experiment.DELETED:
                    if os.path.exists(os.path.join(d['_root'], 'params.yml')):
                        relink += [d['_root']]
                    else:
                        delete.append(d['_root'])
                        requests.append(
                            DeleteExperiment(
                                user=self.user,
                                password=self.password,
                                root=r))
            if delete:
                print('The following experiments were found for deletion with the server:')
                for r in delete:
                    print(r)
                msg = 'y'
                if self.prompt:
                    msg = input('Would you like to remove them from the server? [y | n] ')
                if msg != 'y':
                    requests = []
            if relink:
                print('The following experiments exist but are not currently linked with the config:')
                for r in relink:
                    print(r)
                msg = 'y'
                if self.prompt:
                    msg = input('Would you like to relink them? [y | n] ')
                if msg == 'y':
                    self.link(relink)
        for root in roots:
            data = open(os.path.join(root, 'params.yml'), 'r').read()
            status = self.load_params(root)['_status']
            requests.append(
                UpdateExperiment(
                    user=self.user,
                    password=self.password,
                    root=f'{self.local_user}@{self.local_host}:{root}',
                    data=data,
                    status=status))
        self.send_request(requests, max_processes)
        # finally update self
        self.to_yml()

    def add_python(self, module, name):
        """Add a experiments experiment ``name`` defined in ``module`` either as a class or function. """
        import stat
        from xmen.utils import get_run_script
        script = get_run_script(module, name)

        # Make executable run script
        path = os.path.join(self._dir, 'experiments')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '.'.join([module, name]))
        # Note old experiments will be written over
        open(path, 'w').write(script)
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)
        self.python_experiments.update({name: path})
        self.to_yml()

    def cache(self, *, save: Tuple[str, str] = None, load: bool = None, maximum=5):
        """Cache results recieved from the server"""
        from xmen.utils import dic_from_yml
        import json
        import pickle
        cache_dir = os.path.join(self._dir, 'cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cached = sorted(os.listdir(cache_dir))
        if save:
            dic, timestamp = save
            # delete old cache from the buffer
            while len(cached) > maximum - 1:
                c = cached.pop(0)
                file_path = os.path.join(cache_dir, c)
                os.unlink(file_path)
            # save new cache
            with open(os.path.join(cache_dir, timestamp), 'w') as f:
                json.dump([(k, v) for k, v in dic.items()], f)
        elif load:
            from collections import OrderedDict
            dic, timestamp = OrderedDict(), '1960-01-01 00:00:00'
            if cached:
                while True:
                    timestamp = cached.pop(-1)
                    try:
                        with open(os.path.join(cache_dir, timestamp), 'r') as f:
                            dic = OrderedDict(json.load(f))
                    except json.JSONDecodeError:
                        pass
                    else:
                        break
            return dic, timestamp

    def clean(self):
        """Iteratively search through experiment and remove any that no longer exist.

        Note:
            This is performed by checking that the experiment.yml file exists.
        """
        corrupted = []
        for p in self.linked:
            if not os.path.isfile(os.path.join(p, 'params.yml')):
                corrupted += [p]
        if corrupted:
            print('The following experiments were found for removal from the global configuration... ')
            for c in corrupted:
                print('  - ' + c)
            if self.prompt:
                msg = input('Would you like to remove them from the global configuration? [y | n]')
                if msg != 'y':
                    return

        requests = []
        for c in corrupted:
            self.linked.remove(c)
            requests.append(
                DeleteExperiment(self.user, self.password, f'{self.local_user}@{self.local_host}:{c}'))
        self.send_request(requests)
        self.to_yml()
        print('Experiments were successfully removed from the global configuration')

    def __repr__(self):
        """Pretty print the current configuration"""
        from xmen.utils import recursive_print_lines
        return '\n'.join(recursive_print_lines(self.__dict__))


class NoMatchException(Exception):
    def __init__(self, path):
        """Experiment at ``path`` does not contain a match"""
        super(NoMatchException, self).__init__()
        self.path = path


class GlobalExperimentManager(object):
    """DEPRECIATED A helper class used to manage global configuration of the Experiment Manager"""
    def __init__(self):
        import socket
        self.python_experiments = {}   # A dictionary of paths to experiments modules compatible with the experiment api
        self.python_paths = []         # A list of experiments paths needed to run each module
        self.prompt_for_message = True
        self.save_conda = True         # Whether to save conda info to file
        self.redirect_stdout = True    # Whether to also log the stdout and stderr to a text file in each experiment dir
        self.requeue = True
        self.experiments = {}          # A list of all experiments roots with an Experiment Manager
        # self.experiments = {}
        self.meta = get_meta()
        self.header = ''
        self.host = socket.gethostname()
        self.port = 2030
        self._dir = os.path.join(os.getenv('HOME'), '.xmen')

        if not os.path.isdir(self._dir):
            os.makedirs(self._dir)
            self._to_yml()
        else:
            self._from_yml()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._to_yml()

    def _to_yml(self):
        """Save the current config to an ``config.yaml``"""
        import ruamel.yaml
        params = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        with open(os.path.join(self._dir, 'config.yml'), 'w') as file:
            ruamel.yaml.dump(params, file, Dumper=ruamel.yaml.RoundTripDumper)

    def _from_yml(self):
        """Load the experiment config from a ``config.yml`` file"""
        with open(os.path.join(self._dir, 'config.yml'), 'r') as file:
            import ruamel.yaml
            from ruamel.yaml.comments import CommentedMap
            params = ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)
            to_yml = False
            for k, v in params.items():
                if k == 'experiments' and len(v) != 0 and type(v) is not CommentedMap:
                    experiments = {}
                    to_yml = True
                    for vv in v:
                        em = xmen.manager.ExperimentManager(root=vv, headless=True)
                        experiments[em.root] = {
                            'created': em.created, 'type': em.type, 'purpose': em.purpose, 'notes': em.notes}
                    self.__dict__[k] = experiments
                else:
                    self.__dict__[k] = v
            if 'meta' not in params:
                to_yml = True
        if to_yml:
            self._to_yml()

    def update_meta(self):
        for p in self.experiments:
            em = xmen.manager.ExperimentManager(p)
            em.update_meta()

    def clean(self):
        """Iteratively search through experiment sets and remove any that no longer exist.

        Note:
            This is performed by checking that experiment folder is a valid set using
            using ``ExperimentManager(p).check_initialised()``

        """
        corrupted = []
        for p in self.experiments:
            try:
                em = xmen.manager.ExperimentManager(p)
                em.check_initialised()
            except:
                corrupted.append(p)
        if len(corrupted) != 0:
            print('The following experiments were removed from the global configuration... ')
            for c in corrupted:
                self.experiments.pop(c)
                print('  - ' + c)

    def __repr__(self):
        """The current configuration as a string"""
        string = f'prompt for message: {self.prompt_for_message}\n'
        string += f'save conda: {self.save_conda}\n'
        string += f'enable requeue: {self.requeue}\n'
        string += f'host: {self.host}\n'
        string += f'port: {self.port}\n'
        string += f'redirect stdout: {self.redirect_stdout}\n'
        string += f'experiments Path:\n'
        for p in self.python_paths:
            string += f'  - {p}\n'
        string += 'experiments experiments:\n'
        for k, v in self.python_experiments.items():
           string += f'  - {k}: {v}\n'
        string += 'header:\n'
        string += self.header + '\n'
        string += 'meta:\n'
        for k, m in self.meta.items():
            string += f'  - {k}: {m}\n'
        string += 'experiments:\n'
        for k, e in self.experiments.items():
            e = dict(e)
            string += f'  - {e["created"]}: {k}\n'

        return string

    def find(self, mode='all', last=None, pattern="*", param_match=None, types_match=None,
             load_defaults=False, missing_entry=''):
        """Search through all experiments in the globally filtering by experiment location, parameters, and type.

        Args:
            mode (str): If 'all' then all experiments for each set will be concatenated and summarised individually
                whilst if 'set' then experiments will be summarised and searched for based on the default parameters.
            paths (List[str]): Extract all information from the passed list of path to experiments. If None then
                a search will be performed instead.
            pattern (str): A unix glob pattern of experiments to consider
            param_match (list): A list of strings providing a condition to filter experiments by their parameters.
                Compatible string formats are "regex", "regex op val" or "val1 op1 regex op2 val2" where op is in
                ``[<=,==,>=,<,>,!=]``. If just a regex is supplied then only experiments with parameters matching the
                regex will be returned. If an op is supplied then only experiments with all parameters which match the
                regex and satisfy the condition will be returned.
            types_match (list): A list of types of experiments to search for
            load_defaults (bool): If True then ``defaults.yml`` files will be loaded and the parameters of each
                experiment inferred from the overides defined in the ``experiment.yml`` file for each experiment set.
                This is potentially faster (less io) but means that the messages for each experiment will not be
                returned.
            missing_entry: Assign this parameter to missing values (see notes).

        Returns:
            matches (dict): A dict with possible keys ``['_root', '_purpose', '_type', '_notes',  '_name', '_created',
                '_version', '_status', '_messages', '_experiments', *param_keys]`` where  ``param_keys`` correspond to
                any parameters that satisfy the conditions in param_match. The keys
                ``['_root', '_purpose', '_type', '_notes', '_created']`` will always be present,
                ``['_name', '_version']`` will be added if ``load_defaults==True`` and ``['_name', '_version',
                '_status', '_messages]`` if ``load_defaults==False``. If [mode == 'set'] then
                the key  '_experiments' will also be added (giving a list of paths to the experiments in each set).
                The created dates in this case correspond to the date the set was created
                (given in defaults.yml).
            special_keys (dict): A list of keys starting with '_' present in the dictionary.

        Note:
            Due to the use of regex and conditions it is possible for parameters to match in one experiment which are
                not present in another. In this case missing parameters are given a value equal to ``missing_string``.
        """
        import fnmatch
        import re

        def _comparison(*args):
            arg = ''.join(args)
            try:
               out = eval(arg)
            except:
                out = False
            return out

        # Will use same pattern to search for  parameters and messages
        # All special parameters are recorded
        # From experiment.yml
        table = {'_root': [], '_purpose': [], '_type': [], '_notes': [], '_meta': []}
        # Additional
        if load_defaults:
            table.update({'_name': [], '_created': [], '_version': []})
        else:
            table.update({'_name': [], '_created': [], '_status': [], '_messages': [], '_version': []})
        if mode != 'all':
            table.update({'_experiments': []})

        # Filter experiment by name
        encountered = []
        if last is None:
            # find all possible paths to experiments
            experiments = fnmatch.filter(self.experiments.keys(), pattern)
            paths = []
            for root in experiments:
                em = xmen.manager.ExperimentManager(root)
                if load_defaults:
                    defaults = em.load_defaults()
                if types_match is not None:
                    if em.type not in types_match:
                        raise NoMatchException(root)
                paths += em.experiments if mode == 'all' else [em.defaults]

        else:
            paths = [os.path.join(r, n) for r, n in zip(last['_root'], last['_name'])]
        # for root in experiments:
        #     em = xmen.manager.ExperimentManager(root)
        #     if load_defaults:
        #         defaults = em.load_defaults()
        #     if types_match is not None:
        #         if em.type not in types_match:
        #             raise NoMatchException(root)
        #
        #     paths = em.experiments if mode == 'all' else [em.defaults]
        valid = []
        managers = {}
        for i, path in enumerate(paths):
            try:
                if mode == 'all':
                    if not load_defaults:
                        import ruamel.yaml
                        with open(os.path.join(path, 'params.yml'), 'r') as params_yml:
                            params = ruamel.yaml.load(params_yml, ruamel.yaml.RoundTripLoader)
                        if params is None:
                            print(path)
                    else:
                        from copy import deepcopy
                        params = deepcopy(defaults)
                        params.update(em.overides[i])
                else:
                    params = em.load_defaults()

                # remove notes (_notes will be removed from version 0.2.5)
                try:
                    params.pop('_notes')
                except KeyError:
                    pass

                # Find experiments who's parameters match all conditions
                if param_match is not None:
                    extracted = {}
                    # Search through al param matches
                    for m in param_match:
                        # Condition from param string
                        splits = re.split(r'(<=|==|>=|<|>|!=)', m.replace(' ', ''))
                        if len(splits) == 1:
                            reg = splits[0]
                            valid = lambda x: True
                        elif len(splits) == 3:
                            reg, op, y = splits
                            valid = lambda x: _comparison(str(x), op, y)
                        elif len(splits) == 5:
                            y1, op1, reg, op2, y2 = splits
                            valid = lambda x: _comparison(y1, op1, str(x), op2, y2)
                        else:
                            if len(splits) > 5:
                                print('ERROR: Only strings of the form (3.0 | cat) < param <= (4.0 | dog) can be set but'
                                      'got match_keys')
                                exit()

                        # Keep keys if match or special (_...)
                        keys = [k for k in params if re.match(reg, k)]

                        # the second term here operates as an and operator
                        # in the case that the regex pattern matches multiple parameters in the params
                        # file all parameters must satisfy the condition.
                        if len(keys) == 0 or not all(valid(params[k]) for k in keys):
                            keys += [k for k in params if k.startswith('_')]
                            raise NoMatchException(root)

                        # extract info
                        keys += [k for k in params if k.startswith('_')]
                        for key in keys:
                            extracted[key] = params[key]

                    if len(extracted) == 0:
                        raise NoMatchException(root)
                    else:
                        # If we get here add parameters to experiments
                        # (extracted must have at least one element)
                        # Infer current length of table (same length for every entry)
                        table_length = max([len(v) for v in table.values()])
                        if any([len(next(iter(table.values()))) != len(v) for v in table.values()]):
                            raise RuntimeError('This should not happen')

                        for k, v in extracted.items():
                            if k not in ['_root', '_purpose']:
                                if k not in table:
                                    encountered += [k]
                                    table[k] = [missing_entry] * table_length + [v]
                                else:
                                    table[k] += [v]

                        if '_meta' not in extracted:
                            table['_meta'] += [get_meta()]
                        for k in encountered:
                            if k not in extracted:
                                table[k] += [missing_entry]
                else:
                    for k, v in params.items():
                        if k.startswith('_') and k not in ['_root', '_purpose', '_notes']:
                            if k not in table:
                                table[k] = [v]
                            else:
                                table[k] += [v]
                    if '_meta' not in params:
                        table['_meta'] += [get_meta()]

                root, _ = os.path.split(path)
                if root not in managers:
                    managers[root] = xmen.manager.ExperimentManager(root)
                em = managers[root]
                # Add data to table from experiment.yml
                if load_defaults:
                    table['_name'] += [path.replace(em.root, '')[1:]]
                if mode != 'all':
                    table['_experiments'] += [em.experiments]

                table['_root'] += [em.root]
                table['_purpose'] += [em.purpose]
                table['_type'] += [em.type]
                table['_notes'] += [em.notes]

            except NoMatchException as m:
                continue

        if mode != 'all':
            table.pop('_name')
        return table

    def find_to_dataframe(self, table, verbose=True, display_git=None,
                          display_purpose=None, display_date=None, display_messages=None, display_status=None,
                          display_meta=None, display_params=None):
        """Convert the output of the `find` method to a formatted data frame configured by args. If verbose then all
        entries will be displayed (independent of the other args)"""

        # manipulate meta
        def flatten_dict(dic, prepends=()):
            if isinstance(dic, dict):
                out = {}
                for k, v in dic.items():
                    if isinstance(v, dict):
                        out.update(flatten_dict(v, (*prepends, k)))
                    else:
                        out.update({"_".join([*prepends, k]): v})
                return out

        import pandas as pd
        param_keys = [k for k in table if not k.startswith('_')]
        special_keys = [k for k in table if k.startswith('_')]
        display = {
            'root': table.pop('_root')}

        if '_name' in special_keys:
            display['name'] = table.pop('_name')
        display.update({
            'created': table.pop('_created'),
            'type': table.pop('_type'),
            'purpose': table.pop('_purpose')})

        if '_origin' in special_keys:
            table.pop('_origin')

        # meta = table.pop('_meta')
        # meta_dict = {}
        # for m in meta:
        #     for k, v in m.items():
        #         if k in meta_dict:
        #             meta_dict[k] += [m[k]]
        #         else:
        #             meta_dict[k] = [m[k]]
        # display.update(meta_dict)

        meta = table.pop('_meta')  # List of dict
        meta_dict = {}
        for i, m in enumerate(meta):
            m = flatten_dict(m)
            for k, v in m.items():
                # if isinstance(v, dict):
                #     v = flatten_dict(v, (str(k), ))
                if k not in meta_dict:
                    meta_dict[k] = [''] * i + [v]
                else:
                    meta_dict[k] += [v]
            for k in meta_dict:
                if k not in m:
                    meta_dict[k] += ['']
        display.update(meta_dict)


        # Remove Notes
        table.pop('_notes')

        if '_status' in special_keys:
            display['status'] = table.pop('_status')
        # Add version information to table
        versions = table.pop('_version')

        git_keys = ['local', 'remote', 'commit', 'branch']
        for k in git_keys:
            display[k] = [version.get('git', {}).get(k, None) for version in versions]

        # Add messages to table
        encountered = []
        if '_messages' in special_keys:
            messages = table.pop('_messages')  # List of dict
            message_dict = {}
            for i, m in enumerate(messages):
                for k, v in m.items():
                    if k not in message_dict:
                        message_dict[k] = [''] * i + [v]
                    else:
                        message_dict[k] += [v]
                for k in message_dict:
                    if k not in m:
                        message_dict[k] += ['']
            display.update(message_dict)
        if verbose:
            display_keys = list(display.keys())
        else:
            display_keys = [v for v in ('root', 'name') if v in display]
            if display_date:
                display_keys += ['created']
            if display_status:
                display_keys += ['status']
            if display_git:
                display_keys += git_keys
            if display_meta:
                display_keys += [k for k in meta_dict.keys() if re.match(display_meta if isinstance(
                    display_meta, str) else '.*', k)]  + ['origin']
                # display_keys += list(meta_dict.keys()) + ['origin']
            if (isinstance(display_messages, str) or display_messages) and '_messages' in special_keys:
                display_keys += [k for k in message_dict.keys() if re.match(display_messages if isinstance(
                    display_messages, str) else '.*', k)]
                    # list(message_dict.keys())
                display_keys += ['date']
            if display_purpose:
                display_keys += ['purpose']

            if display_params:
                display_keys += [k for k in param_keys if re.match(display_params, k)]
            else:
                display_keys += list(table.keys())

        # Add other params
        display.update(table)
        df = pd.DataFrame(display)
        # Shorten roots
        roots = [v["root"] for v in df.transpose().to_dict().values()]
        prefix = os.path.dirname(os.path.commonprefix(roots))
        if prefix != '/':
            roots = [r[len(prefix) + 1:] for r in roots]
        df.update({'root': roots})
        # Finally filter
        df['name'] = df['root'] + '/' + df['name']

        # display_keys += ['path']
        # display_keys.remove('name')
        display_keys.remove('root')
        df = df.filter(items=display_keys)

        keys = [s.replace('slurm_', 's') for s in display_keys]
        keys = [s.replace('virtual_', 'v') for s in keys]

        updated_keys = {o: n for o, n in zip(display_keys, keys)}
        get_rid = {'sid', 'sMCS_label', 'sQOS', 'sTimeMin', 'sEligibleTime', 'sAccrueTime', 'sDeadline',
                   'sSecsPreSuspend', 'sJobName',  'sSuspendTime',  'sLastSchedEval', 'sReqNodeList', 'sExcNodeList',
                   'sNice'}

        df = df.rename(updated_keys, axis='columns')
        df = df.filter(items=[k for k in keys if k not in get_rid])
        return df, prefix

    def add_experiment(self, module, name):
        """Add a experiments experiment ``name`` defined in ``module`` either as a class or function."""
        import stat
        from xmen.utils import get_run_script
        script = get_run_script(module, name)

        # Make executable run script
        path = os.path.join(self._dir, 'experiments')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '.'.join([module, name]))
        # Note old experiments will be written over
        open(path, 'w').write(script)
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)
        self.python_experiments.update({name: path})
        self._to_yml()

    def paths(self, pattern=None, types_match=None):
        import fnmatch
        experiments = self.experiments.keys()
        if pattern is not None:
            experiments = fnmatch.filter(experiments, pattern)
        paths = []
        for root in experiments:
            em = xmen.manager.ExperimentManager(root)
            try:
                em.check_initialised()
            except xmen.manager.InvalidExperimentRoot as m:
                continue
            if types_match is not None:
                if em.type not in types_match:
                    continue
            paths += em.experiments
        return paths




