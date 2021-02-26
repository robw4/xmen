"""A module containing the Experiment class definition."""
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

import sys
import datetime
import argparse
from typing import Optional, Dict, List, Any
import signal
import io

from argparse import RawTextHelpFormatter
from xmen.utils import get_meta, get_version, commented_to_py, DATE_FORMAT, recursive_print_lines, TypedMeta, MultiOut
from xmen.server import *
import os

helps = {
    'execute': 'Execute the experiment from a given params.yml file linked to folder. If no folder is '
               'passed the experiment will be run in a detached state. This is useful for debugging but is'
               'not recommended for deployment.',
    'update': 'Update the parameters given by a yaml string. Note this will be called before '
              'other flags and can be used in combination with --to_root, --to_defaults, and --register',
    'root': 'Generate a run script and defaults.yml file for interfacing with xgent',
    'debug': 'Run experiment in debug mode. The experiments debug will be called before registering.',
    'txt': 'Also log stdout and stderr to an out.txt file. Enabled by default (default taken from xmen config)',
    'restart': 'Restart the experiment',
    'purpose': 'A string giving the purpose of the current experiment'}

from xmen.config import Config

CONFIG = Config()

import textwrap
for k in helps:
    helps[k] = '\n'.join(textwrap.wrap(helps[k], 50))


class NullRoot(str):
    def __new__(cls):
        obj = str.__new__(cls, os.devnull)
        return obj

    def __add__(self, other):
        return self


experiment_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
experiment_parser.add_argument('--update', '-u', type=str, default=None, nargs='+', help=helps['update'])
experiment_parser.add_argument('--purpose', '-p', type=str, default=None, nargs='+', help=helps['purpose'])
experiment_parser.add_argument('--execute', '-x', type=str, default=None, help=helps['execute'],
                               nargs='?', const=NullRoot())
experiment_parser.add_argument('--to_root', '-r', type=str, default=None, help=helps['root'])
# optional extras
experiment_parser.add_argument('--debug', '-d', default=None, action='store_true', help=helps['debug'])
experiment_parser.add_argument('--txt', '-t', default=not CONFIG.redirect_stdout, action='store_true', help=helps['txt'])
experiment_parser.add_argument('--restart', '-f', default=None, action='store_true', help=helps['restart'])
_SPECIALS = ['_root', '_status', '_purpose', '_messages', '_version', '_meta']
_SPECIALS += ['_user', '_host', '_notes', '_timestamps']
_DEPRECIATED = ['_name', '_created']


class TimeoutException(Exception):
    pass


def get_time():
    return datetime.datetime.now().strftime(DATE_FORMAT)


DEFAULT = 'default'
REGISTERED = 'registered'
RUNNING = 'running'
ERROR = 'error'
STOPPED = 'stopped'
TIMEOUT = 'timeout'
FINISHED = 'finished'
DETACHED = 'detached'
REQUEUE = 'requeue'
DELETED = 'deleted'


def get_timestamps(created=None, start=None, stopped=None, last=None, registered=None):
    return {
        'created': created,
        'start': start,
        'stopped': stopped,
        'last': last,
        'registered': registered}


class Experiment(object, metaclass=TypedMeta):
    """Base class from which all other experiments derive. Experiments are defined by:

    1. *Parameters*: class attributes declared with the special parameter ``# @p`` in a comment after the definition.
    2. *Execution*: defined by overloading the ``run()`` method

    For example::

        class AnExperiment(Experiment):
             ''' Doc strings should be used to document the purpose of the experiment'''
            # experiment parameters are defined as class attributes with an @p tag in there comment
            a: str = 'Hello'   # @p a parameter
            b: str = 'World!'  # @p another parameter

            # experiment execution code is defined in the experiments run method
            def run(self):
                print(f'{self.a} {self.b})
    """
    _params = {}  # Used to store parameters registered by the MetaClass

    def __init__(self, root=None, purpose='', copy_params=True, **kwargs):
        """Create a new experiment object.

        Args:
            root, name (str): If not None then the experiment will be registered to a folder ``{root}\{name}``
            purpose (str): An optional string giving the purpose of the experiment.
            copy_params (bool): If True then parameters are deep copied to the object instance from the class definition.
                Mutable attributes will no longer be shared.
            **kwargs: Override parameter defaults.
        """
        import copy
        if copy_params:
            for k in [k for k in dir(self) if k in self._params]:
                setattr(self, k, copy.deepcopy(getattr(self, k)))

        if root is None:
            self._root: Optional[str] = None     # @p The root directory of the experiment
            self._status: str = DEFAULT  # @p One of ['default' | 'created' | 'running' | 'error' | 'finished']
            # self._created: Optional[float] = None  # @p Initial time the experiment was created
            self._notes: Optional[List[str]] = None  # @p Notes attached to the experiment
            self._purpose: Optional[str] = None  # @p A description of the experiment purpose
            # new attributes
            self._user: Optional[str] = CONFIG.user  # @p The user of the experiment
            self._host: Optional[str] = CONFIG.local_host  # @p The name of the default host
            self._timestamps: Dict[str, Optional[str]] = get_timestamps()   # @p timestamps attached to the experiment
            # These can all be varied
            self._messages: Dict[Any, Any] = {}  # @p Messages left by the experiment
            self._version: Optional[Dict[Any, Any]] = None  # @p Experiment version information. See `get_version`
            self._meta: Optional[Dict] = None  # @p The global configuration for the experiment manager

            # depreciated
            self._specials: List[str] = _SPECIALS
            self._helps: Optional[Dict] = None

            # queues
            self._queue = None
            # self._queues = []
            self._processes = []

        else:
            raise ValueError("Either both or neither of name and root can be set")

        # Update kwargs
        self.update(kwargs)
        if root is not None:
            self.register(root, purpose=purpose)

    @property
    def root(self):
        """The root directory to which the experiment belongs"""
        return self._root

    @property
    def directory(self):
        return self._root

    @property
    def status(self):
        """The status of the experiment. One of ``'default'``, ``'registered'``, ``'running'``, ``'finished'`` or
         ``'error'``."""
        return self._status

    @property
    def created(self):
        """The date the experiment parameters was first registered."""
        return self._timestamps['created']

    @property
    def start(self):
        """The time the experiment last started running."""
        return self._timestamps['start']

    @property
    def registered(self):
        """The time the experiment was last registered."""
        return self._timestamps['registered']

    @property
    def stopped(self):
        """The time the experiment was last stopped."""
        return self._timestamps['stopped']

    @property
    def last(self):
        """The time the experiment state was last communicated."""
        return self._timestamps['last']

    @property
    def purpose(self):
        """A string giving a purpose message for the experiment"""
        return self._purpose

    @property
    def messages(self):
        """A dictionary of messages logged by the experimet."""
        return self._messages

    @property
    def version(self):
        """A dictionary giving the version information for the experiment"""
        return self._version

    @property
    def notes(self):
        """A dictionary containing the notes attached to the experiment"""
        return self._notes

    @property
    def user(self):
        """The current user of the experiment"""
        return self._user

    @property
    def host(self):
        """The host of the experiment"""
        return self._host

    @root.setter
    def root(self, value):
        raise AttributeError('Property root cannot be set.')

    @status.setter
    def status(self, value):
        raise AttributeError('Property status cannot be set.')

    @created.setter
    def created(self, value):
        raise AttributeError('Property created cannot be set.')

    @purpose.setter
    def purpose(self, value):
        raise AttributeError('Property purpose cannot be set.')

    @messages.setter
    def messages(self, value):
        raise AttributeError('Property messages cannot be set.')

    @version.setter
    def version(self, value):
        raise AttributeError('Property version cannot be set.')

    @user.setter
    def user(self, value):
        raise AttributeError('Property user cannot be set.')

    @host.setter
    def host(self, value):
        """The host of the experiment"""
        raise AttributeError('Property host cannot be set.')

    def note(self, string, remove=False):
        """Leave a note with the experiment. Will be removed if ``remove`` is ``False``"""
        if self._notes is None:
            self._notes = []
        if remove:
            self._notes.remove(string)
        else:
            self._notes.append(string)
        if not self._notes:
            self._notes = None

    def get_param_helps(self):
        """Get help for all attributes in class (including inherited and private)."""
        return {k: v[3].strip() for k, v in self._params.items()}

    def update_version(self):
        if hasattr(self, 'fn'):
            self._version = get_version(fn=self.fn)
        else:
            self._version = get_version(cls=self.__class__)

    def update_meta(self, get_platform=False, get_cpu=False, get_memory=False, save=False, **kwargs):
        self._meta = get_meta(get_platform, get_cpu, get_memory, **kwargs)
        if save:
            self._save()

    def to_defaults(self, defaults_dir):
        """Create a ``defaults.yml`` file from experiment object.
         Any base class inheriting from Experiment can create a default file as::

            MyExperiment().to_yaml('/dir/to/defaults/root')
        """
        assert self._status == DEFAULT, 'An experiment can only be converted to default if it has not been registered'
        self.update_version()
        self.update_meta()
        self._timestamps['created'] = get_time()
        if not os.path.exists(defaults_dir):
            os.makedirs(defaults_dir)
        self._save(defaults_dir)

    def as_yaml(self, as_string=True):
        from xmen.utils import dic_to_yaml
        from ruamel.yaml.comments import CommentedMap
        params = {k: getattr(self, k) for k in dir(self) if k in self.param_keys() or k in self._specials}
        params = {k: v for k, v in params.items() if '_' + k not in self.__dict__}
        helps = self.get_param_helps()

        # Add definition module to experiment object
        map = CommentedMap()
        for i, (k, v) in enumerate(params.items()):
            if self._status == DEFAULT:
                if k in ['_root', '_status', '_purpose',
                         '_messages', '_origin', '_timestamps', '_notes', '_type']:
                    continue
            comment = helps[k]
            if comment == '':
                comment = None
            map.insert(i, k, v, comment=comment)

        if as_string:
            return dic_to_yaml(map)
        else:
            return map

    def to_update_request(self):
        from xmen.utils import dic_from_yml
        from xmen.config import Config
        import json
        config = Config()
        string = self.as_yaml()
        dic = dic_from_yml(string=string)
        data = json.dumps(dic)
        root = f'{config.local_user}@{config.local_host}:{self.root}'
        return UpdateExperiment(
            user=config.user, password=config.password,
            root=root, data=data, status=self.status)

    def _save(self, defaults_dir=None):
        """Save experiment to either a defaults.yml file or a params.yml file depending on its status"""
        if self._status == DEFAULT:
            path = os.path.join(os.path.join(defaults_dir, 'defaults.yml'))
        else:
            path = os.path.join(self.directory, 'params.yml')

        if self.status == RUNNING:
            self._timestamps['last'] = get_time()

        # save parameters (always)
        string = self.as_yaml()
        with open(path, 'w') as file:
            file.write(string)

        if self.status not in [DEFAULT, REGISTERED]:
            request = self.to_update_request()
            # send request to queues
            if self._queue is not None:
                self._queue.put(request)
            # for q in self._queues:
            #     print('Putting on queue')
            #     q.put(request)
        elif self.status == REGISTERED:
            # the global configuration will register with the server...
            Config().register(self.root)

    def debug(self):
        """Inherited classes may overload debug. Used to define a set of open_socket for minimum example"""
        return self

    def from_yml(self, path, copy=False):
        """Load state from either a ``params.yml`` or ``defaults.yml`` file (inferred from the filename).
        The status of the experiment will be updated to ``'default'`` if ``'defaults.yml'``
        file else ``'registered'`` if ``params.yml`` file.

        If copy is ``True`` then only user defined parameters themselves will be copied from the params.yml file.
        """
        from xmen.utils import dic_from_yml
        params = dic_from_yml(path=path)

        # backward compatibility
        if '_name' in params:
            params['_root'] = os.path.join(params['_root'], params.pop('_name'))
        if '_created' in params:
            params['_timestamps'] = get_timestamps()
            params['_timestamps']['created'] = params.pop('_created')
        params = {k: commented_to_py(v) for k, v in params.items() if k in self.__dict__}

        if copy:
            # Copy only parameter values themselves (and not specials)
            params = {k: v for k, v in params.items() if not k.startswith('_')}

        # update created date
        self.__dict__.update(params)

    def register(self, root, purpose='', force=True, same_names=100, restart=False, **_):
        """Register an experiment to an experiment directory. Its status will be updated to ``registered``. If an
        experiment called ``name`` exists in ``root`` and ``force==True`` then name will be appended with an int
        (eg. ``{name}_0``) until a unique name is found in ``root``. If ``force==False`` a ``ValueError`` will be raised.

        If restart is also passed, and an experiment called name also exists, then the experiment will be loaded
        from the params.yml file found in ``'{root}/{name}'``.

        Raises:
            ValueError: if ``{root}/{name}`` already contains a ``params.yml`` file
        """
        exists = os.path.exists(os.path.join(root, 'params.yml'))
        if not restart or (restart and not exists):
            if exists:
                i = 0
                if force:
                    while i < same_names:
                        if not os.path.exists(os.path.join(root + '_' + str(i), 'params.yml')):
                            root += '_' + str(i)
                            # name += '_' + str(i)
                            break
                        i += 1
                elif i == same_names or not force:
                    raise ValueError(f'Experiment folder {root} already contains a params.yml file. '
                                     f'An Experiment cannot be created in an already existing experiment folder')

            # Make the folder if it does not exist
            if not os.path.isdir(root):
                os.makedirs(root)

            self.update_version()  # Get new version information
            self.update_meta()   # Get the newest meta information
            self._root = root
            self._purpose = purpose
            if self._timestamps['created'] is None:
                self._timestamps['created'] = get_time()
            self._timestamps['registered'] = get_time()
            self._status = REGISTERED
            self._save()
        else:
            self.from_yml(os.path.join(root, 'params.yml'))
            if self.status != REGISTERED:
                self._status = REGISTERED
                self._save()

    def to_root(self, root_dir, shell='/bin/bash'):
        """Generate a ``defaults.yml`` file and ``script.sh`` file in ``root_dir``.

        Args:
            root_dir (str): A path to the root directory in which to generate a script.sh and defaults.yml to
                run the experiment.
        """
        import stat
        # get_git is deliberately called outside to_defaults as git information is also added to script.sh
        self.update_version()
        self.update_meta(save=False)

        from xmen.utils import get_run_script
        if hasattr(self, 'fn'):
            script = get_run_script(*self.fn)
            path = os.path.join(root_dir, '.'.join(self.fn))
        else:
            path = os.path.join(root_dir, '.'.join([self.__class__.__module__, self.__class__.__name__]))
            script = get_run_script(self.__class__.__module__, self.__class__.__name__)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        # Save to root directory
        open(path, 'w').write(script)
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)

        open(os.path.join(root_dir, 'script.sh'), 'w').write(
            f'#!{shell}\nexec {path} --execute ${{1}}')

        self.to_defaults(root_dir)

    def _update_status(self, status):
        """Update the status of the experiment"""
        self._status = status
        self._save()

    def detach(self):
        self._root = NullRoot()
        self._status = DETACHED

    def update(self, kwargs):
        """Update the parameters with a given dictionary"""
        if self._status in [DEFAULT, DETACHED]:
            if any([k not in self.param_keys() and k in self._specials for k in kwargs]):
                raise ValueError('Key not recognised!')
            else:
                self.__dict__.update(kwargs)
        else:
            raise ValueError('Parameters of a created experiment cannot be updated.')

    def param_keys(self):
        return self._params.keys()

    def __setattr__(self, key, value):
        """Attributes can only be changed when the status of the experiment is default"""
        if '_status' in self.__dict__:
            if key in self.param_keys() and self._status not in [DEFAULT, DETACHED] and key not in self._specials:
                raise AttributeError('Parameters can only be changed when status = "default" or "detached"')
        self.__dict__.update({key: value})

    def __enter__(self):
        def _sigusr1_handler(signum, handler):
            raise TimeoutException
        # set up the signal usr1 signal handler
        signal.signal(signal.SIGUSR1, _sigusr1_handler)
        # get all the meta information for the current system
        meta = get_meta(get_platform=True, get_cpu=True, get_memory=True, get_disk=True,
                        get_slurm=True, get_conda=CONFIG.save_conda, get_network=True, get_gpu=True,
                        get_environ=True)
        # save conda environment information
        # saved seperately from the params.yml file for clarity
        conda = meta.pop('conda', None)
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.default_flow_style = False
        if conda is not None:
            with open(os.path.join(self.directory, 'environment.yml'), 'w') as f:
                yaml.dump(conda, f)

        # update internal state
        self._meta = meta
        self._update_status(RUNNING)
        self._timestamps['start'] = get_time()

        # start messaging threads
        # from queue import Queue
        from multiprocessing import Process, Queue
        self._queue = Queue(maxsize=1)
        p = Process(target=send_request_task, args=(self._queue, ))
        p.start()
        # self._queues += [q]
        self._processes += [p]
        # finally save the experiment
        self._save()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self._timestamps['stopped'] = get_time()
        if exc_type is None:
            self._update_status(FINISHED)
        elif exc_type is KeyboardInterrupt:
            print('########################')
            print('Stopping experiment')
            print('########################')
            self._update_status(STOPPED)
        elif exc_type is TimeoutException:
            print('########################')
            print('Timeout encountered')
            print('########################')
            self._update_status(TIMEOUT)
            slurm_job = os.environ.get('SLURM_JOBID', None)
            if slurm_job is not None and CONFIG.reque:
                import subprocess
                self._update_status(REQUEUE)
                subprocess.call(['scontrol', 'requeue', f'{slurm_job}'])
        else:
            print('########################')
            print('An error occurred encountered')
            print('########################')
            if self.status not in [TIMEOUT, STOPPED]:
                self._update_status(ERROR)
        # stop running processes
        time.sleep(0.1)
        for p in self._processes:
            p.terminate()

    def __call__(self, *args, **kwargs):
        """Used to run experiment. Upon entering the experiment status is updated to ``'running`` before ``args`` and
        ``kwargs`` are passed to ``run()``. If ``run()`` is successful the experiment ``status`` is updated to
        ``'finished'`` else it will be given ``status='error'``.

        Both *args and **kwargs are passed to self.run. """
        if self._status == 'default':
            raise ValueError('An experiment in default status must be registered before it can be executed')
        with self:
            self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError('Derived classes must implement the run method in order to be called')

    def message(self, messages, keep='latest', leader=None):
        """Add a message to the experiment (and an experiments params.yml file). If the experiment is not registered to
        a root then no messages will be logged.

        Args:
            messages (dict): A dictionary of messages. Keys are interpreted as subjects and values interpreted as
                messages. If the ``defaults.yml`` already contains subject then the message for subject will be
                updated.
            keep (str): which message to keep in the case of collision. One of ['latest', 'min', 'max']
            leader (str): If not None then all messages will be saved if the keep condition is met for the leader key.

        Note:
            Only messages of type float, int and string are supported. Any other message will be converted to type float
            (if possible) then string thereafter.

        """
        if self._root is not None:
            # Add leader to messages group
            if leader is not None:
                best = self.compare(leader, messages[leader], keep)[0]
                if best:
                    self._messages.update({k: self.convert_type(v) for k, v in messages.items()})
            else:
                for k, v in messages.items():
                    if self.compare(k, v, keep)[0]:
                        self._messages.update({k: self.convert_type(v)})
            self._save()

    @staticmethod
    def convert_type(v):
        if not isinstance(v, (str, float, int)):
            try:
                v = float(v)
            except (ValueError, TypeError):
                v = str(v)
                pass
        return v

    def compare(self, k, v, keep='latest'):
        assert keep in ['max', 'min', 'latest']
        cur = self.messages.get(k, None)
        if cur is not None and keep in ['max', 'min']:
            out = {'max': max, 'min': min}[keep](v, cur)
            return out == v, out
        else:
            return True, v

    def parse_args(self):
        """Configure the experiment instance from the command line arguments.
        """
        from xmen.utils import IncompatibleYmlException
        experiment_parser.prog = f'xmen {self.__class__.__name__}'

        # Configure help information from the class
        n = 7
        experiment_parser.usage = '\n'.join(
            [' ' * 0 + experiment_parser.prog + ' ' + '-u YML -r PATH',
             ' ' * n + experiment_parser.prog + ' ' + '-u YML -x DIR ',
             ' ' * n + experiment_parser.prog + ' ' + '-x PARAMS'])
        # experiment_parser.description = self.__doc__
        args = experiment_parser.parse_args()

        # Run the debug method if implemented and
        # debug flag is passed.
        if args.debug is not None:
            # Update debug parameters
            print('Running as debug')
            self.debug()
        # update the parameter values from either
        # a parameter string or from a defaults.yml file.
        if args.update is not None:
            for update in args.update:
                # Case (1) update parameters from yaml string
                try:
                    self.from_yml(update, copy=True)
                except IncompatibleYmlException:
                    try:
                        # Update passed parameters
                        import ruamel.yaml
                        overrides = ruamel.yaml.load(update, Loader=ruamel.yaml.Loader)
                        print(f'Updating parameters {overrides}')
                        self.update(overrides)
                    except:
                        print(f'ERROR: {update} is either not a valid yaml string or '
                              f'is not a path to a defaults.yml or params.yml file')
                        exit()
        # Then execute the experiment
        if args.execute is not None:
            # (1) register experiment from pre-existing params.yml file
            if args.execute != os.devnull:
                if os.path.isfile(args.execute):
                    try:
                        self.from_yml(args.execute)
                        if not self.status == 'registered':
                            raise IncompatibleYmlException
                    except IncompatibleYmlException:
                        print(f'ERROR: File {args.execute} is not a valid params.yml file')
                # (2) register experiment to a repository
                else:
                    purpose = ''
                    if args.purpose is None and CONFIG.prompt:
                        purpose = input('Enter the purpose of the experiment: ')
                    self.register(args.execute, restart=args.restart, purpose=purpose)
            else:
                self.detach()
        return args

    def main(self, args=None):
        """Take the command line args and execute the experiment (see ``parse_args`` for more
         information). In order to expose the command line interface::

            if __name__ == '__main__':
                exp = AnExperiment().main()

        Note that for backwards compatibility it is also possible to pass ``args`` as an argument to ``main``.
        This allows the experiment to be run from the commandline as::

            if __name__ == '__main__':
                exp = AnExperiment()
                args = exp.parse_args()
                exp.main(args)
        """
        if args is None:
            args = self.parse_args()

        if all(a is None for a in (args.debug, args.execute, args.to_root, args.txt, args.update)):
            print(self.__doc__)
            print('\nFor more help use --help.')

        # Generate experiment root
        if args.to_root is not None:
            print(f'Generating experiment root at {args.to_root}')
            self.to_root(args.to_root)

        # Run the experiment
        if args.execute is not None:
            assert self.status in [REGISTERED, DETACHED, TIMEOUT, REQUEUE],\
                f'Experiment must be registered before execution but got {self.status}'
            # Configure standard out to print to the registered directory as well as
            # the original standard out
            if args.txt:
                self.stdout_to_txt()
            # Execute experiment
            try:
                self.__call__()
            except NotImplementedError:
                print(f'WARNING: The --execute flag was passed but run is not implemented for {self.__class__}')
                pass

    def stdout_to_txt(self):
        """Configure stdout to also log to a text file in the experiment directory"""
        sys.stdout = MultiOut(sys.__stdout__, open(os.path.join(self.directory, 'out.txt'), 'a+'))
        sys.stderr = MultiOut(sys.__stderr__, open(os.path.join(self.directory, 'out.txt'), 'a+'))

    def __repr__(self):
        """Provides a useful help message for the experiment"""
        # params = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        helps = self.get_param_helps()
        base_params = {k[1:]: v for k, v in self.__dict__.items() if k in self._specials}
        params = {k: getattr(self, k) for k in helps if not k.startswith('_')}
        lines = recursive_print_lines(base_params)
        lines += ['parameters:']
        lines += ['  ' + f'{k}: {v}' for k, v in params.items()]
        return '\n'.join(lines)

