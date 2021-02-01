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

import os

helps = {
    'execute': 'Execute the experiment from a given params.yml file linked to folder. If no folder is '
               'passed the experiment will be run in a detached state. This is useful for debugging but is'
               'not recommended for deployment.',
    'update': 'Update the parameters given by a yaml string. Note this will be called before '
              'other flags and can be used in combination with --to_root, --to_defaults, and --register',
    'root': 'Generate a run script and defaults.yml file for interfacing with xgent',
    'debug': 'Run experiment in debug mode. The experiments debug will be called before registering.',
    'txt': 'Also log stdout and stderr to an out.txt file. Enabled by default',
    'restart': 'Restart the experiment'}

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
experiment_parser.add_argument('--execute', '-x', type=str, default=None, help=helps['execute'],
                               nargs='?', const=NullRoot())
experiment_parser.add_argument('--to_root', '-r', type=str, default=None, help=helps['root'])
# optional extras
experiment_parser.add_argument('--debug', '-d', default=None, action='store_true', help=helps['debug'])
experiment_parser.add_argument('--to_txt', '-t', default=None, action='store_true', help=helps['txt'])
experiment_parser.add_argument('--restart', '-f', default=None, action='store_true', help=helps['restart'])
_SPECIALS = ['_root', '_name', '_status', '_created', '_purpose', '_messages', '_version', '_meta']


class IncompatibleYmlException(Exception):
    pass


class TimeoutException(Exception):
    pass


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

    def __init__(self, root=None, name=None, purpose='', copy=True, **kwargs, ):
        """Create a new experiment object.

        Args:
            root, name (str): If not None then the experiment will be registered to a folder ``{root}\{name}``
            purpose (str): An optional string giving the purpose of the experiment.
            copy (bool): If True then parameters are deep copied to the object instance from the class definition.
                Mutable attributes will no longer be shared.
            **kwargs: Override parameter defaults.
        """
        if copy:
            import copy
            for k in [k for k in dir(self) if k in self._params]:
                setattr(self, k, copy.deepcopy(getattr(self, k)))

        if (name is None) == (root is None):
            now_time = datetime.datetime.now().strftime(DATE_FORMAT)
            self._root: Optional[str] = None     # @p The root directory of the experiment
            self._name: Optional[str] = None     # @p The name of the experiment (under root)
            self._status: str = 'default'        # @p One of ['default' | 'created' | 'running' | 'error' | 'finished']
            self._created: str = now_time        # @p The date the experiment was created
            self._purpose: Optional[str] = None  # @p A description of the experiment purpose
            self._messages: Dict[Any, Any] = {}  # @p Messages left by the experiment
            self._version: Optional[Dict[Any, Any]] = None   # @p Experiment version information. See `get_version`
            self._meta: Optional[Dict] = None    # @p The global configuration for the experiment manager
            self._notes: Dict[str, str] = {}     # @p Notes attached to the experiment
            # self._origin: Optional[str] = None    # @p The path the experiment was initially registered at
            self._specials: List[str] = _SPECIALS
            self._helps: Optional[Dict] = None
        else:
            raise ValueError("Either both or neither of name and root can be set")

        # Update kwargs
        self.update(kwargs)
        if name is not None:
            self.register(name, root, purpose)

    @property
    def root(self):
        """The root directory to which the experiment belongs"""
        return self._root

    @property
    def name(self):
        """The name of the current experiment"""
        return self._name

    @property
    def directory(self):
        """The directory assigned to the experiment"""
        if self._status == 'default':
            return None
        else:
            return os.path.join(self._root, self._name)

    @property
    def status(self):
        """The status of the experiment. One of ``'default'``, ``'registered'``, ``'running'``, ``'finished'`` or
         ``'error'``."""
        return self._status

    @property
    def created(self):
        """The date the experiment parameters were last updated."""
        return self._created

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

    @root.setter
    def root(self, value):
        raise AttributeError('Property root cannot be set.')

    @name.setter
    def name(self, value):
        raise AttributeError('Property name cannot be set.')

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

    def get_param_helps(self):
        """Get help for all attributes in class (including inherited and private)."""
        return {k: v[3].strip() for k, v in self._params.items()}

    def update_version(self):
        if hasattr(self, 'fn'):
            self._version = get_version(fn=self.fn)
        else:
            self._version = get_version(cls=self.__class__)

    def update_meta(self):
        self._meta = get_meta()

    def to_defaults(self, defaults_dir):
        """Create a ``defaults.yml`` file from experiment object.
         Any base class inheriting from Experiment can create a default file as::

            MyExperiment().to_yaml('/dir/to/defaults/root')

        """
        self.update_version()
        self.update_meta()
        if not os.path.exists(defaults_dir):
            os.makedirs(defaults_dir)

        if self._status != 'default':
            raise ValueError('An experiment can only be converted to default if it has not been registered')
        else:
            # self.defaults_created = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"
            self._to_yaml(defaults_dir)

    def _to_yaml(self, defaults_dir=None):
        """Save experiment to either a defaults.yml file or a params.yml file depending on its status"""
        import ruamel.yaml
        from ruamel.yaml.comments import CommentedMap

        self.update_version()
        params = {k: getattr(self, k) for k in dir(self) if k in self.param_keys() or k in self._specials}
        params = {k: v for k, v in params.items() if '_' + k not in self.__dict__}
        helps = self.get_param_helps()

        # Add definition module to experiment object
        defaults = CommentedMap()
        for i, (k, v) in enumerate(params.items()):
            if self._status == 'default':
                if k in ['_root', '_name', '_status', '_purpose', '_messages', '_origin']:
                    continue
            # comment = helps[k].split(k)[1] if helps[k] is not None else None
            comment = helps[k]
            if comment == '':
                comment = None
            defaults.insert(i, k, v, comment=comment)

        if self._status == 'default':
            path = os.path.join(os.path.join(defaults_dir, 'defaults.yml'))
        else:
            path = os.path.join(self._root, self._name, 'params.yml')

        # Convert to yaml
        yaml = ruamel.yaml.YAML()
        yaml.register_class(NullRoot)
        try:
            if self.status == 'detached':
                pass
            yaml.dump(defaults, io.StringIO())
        except yaml.representer.RepresenterError as m:
            raise RuntimeError(f'Invalid yaml encountered with error {m}')

        with open(path, 'w') as file:
            yaml.dump(defaults, file)

    def debug(self):
        """Inherited classes may overload debug. Used to define a set of setup for minimum example"""
        return self

    def from_yml(self, path, copy=False):
        """Load state from either a ``params.yml`` or ``defaults.yml`` file (inferred from the filename).
        The status of the experiment will be updated to ``'default'`` if ``'defaults.yml'``
        file else ``'registered'`` if ``params.yml`` file."""
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        try:
            with open(path, 'r') as file:
                params = yaml.load(file)
        except:
            raise IncompatibleYmlException
        params = {k: commented_to_py(v) for k, v in params.items() if k in self.__dict__}
        if copy:
            # Copy only parameter values themselves (and not specials)
            params = {k: v for k, v in params.items() if not k.startswith('_')}
        # Update created date
        self.__dict__.update(params)
        self._created = datetime.datetime.now().strftime(DATE_FORMAT)

    def register(self, root, name, purpose='', force=True, same_names=100, generate_script=False, restart=False):
        """Register an experiment to an experiment directory. Its status will be updated to ``registered``. If an
        experiment called ``name`` exists in ``root`` and ``force==True`` then name will be appended with an int
        (eg. ``{name}_0``) until a unique name is found in ``root``. If ``force==False`` a ``ValueError`` will be raised.

        If restart is also passed, and an experiment called name also exists, then the experiment will be loaded
        from the params.yml file found in ``'{root}/{name}'``.

        Raises:
            ValueError: if ``{root}/{name}`` already contains a ``params.yml`` file
        """
        folder = os.path.join(root, name)
        exists = os.path.exists(os.path.join(folder, 'params.yml'))
        if not restart or (restart and not exists):
            if exists:
                i = 0
                if force:
                    while i < same_names:
                        if not os.path.exists(os.path.join(folder + '_' + str(i), 'params.yml')):
                            folder += '_' + str(i)
                            name += '_' + str(i)
                            break
                        i += 1
                elif i == same_names or not force:
                    raise ValueError(f'Experiment folder {os.path.join(root, name)} already contains a params.yml file. '
                                     f'An Experiment cannot be created in an already existing experiment folder')
            # Make the folder if it does not exist
            if not os.path.isdir(folder):
                os.makedirs(folder)

            # Generate a script.sh in each folder that can be used to run the experiment
            if generate_script:
                script = self.get_run_script(type='individual')
                with open(os.path.join(self.root, self.name, 'run.sh'), 'w') as f:
                    f.write(script)

            self.update_version()  # Get new version information
            self.update_meta()   # Get the newest meta information
            self._root = root
            self._name = name
            self._purpose = purpose
            self._status = 'registered'
            self._to_yaml()
        else:
            self.from_yml(os.path.join(folder, 'params.yml'))
            if self.status != 'registered':
                self._status = 'registered'
                self._to_yaml()

    def get_run_script(self, type='set', shell='/usr/bin/env python3', comment='#'):
        assert type in ['set', 'indiviual'], 'Only types "set" and "individual" are currently supported.'
        sh = [f'#!{shell}']
        self.update_version()
        sh += [f'# File generated on the {datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")}']
        if 'git' in self._version:
            sh += [f'{comment} GIT:']
            sh += [f'{comment} - repo {self._version["git"]["local"]}']
            sh += [f'{comment} - branch {self._version["git"]["branch"]}']
            sh += [f'{comment} - remote {self._version["git"]["remote"]}']
            sh += [f'{comment} - commit {self._version["git"]["commit"]}']
            sh += ['']
        possible_roots = sorted([p for p in sys.path if p in self._version['module']])
        root = None
        if len(possible_roots) > 0:
            root = possible_roots[0]

        if 'python' in shell:
            sh += ['import sys']
            sh += ['import importlib']
            if root is not None:
                sh += [f'sys.path.append("{root}")']
            sh += ['import logging']
            sh += ['logger = logging.getLogger()']
            sh += ['logger.setLevel(logging.INFO)']
            sh += ['']
            sh += [f'module = importlib.import_module("{self.__class__.__module__}")']
            sh += [f'X = getattr(module, "{self.__class__.__name__}")']
            sh += ['X().main()']
        else:
            sh = [f'#!{shell}']
            sh += ['python3 ' + self._version['module'] + ' --execute ']
            sh[-1] += '${1}' if type == 'set' else os.path.join(self.root, 'params.yml')
        return '\n'.join(sh)

    def to_root(self, root_dir, shell='/bin/bash'):
        """Generate a ``defaults.yml`` file and ``script.sh`` file in ``root_dir``.

        Args:
            root_dir (str): A path to the root directory in which to generate a script.sh and defaults.yml to
                run the experiment.
        """
        import stat
        # get_git is deliberately called outside to_defaults as git information is also added to script.sh
        self.update_version()
        self.update_meta()

        from xmen.utils import get_run_script
        if hasattr(self, 'fn'):
            script = get_run_script(*self.fn)
        else:
            script = get_run_script(self.__class__.__module__, self.__class__.__name__)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        # Save to root directory
        path = os.path.join(root_dir, f'{self.__class__.__module__}.{self.__class__.__name__}')
        open(path, 'w').write(script)
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)

        open(os.path.join(root_dir, 'script.sh'), 'w').write(
            f'#!{shell}\n{path} --execute ${{1}}')

        self.to_defaults(root_dir)

    def _update_status(self, status):
        """Update the status of the experiment"""
        self._status = status
        self._to_yaml()

    def detach(self):
        self._root = NullRoot()
        self._name = ''
        self._status = 'detached'

    def update(self, kwargs):
        """Update the parameters with a given dictionary"""
        if self._status in ['default', 'detached']:
            if any([k not in self.param_keys() and k in self._specials for k in kwargs]):
                raise ValueError('Key not recognised!')
            else:
                self.__dict__.update(kwargs)
            # Update the created date
            self._created = datetime.datetime.now().strftime("%m-%d-%y-%H:%M:%S")
        else:
            raise ValueError('Parameters of a created experiment cannot be updated.')

    def param_keys(self):
        return self._params.keys()

    def __setattr__(self, key, value):
        """Attributes can only be changed when the status of the experiment is default"""
        # specials = ['name', 'root', 'status', 'created', 'purpose', 'messages', 'version']
        if '_status' in self.__dict__:
            if key in self.param_keys() and self._status not in ['default', 'detached'] and key not in self._specials:
                raise AttributeError('Parameters can only be changed when status = "default" or "detached"')
        self.__dict__.update({key: value})

    def __enter__(self):
        def _sigusr1_handler(signum, handler):
            raise TimeoutException()
        signal.signal(signal.SIGUSR1, _sigusr1_handler)

        self._update_status('running')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._update_status('finished')
        elif exc_type is KeyboardInterrupt:
            self._update_status('stopped')
        elif exc_type is TimeoutException:
            self._update_status('timeout')
        else:
            if self.status not in ['timeout', 'stopped']:
                self._update_status('error')

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
            self._to_yaml()

    @staticmethod
    def convert_type(v):
        if not isinstance(v, (str, float, int)):
            try:
                v = float(v)
            except (ValueError, TypeError):
                v = str(v)
                pass
        return v

    def note(self, txt, rm=False):
        if not rm:
            now = datetime.datetime.now().strftime(DATE_FORMAT)
            self._notes[now] = txt
        else:
            for k in list(self._notes.keys()):
                if txt.strip() == self._notes[k].strip():
                    self._notes.pop(k)
        self._to_yaml()

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
        # Update the parameter values from either
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
                    name = os.path.basename(os.path.normpath(args.execute))
                    root = os.path.dirname(os.path.normpath(args.execute))
                    self.register(root, name, restart=args.restart)
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

        if all(a is None for a in (args.debug, args.execute, args.to_root, args.to_txt, args.update)):
            print(self.__doc__)
            print('\nFor more help use --help.')

        # Generate experiment root
        if args.to_root is not None:
            print(f'Generating experiment root at {args.to_root}')
            self.to_root(args.to_root)

        # Run the experiment
        if args.execute is not None:
            assert self.status in ['registered', 'detached'], 'Experiment must be registered before execution'
            # Configure standard out to print to the registered directory as well as
            # the original standard out
            if args.to_txt:
                self.stdout_to_txt()
            # Execute experiment
            try:
                # print(self)
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

