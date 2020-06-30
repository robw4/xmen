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
import os
import datetime
import argparse
from typing import Optional, Dict, List, Any
import signal
import io

from argparse import RawTextHelpFormatter
from xmen.utils import get_meta, get_version, commented_to_py, DATE_FORMAT, recursive_print_lines, TypedMeta, MultiOut


experiment_parser = argparse.ArgumentParser(description='Run the Experiment command line interface',
                                            formatter_class=RawTextHelpFormatter)
experiment_parser.add_argument('--update', type=str, default=None,
                               help='Update the parameters given by a yaml string. Note this will be called before'
                                    'other flags and can be used in combination with --to_root, --to_defaults,'
                                    'and --register.', metavar='YAML_STRING')
experiment_parser.add_argument('--execute', type=str, default=None, metavar='PARAMS',
                               help='Execute the experiment from the given params.yml file.'
                                    ' Cannot be called with update.', nargs='?', const=True)
experiment_parser.add_argument('--to_root', type=str, default=None, metavar='DIR',
                               help='Generate a run script and defaults.yml file for interfacing with the experiment'
                                    ' manager. If the directory does not exist then it is first created.')
experiment_parser.add_argument('--to_defaults', type=str, default=None, metavar='DIR',
                               help='Generate a defaults.yml file from the experiment defaults in the given directory')
experiment_parser.add_argument('--register', type=str, nargs=2, default=None, metavar=('ROOT', 'NAME'),
                               help='Register an experiment at root (1st positional) name (2nd'
                                    'positional)')
experiment_parser.add_argument('--debug', default=None, const='/tmp/test', nargs="?",
                               help='Run experiment in debug mode. Experiment is registered to a folder in /tmp')
experiment_parser.add_argument('--to_txt', default=None,
                               help='Also log stdout and stderr to an out.txt file. Enabled by default')
experiment_parser.add_argument('--name', action='store_true', help='Return the name of the experiment class')


_SPECIALS = ['_root', '_name', '_status', '_created', '_purpose', '_messages', '_version', '_meta']


class TimeoutException(Exception):
    pass


class Experiment(object, metaclass=TypedMeta):
    """A generic experiment type.

    **The Experiment Class**
    Experiments are defined by:

    1. *Parameters*: public attributes to the class definition (eg. ``self.a``, ``self.b`` etc.), decalared with the
        special parameter ``# @p`` in a comment after the definition.
    2. *Execution*: defined by overloading the ``run()`` method

    For example::

       class AnExperiment(Experiment):
          def __init__(self):
             ''' Doc strings should be used to document the purpose of the experiment.
             Parameters are defined as public class attributes. You are encouraged to
             use inline typed parameter definitions allowing the type, default and
             documentation for a parameter to be generated in a single line.
             In this case Attributes documentation will be added to the docstring
             automatically by the TypedMeta metaclass.
             '''
             self.a: int = 3     # @p This is the first parameter
             self.b: str = '4'   # @p This is the second parameter
             self.c: int = 2     # @p This is the third parameter

             # Other attributes are not assumed to be parameters and cun run throughout excution
             self.a = None   # This is not a parameter

       def run(self):
           '''The execution of an experiment is defined by overloading the run method.
           This method should not be called directly. Instead an experiment is run by
           calling it (eg. exp())'''.
           print(self.a)
           print(self.b)

          # You may assume that an experiment is able to write to a folder
          # {self.root}/{self.name}. In fact this is ENCOURAGED!
          with open(self.directory + '/' + 'log.txt', 'w') as f:
              f.write('experiment finished')

       exp = AnExperiment()     # Experiments are instantiated with 'default' status. They cannot be run!

    The ``status`` of an experiment is used to control what an experiment object is allowed to do:

    * ``'default'``: When initialised (or loaded from a ``defaults.yml`` file) experiments are given a ``'default'``
      status. In this case their parameters can be changed using the ``update()`` method call or by
      setting parameters directly (eg. ``self.a = 3``) before saving their state to a ``defaults.yml`` through the
      ``to_defaults()`` method call. They **cannot be executed**.
    * ``'registered'``: In order to be executed experiments must first be ``'registered'``. In doing so an experiment
      object is linked with a unique experiment repository, its parameters fixed, and stored in a ``params.yml file``.
       An experiment is `registered` either through the ``register()`` method call
       or by loading from a previously created``params.yml`` file using the method call
      ``from_yml()``.

    Additionally the following statuses are used to communicate the state of an execution:

    * ``'running'``: Experiments are executed by the ``__call__`` method. Upon entering this method the experiment
      status is changed to ``'running'`` (also updated in the ``params.yml`` file) before the ``run()`` method is
      called.
    * ``'finished'``: If the run executes correctly then the experiment status will be updated to ``'finished'``
    * ``'error'``: If the run returns with an uncaught exception then the status of the experiment will be set to
      ``'error'``.

    **Interfacing with the Experiment Manager**
    The ``Experiment`` class also provides several features in order to interface with the ``ExperimentManager``
    class:

    * *Generating Experiment Roots*: Any experiment with ``'default'`` status can be used to generate a ``defaults.yml``
      file and an execution script (``script.sh``) through the method
      call ``to_root()``. This allows config files to be generated from code definition. Parameters evolve as code
      evolves: as code is versioned so to are the experiment parameters::

         exp = AnExperiment()
         exp.update({'a': 3, 'b': 'z'})
         exp.c = 4    # Direct parameter access is also aloud
         exp.to_root('/path/to/root/dir')

    * *Loading experiments configured with the experiment manager*: Each experiment can also be loaded from a
      ``params.yml`` file generated through the method call ``from_yml()``. In doing
      so an experiments status is updated to ``'registered'``::

         exp = AnExperiment()
         exp.from_params_yml('/path/to/params.yml')    # The status is updated to 'registered'
         exp()                                         # Execute the experiment

    * *Communicating with the experiment manager*: Registered experiment objects are also able to save their dynamic
      state in their ``params.yml`` file by adding to the ``messages`` dictionary. This allows ``Experiment`` objects
      to communicate with the experiment manager through shared storage. This is achieved through the ``send_message``
      method call::

         class AnExperiment(Experiment):
              def __init__(self)
                   self.a: int = 3     # A param

                   self._i = 0

              def run(self):
                 for i in range(1000):
                     if i % 3 == 0:
                        self.send_message({'step': f'self._i'})

    * *Version Control*: As code evolves parameters are added or depreciated and their defaults changed. Config
      files generated for a past experiment are no longer valid for the current version. When ``git`` is available
      each ``defaults.yml`` has appended to it the git commit and local repo it was generated from. Similarly, this
      process is repeated each time an experiment is created (either adding to the existing params.yml file or
      creating a new one). This helps the user keep track of experiments versions automatically.

    **Command Line Interface**
    The class also exposes a basic main loop alongside several command line flags. For example suppose we
    create a module at path ``path/to/an_experiment.py`` inside which lives::

       from xmen.experiment import Experiment, experiment_parser
       class AnExperiment(Experiment):
              def __init__(self)
                   self.a: int = 3     # A param

                   self._i = 0

              def run(self):
                 for i in range(1000):
                     if i % 3 == 0:
                        self.send_message({'step': f'self._i'})

       if __name__ == '__main__':
            args = experiment_parser.parse()
            exp = AnExperiment()
            exp.main(args)

    By including the last two lines we expose the command line interface. As such the module can be run from the
    command line. For available functionality run::

       path/to/an_experiment.py -h

    """
    _params = {}  # Used to store parameters registered by the MetaClass

    def __init__(self, name=None, root=None, purpose='', copy=True, **kwargs, ):
        """Initialise the experiment object. If name and root are not None then the experiment is initialised in
        default mode else it is created at '{root}/{name}'.
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
        if self._status is 'default':
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
        return {k: v[-1].strip() for k, v in self._params.items()}

    def update_version(self):
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
            comment = helps[k].split(':')[1] if helps[k] is not None else None
            defaults.insert(i, k, v, comment=comment)

        if self._status == 'default':
            path = os.path.join(os.path.join(defaults_dir, 'defaults.yml'))
        else:
            path = os.path.join(self._root, self._name, 'params.yml')

        # Convert to yaml
        yaml = ruamel.yaml.YAML()
        try:
            yaml.dump(defaults, io.StringIO())
        except yaml.representer.RepresenterError as m:
            raise RuntimeError(f'Invalid yaml encountered with error {m}')

        with open(path, 'w') as file:
            yaml.dump(defaults, file)

    def debug(self):
        """Inherited classes may overload debug. Used to define a set of setup for minimum example"""
        return self

    def from_yml(self, path):
        """Load state from either a ``params.yml`` or ``defaults.yml`` file (inferred from the filename).
        The status of the experiment will be equal to ``'default'`` if ``'defaults.yml'``
        file else ``'registered'`` if ``params.yml`` file."""
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()

        with open(path, 'r') as file:
            params = yaml.load(file)
        params = {k: commented_to_py(v) for k, v in params.items() if k in self.__dict__}
        # Update created date
        self.__dict__.update(params)
        self._created = datetime.datetime.now().strftime(DATE_FORMAT)

    def register(self, root, name, purpose='', force=True, same_names=100, generate_script=False,
                 header=None):
        """Register an experiment to an experiment directory. Its status will be updated to ``registered``. If an
        experiment called ``name`` exists in ``root`` and ``force==True`` then name will be appended with an int
        (eg. ``{name}_0``) until a unique name is found in ``root``. If ``force==False`` a ``ValueError`` will be raised.

        Raises:
            ValueError: if ``{root}/{name}`` already contains a ``params.yml`` file
        """
        folder = os.path.join(root, name)
        if os.path.exists(os.path.join(folder, 'params.yml')):
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

    def get_run_script(self, type='set', shell='/bin/bash'):
        if type not in ['set', 'indiviual']:
            raise NotImplementedError('Only types "set" and "individual" are currently supported.')
        sh = [f'#!{shell}']
        sh += [f'# File generated on the {datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")}']
        if 'git' in self._version:
            sh += [f'# GIT:']
            sh += [f'# - repo {self._version["git"]["local"]}']
            sh += [f'# - branch {self._version["git"]["branch"]}']
            sh += [f'# - remote {self._version["git"]["remote"]}']
            sh += [f'# - commit {self._version["git"]["commit"]}']
            sh += ['']
        possible_roots = sorted([p for p in sys.path if p in self._version['module']])
        if len(possible_roots) > 0:
            root = possible_roots[0]
            sh += ['export PYTHONPATH="${PYTHONPATH}:' + f'{root}"']
        sh += ['python3 ' + self._version['module'] + ' --execute ']
        sh[-1] += '${1}' if type == 'set' else os.path.join(self.root, 'params.yml')
        return '\n'.join(sh)

    def to_root(self, root_dir):
        """Generate a ``defaults.yml`` file and ``script.sh`` file in ``root_dir``.

        Args:
            root_dir (str): A path to the root directory in which to generate a script.sh and defaults.yml to
                run the experiment.
        """
        # get_git is deliberately called outside to_defaults as git information is also added to script.sh
        self.update_version()
        self.update_meta()

        print(self.get_run_script())

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        script = self.get_run_script()
        # Save to root directory
        open(os.path.join(root_dir, 'script.sh'), 'w').write(script)
        self.to_defaults(root_dir)

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

        if args.debug is not None:
            self.register(*os.path.split(args.debug))
            print(self.directory)
            self.__call__()

        if args.execute is not None:
            if isinstance(args.execute, str):
                self.from_yml(args.execute)
            self.__call__()

        if args.name:
            print(self.__class__.__name__)

    def _update_status(self, status):
        """Update the status of the experiment"""
        self._status = status
        self._to_yaml()

    def detach(self):
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
        """Add a message to the params.yml file.

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
        if self._status != 'default':
            # Add leader to messages group
            if leader is not None:
                best_leader = self.compare(leader, messages[leader], keep)[0]
            else:
                best_leader = False

            for k, v in messages.items():
                best_current, v = self.compare(k, v, keep)
                best = best_leader if leader is not None else best_current
                # Convert type
                if not isinstance(v, (str, float, int)):
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        v = str(v)
                        pass
                # Update message
                if best:
                    self._messages.update({k: v})
            self._to_yaml()
        else:
            raise ValueError('An experiment must be registered to leave a message')

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
        short, params = self.__doc__.split('Parameters:')
        epilog = 'Parameters:' + params
        experiment_parser.description = short
        experiment_parser.epilog = epilog
        n = len(experiment_parser.prog) + 8
        experiment_parser.usage = experiment_parser.prog + '\n'.join(
            [' ' * 1 + '[--update PARAMS] --to_defaults | --to_root [--to_txt]',
             ' ' * n + '[--update PARAMS] --register ROOT NAME [--execute] [--to_txt]',
             ' ' * n + ' --execute PATH_TO_PARAMS',
             ' ',
             'example:',
             'python -m xmen.tests.experiment --update "{a: cat, x: [10., 20.], t: {100: x, 101: y}, h: }" \ ',
             '                                --register /tmp/test/xmen command_line',
             '',
             'Note PARAMS is a yaml dictionary - None is given by the null string. '
             'For a list of valid parameters see below'])
        args = experiment_parser.parse_args()

        if isinstance(args.execute, str):
            assert all(getattr(args, k) is None for k in ('update', 'register', 'to_defaults', 'to_root')), \
             "If execute is a path then none of ('update', 'register', 'to_defaults', 'to_root') can be set"

        assert sum(
            getattr(args, k) is not None for k in ('to_defaults', 'to_root', 'execute')) == 1 or sum(
            getattr(args, k) is not None for k in ('to_defaults', 'to_root', 'register')) == 1, \
            "Exactly one of to_defaults, to_root and register or execute can be set"

        assert sum(getattr(args, k) is not None for k in ('execute', 'debug')) != 2, \
            'A maximum of one of execute and debug can be set at the same time'

        if args.debug is not None:
            # Update debug parameters
            if hasattr(self, 'debug_defaults'):
                print(f'Updating debug parameters {self.debug_defaults}')
                self.update(self.debug_defaults)
        if args.update is not None:
            # Update passed parameters
            import ruamel.yaml
            overrides = ruamel.yaml.load(args.update, Loader=ruamel.yaml.Loader)
            print(f'Updating parameters {overrides}')
            self.update(overrides)

        # Generate experiment repo (one of three types)
        if args.to_defaults is not None:
            print(f'Generating default parameters at {args.to_defaults}')
            self.to_defaults(args.to_defaults)
        if args.to_root is not None:
            print(f'Generating experiment root at {args.to_root}')
            self.to_root(args.to_root)
        if args.register is not None:
            print(f'Registering experiment at {args.register[0]}/{args.register[1]}(...)')
            self.register(args.register[0], args.register[1])

        if args.register is not None and args.to_txt:
            sys.stdout = MultiOut(sys.__stdout__, open(os.path.join(self.directory, 'out.txt'), 'a+'))
            sys.stderr = MultiOut(sys.__stderr__, open(os.path.join(self.directory, 'out.txt'), 'a+'))

        return args

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

