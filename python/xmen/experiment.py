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
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap
import pandas as pd
import collections
import argparse
from typing import Optional, Dict, List, Any

from xmen.utils import *
from xmen.experiment_manager import GlobalExperimentManager

pd.set_option('expand_frame_repr', False)

experiment_parser = argparse.ArgumentParser(description='Run the Experiment command line interface')
experiment_parser.add_argument('--update', type=str, default=None,
                               help='Update the parameters given by a yaml string. Note this will be called before'
                                    'other flags and can be used in combination with --to_root, --to_defaults,'
                                    'and --register.', metavar='YAML_STRING')
experiment_parser.add_argument('--execute', type=str, default=None, metavar='PARAMS',
                               help='Execute the experiment from the given params.yml file.'
                                    ' Cannot be called with update.')
experiment_parser.add_argument('--to_root', type=str, default=None, metavar='DIR',
                               help='Generate a run script and defaults.yml file for interfacing with the experiment'
                                    ' manager. If the directory does not exist then it is first created.')
experiment_parser.add_argument('--to_defaults', type=str, default=None, metavar='DIR',
                               help='Generate a defaults.yml file from the experiment defaults in the given directory')
experiment_parser.add_argument('--register', type=str, nargs=2, default=None, metavar=('ROOT', 'NAME'),
                               help='Register an experiment at root (1st positional) name (2nd'
                                    'positional)')
experiment_parser.add_argument('--debug', type=bool, default=None, help='Run experiment in debug mode. Experiment is '
                                                                         'registered to a folder in /tmp')
experiment_parser.add_argument('--name', action='store_true', help='Return the name of the experiment class')

_SPECIALS = ['_root', '_name', '_status', '_created', '_purpose', '_messages', '_version', '_meta']


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
    __params = {}   # Used to store parameters registered with Experiment

    def __init__(self, name=None, root=None, purpose=''):
        """Initialise the experiment object. If name and root are not None then the experiment is initialised in
        default mode else it is created at '{root}/{name}'.
        """
        if (name is None) == (root is None):
            now_time = datetime.datetime.now().strftime(DATE_FORMAT)
            self._root: Optional[str] = None  # @p The root directory of the experiment
            self._name: Optional[str] = None  # @p The name of the experiment (under root)
            self._status: str = 'default'     # @p One of ['default' | 'created' | 'running' | 'error' | 'finished']
            self._created: str = now_time     # @p The date the experiment was created
            self._purpose: Optional[str] = None   # @p A description of the experiment purpose
            self._messages: Dict[Any, Any] = {}   # @p Messages left by the experiment
            self._version: Optional[Dict[Any, Any]] = None   # @p Experiment version information. See `get_version`
            self._meta: Optional[Dict] = None    # @p The global configuration for the experiment manager
            self._specials: List[str] = _SPECIALS
            self._helps: Optional[Dict] = None
        else:
            raise ValueError("Either both or neither of name and root can be set")
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

    def register_param(self, k, default, type=None, help=None):
        self._Experiment_params.update(k, (default, type, help))

    def param_keys(self):
        return self._Experiment__params.keys()

    def get_attributes_help(self):
        """Get help for all attributes in class (including inherited and private)."""
        return {k: v[-1].strip() for k, v in self._Experiment__params.items()}

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
        self.update_version()
        params = {k: v for k, v in self.__dict__.items() if k in self.param_keys() or k in self._specials}
        params = {k: v for k, v in params.items() if '_' + k not in self.__dict__}
        helps = self.get_attributes_help()

        # Add definition module to experiment object
        defaults = CommentedMap()
        for i, (k, v) in enumerate(params.items()):
            if self._status == 'default':
                if k in ['_root', '_name', '_status', '_purpose', '_messages']:
                    continue
            comment = helps[k].split(':')[1] if helps[k] is not None else None
            defaults.insert(i, k, v, comment=comment)

        if self._status == 'default':
            path = os.path.join(os.path.join(defaults_dir, 'defaults.yml'))
        else:
            path = os.path.join(self._root, self._name, 'params.yml')

        # Convert to yaml
        yaml = ruamel.yaml.YAML()
        with open(path, 'w') as file:
            yaml.dump(defaults, file)

    def from_yml(self, path):
        """Load state from either a ``params.yml`` or ``defaults.yml`` file (inferred from the filename).
        The status of the experiment will be equal to ``'default'`` if ``'defaults.yml'``
        file else ``'registered'`` if ``params.yml`` file."""
        yaml = ruamel.yaml.YAML()
        with open(path, 'r') as file:
            params = yaml.load(file)
        params = {k: commented_to_py(v) for k, v in params.items() if k in self.__dict__}
        self.__dict__.update(params)
        # Update created date
        self._created = datetime.datetime.now().strftime(DATE_FORMAT)

    def register(self, root, name, purpose='', force=True, same_names=100):
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

        self.update_version()  # Get new version information
        self.update_meta()   # Get the newest meta information
        self._root = root
        self._name = name
        self._purpose = purpose
        self._status = 'registered'
        self._to_yaml()

    def to_root(self, root_dir):
        """Generate a ``defaults.yml`` file and ``script.sh`` file in ``root_dir``.

        Args:
            root_dir (str): A path to the root directory in which to generate a script.sh and defaults.yml to
                run the experiment.
        """
        # get_git is deliberately called outside to_defaults as git information is also added to script.sh
        self.update_version()
        self.update_meta()
        sh = ['#!/bin/bash']
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
        sh += ['python ' + self._version['module'] + ' --execute ${1}']
        print('\n'.join(sh))

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        # Save to root directory
        open(os.path.join(root_dir, 'script.sh'), 'w').write('\n'.join(sh))
        self.to_defaults(root_dir)

    def main(self, args):
        """A main loop used to expose the command line interface. In order to expose the command line
        interface within a module definition the lines::

            import experiment_parser as exp_parser

            ...

            if __name__ == '__main__':
                args = exp_parser.parse_args()
                exp = AnExperiment()   # Initialise
                exp.main(args)

        should be added to the module script (where ``AnExperiment`` inherits from ``Experiment``).

        """
        if args.update is not None and args.execute is not None:
            print('ERROR: parameters cannot be updated when executing an experiment from a params.yml')
            exit()
        sum_check = sum([args.register is not None, args.to_defaults is not None, args.to_root is not None])
        if sum_check > 1:
            print(f'ERROR: Only one of register, to_defaults and root can be set but {sum_check} were set')
            exit()
        elif (sum_check == 1) == (args.execute is not None or args.debug is not None or args.name):  # exclusive or
            print('ERROR: Either one of --register, --to_defaults and --to_root must be passed or --execute, --debug'
                  'and --name must be passed.')
        if args.debug is not None and args.execute is not None:
            print(f'ERROR: Only one of debug and execute can be set')
            exit()

        if args.update is not None:
            overrides = ruamel.yaml.load(args.update, Loader=ruamel.yaml.Loader)
            print(f'Updating parameters {overrides}')
            self.update(overrides)

        if args.to_defaults is not None:
            self.to_defaults(args.to_defaults)
        if args.to_root is not None:
            self.to_root(args.to_root)
        if args.register is not None:
            self.register(args.register[0], args.register[1])

        if args.debug is not None:
            self.register(root='/tmp', name='test')
            print(self.directory)
            self.__call__()

        if args.execute is not None:
            self.from_yml(args.execute)
            self.__call__()

        if args.name is not None:
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

    def __setattr__(self, key, value):
        """Attributes can only be changed when the status of the experiment is default"""
        # specials = ['name', 'root', 'status', 'created', 'purpose', 'messages', 'version']
        if '_status' in self.__dict__:
            if key in self.param_keys() and self._status not in ['default', 'detached'] and key not in self._specials:
                raise AttributeError('Parameters can only be changed when status = "default" or "detached"')
        self.__dict__.update({key: value})

    def __enter__(self):
        self._update_status('running')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._update_status('finished')
        else:
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

    def message(self, message_dict):
        """Add a message to the params.yml file.

        Args:
            message_dict (dict): A dictionary of messages. Keys are interpreted as subjects and values interpreted as
                messages. If the ``defaults.yml`` already contains subject then the message for subject will be
                updated.
        """
        if self._status != 'default':
            self._messages.update(message_dict)
            self._to_yaml()
        else:
            raise ValueError('An experiment must be registered to leave a message')

    def __repr__(self):
        """Provides a useful help message for the experiment"""
        # params = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        helps = self.get_attributes_help()
        # base_keys = ['root', 'name', 'status', 'created', 'purpose', 'messages', 'version']
        base_params = {k[1:]: v for k, v in self.__dict__.items() if k in self._specials}
        params = {k: v for k, v in self.__dict__.items() if k[0] != '_' and k not in self._specials}
        lines = recursive_print_lines(base_params)
        lines += ['parameters:']
        lines += ['  ' + l for l in recursive_print_lines(params, helps)]
        return '\n'.join(lines)

