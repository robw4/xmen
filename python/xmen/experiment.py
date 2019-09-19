"""A module containing the Experiment class definition."""
import sys
import datetime
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap
import pandas as pd
import collections
import argparse

from xmen.utils import *

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


class Experiment(object, metaclass=TypedMeta):
    """A generic experiment type.

    **The Experiment Class**
    Experiments are defined by:

    1. *Parameters*: public attributes to the class definition (eg. ``self.a``, ``self.b`` etc.)
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
             self.a: int = 3     # This is the first parameter
             self.b: str = '4'   # This is the second parameter
             self.c: int = 2     # This is the third parameter

             # Private attributes are not assumed to be parameters. Use these to store
             # state information that vary with program execution.
             self._a = None

       def run(self):
           '''The execution of an experiment is defined by overloading the run method.
           This method should not be called directly. Instead an experiment is run by
           calling it (eg. exp())'''.
           print(self.a)
           print(self.b)

          # You may assume that an experiment is able to write to a folder
          # {self.root}/{self.name}. In fact this is ENCOURAGED!
          with open(self.root + '/' + self.name + '/' + 'log.txt', 'w') as f:
              f.write('experiment finished')

       exp = AnExperiment()     # Experiments are instantiated with 'default' status. They cannot be run!

    The ``status`` of an experiment is used to control what an experiment object is allowed to do:

    * ``'default'``: When initialised (or loaded from a ``defaults.yml`` file) experiments are given a ``'default'``
      status. In this case their parameters can be changed using the ``update()`` method call but they cannot be run.
    * ``'created'``: In order to be executed experiments must first be ``'created'``. In doing so an experiment
      object is linked with a unique experiment repository and its parameters are fixed. An experiment is created
      through the ``register()`` method call (or by loading from a `params.yml` file using the method call
      ``from_params_yml()``). Its parameters are saved in a ``defaults.yml`` file within the experiment folder
      perfectly created at a later data.

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
      call ``to_defaults()``. This allows config files to be generated from code definition. Parameters evolve as code
      evolves: as code is versioned so to are the experiment parameters::

         exp = AnExperiment()
         exp.update({'a': 3, 'b': 'z'})
         exp.to_root('/path/to/root/dir')

    * *Loading experiments configured with the experiment manager*: Each experiment can also be loaded from a
      ``params.yml`` file generated by the experiment manager through the method call ``from_params_yml()``. In doing
      so an experiments status is updated to ``'created'``: instead of creating a ``params.yml`` it uses the
      ``params.yml`` file generated by the ``ExperimentManager``::

         exp = AnExperiment()
         exp.from_params_yml('/path/to/params.yml')    # The status is updated to 'created'
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

       class AnExperiment(Experiment):
              def __init__(self)
                   self.a: int = 3     # A param

                   self._i = 0

              def run(self):
                 for i in range(1000):
                     if i % 3 == 0:
                        self.send_message({'step': f'self._i'})

       if __name__ == '__main__':
            exp = AnExperiment()
            exp.main(sys.argv)

    By including the last two lines we expose the command line interface. As such the module can be run from the
    command line as::

       path/to/an_experiment.py --call /path/to/params.yml
       path/to/an_experiment.py --to_root /path/to/experiment/root

    In the first case an experiment is loaded and run from a ``params.yml`` file. In the second a ``script.sh`` and
    ``defaults.yml`` file is created at /path/to/experiment/root.
    """

    def __init__(self, name=None, root=None, purpose=''):
        """Initialise the experiment object. If name and root are not None then the experiment is initialised in
        default mode else it is created at '{root}/{name}'.

        Parameters:
            root (str): The root directory to which the experiment belongs (should not be set)
            name (str): The name of the experiment (should not be set)
            status (str): The status of the experiment (one of ['default' | 'created' | 'running' | 'error' | 'finished'])
            created (str): The date the last time the parameters of the model were updated.
            purpose(str): The purpose for the experiment (should not be set)
            messages (dict): A dictionary of messages which are able to vary throughout the experiment (should not be set)
            version (dict): A dictonary containing the experiment version information. See `get_version` for more info
        """
        if (name is None) == (root is None):
            self.root = None
            self.name = None
            self.status = 'default'
            self.created = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")
            self.purpose = None
            self.messages = {}
            self.version = None
            self._helps = None
        else:
            raise ValueError("Either both or neither of name and root can be set")
        if name is not None:
            self.register(name, root, purpose)

    def get_attributes_help(self):
        """Get help for all attributes in class (including inherited)."""
        if self._helps is None:
            self._helps = get_attribute_helps(type(self))
        return self._helps

    def update_version(self):
        self.version = get_version(cls=self.__class__)

    def to_defaults(self, defaults_dir):
        """Create a defaults.yml file from a class instantiation. This method can be called independent of the
        experiment status. Any base class inheriting from Experiment can create a default file as::

            MyExperiment(a=3, b=9, c='a').to_yaml('/dir/to/defaults/root')
        """
        self.version = self.update_version()
        # self.git = self.get_git()
        # self.definition = self.get_definition()
        if not os.path.exists(defaults_dir):
            os.makedirs(defaults_dir)

        if self.status != 'default':
            raise ValueError('An experiment can only be converted to default if it has not been created')
        else:
            # self.defaults_created = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"
            self._to_yaml(defaults_dir)

    def _to_yaml(self, defaults_dir=None):
        """Save experiment to either a defaults.yml file or a params.yml file depending on its status"""
        self.update_version()
        # print(self.__class__.__module__)
        # if definition.__name__ == '__main__':
        #     self.definition = sys.modules['__main__']
        #     print(self.definition)
        # else:
        #     self.definition = definition
        # print(self.definition)
        params = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        helps = self.get_attributes_help()

        # Add definition module to experiment object
        defaults = CommentedMap()
        for i, (k, v) in enumerate(params.items()):
            if self.status == 'default':
                if k in ['root', 'name', 'status', 'purpose', 'messages']:
                    continue
            comment = helps[k].split(':')[1] if helps[k] is not None else None
            defaults.insert(i, k, v, comment=comment)

        if self.status == 'default':
            path = os.path.join(os.path.join(defaults_dir, 'defaults.yml'))
        else:
            path = os.path.join(self.root, self.name, 'params.yml')

        # Convert to yaml
        yaml = ruamel.yaml.YAML()
        with open(path, 'w') as file:
            yaml.dump(defaults, file)

    def from_yml(self, path):
        """Load state from either a ``params.yml`` or ``defaults.yml`` file. The status of the experiment will be equal to
        ``'default'`` if ``'defaults.yml'`` file else ``'registered'`` if params file."""
        yaml = ruamel.yaml.YAML()
        with open(path, 'r') as file:
            params = yaml.load(file)
        params = {k: v for k, v in params.items() if k in self.__dict__}
        self.__dict__.update(params)
        self.created = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")

    def register(self, root, name, purpose=''):
        """Register an experiment of the experiment."""
        if os.path.exists(os.path.join(root, name, 'params.yml')):
            raise ValueError(f'Experiment folder {os.path.join(root, name)} already contains a params.yml file. '
                             f'An Exeperiment cannot be created in an already existing experiment folder')
        if not os.path.isdir(os.path.join(root, name)):
            os.makedirs(os.path.join(root, name))
        self.update_version()           # Get new version information
        self.root = root
        self.name = name
        self.status = 'registered'
        self.purpose = purpose
        self._to_yaml()

    def to_root(self, root_dir):
        """Generate a defaults.yml file and script.sh file in the root directory.

        Args:
            root_dir (str): A path to the root directory in which to generate a script.sh and defaults.yml to
                run the experiment.
        """
        # get_git is deliberately called outside to_defaults as git information is also added to script.sh
        self.update_version()
        sh = ['#!/bin/bash']
        sh += [f'# File generated on the {datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")}']
        if 'git' in self.version:
            sh += [f'# GIT:']
            sh += [f'# - repo {self.version["git"]["local"]}']
            sh += [f'# - branch {self.version["git"]["branch"]}']
            sh += [f'# - remote {self.version["git"]["remote"]}']
            sh += [f'# - commit {self.version["git"]["commit"]}']
            sh += ['']

        possible_roots = sorted([p for p in sys.path if p in self.version['module']])
        if len(possible_roots) > 0:
            root = possible_roots[0]
            sh += ['export PYTHONPATH="${PYTHONPATH}:' + f'{root}"']
        sh += ['python ' + self.version['module'] + ' --execute ${1}']
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
        elif (sum_check == 1) == (args.execute is not None):  # exclusive or
            print('ERROR: Either one of --register, --to_defaults and --to_root must be passed or --execute must'
                  'be passed.')

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

        if args.execute is not None:
            self.from_yml(args.execute)
            self.__call__()

    def _update_status(self, status):
        """Update the status of the experiment"""
        self.status = status
        self._to_yaml()

    def update(self, kwargs):
        """Update the parameters with a given dictionary"""
        if self.status == 'default':
            if any([k not in self.__dict__.keys() for k in kwargs]):
                raise ValueError('Key not recognised!')
            else:
                self.__dict__.update(kwargs)
            # Update the created date
            self.created = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")
        else:
            raise ValueError('Parameters of a created experiment cannot be updated.')

    # TODO(rw): Fix fixing of attributes
    # def __setattr__(self, key, value):
    #     """Attributes can only be changed when the status of the experiment is default"""
    #     if key[0] != '_' and self.status != 'default':
    #         raise ValueError('Parameters (public attributes) can only be changed when status = "default"')
    #     else:
    #         self.__dict__.update({key: value})

    def __enter__(self):
        self._update_status('running')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._update_status('finished')
        else:
            self._update_status('error')

    def __call__(self, *args, **kwargs):
        """Used to run experiment. Upon entering the experimetn status is updated to ``'running`` before ``args`` and
        ``kwargs`` are passed to ``run()``. If ``run()`` is successful the experiment ``status`` is updated to
        ``'finished'`` else it will be given ``status='error'``.

        Both *args and **kwargs are passed to self.run. """
        if self.status == 'default':
            raise ValueError('An experiment in default status must be registered before it can be executed')
        with self:
            self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError('Derived classes must implement the run method in order to be called')

    def leave_message(self, message_dict):
        """Add a message to the params.yml file.

        Args:
            message_dict (dict): A dictionary of messages. Keys are interpreted as subjects and values interpreted as
                messages. If the defaults.yml already contains subject then the message for subject will be
                updated.
        """
        if self.status != 'default':
            self.messages.update(message_dict)
            self._to_yaml()
        else:
            raise ValueError('An experiment must be created to leave a message')

    def __repr__(self):
        """Provides a useful help message for the experiment"""
        def _recursive_print_dict(dic, H=None):
            lines = []
            for k, v in dic.items():
                if type(v) is dict or type(v) is collections.OrderedDict or type(v) is CommentedMap:
                    lines += [f'{k}:']
                    lines += ['  ' + l for l in _recursive_print_dict(v)]
                elif v is not None:
                    h = ''
                    if H is not None:
                        if H[k] is not None:
                            h = H[k].split(":")[1].strip()
                        else:
                            h = ''
                    if h != '':
                        lines += [f'{k}: {v}   # {h}']
                    else:
                        lines += [f'{k}: {v}']
            return lines
        # params = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        helps = self.get_attributes_help()
        base_keys = ['root', 'name', 'status', 'created', 'purpose', 'messages', 'version']
        base_params = {k: v for k, v in self.__dict__.items() if k in base_keys}
        params = {k: v for k, v in self.__dict__.items() if k[0] != '_' and k not in base_keys}
        lines = _recursive_print_dict(base_params)
        lines += ['parameters:']
        lines += ['  ' + l for l in _recursive_print_dict(params, helps)]
        return '\n'.join(lines)
