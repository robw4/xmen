#!/usr/bin/env python3
"""A module holding the ExperimentManager implementation. The module can also be run activating the
experiment managers command line interface"""

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
from shutil import copyfile, rmtree
import datetime
import ruamel.yaml
import pandas as pd
import subprocess
import glob
import time
import importlib.util
import argparse
import copy
from xmen.utils import *
pd.set_option('expand_frame_repr', False)

from xmen.experiment import Experiment


def _init(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.initialise(defaults=args.defaults, script=args.script, purpose=args.purpose, name=args.name)


def _register(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.register(args.updates, args.purpose, args.header)


def _reset(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.reset(args.experiments)


def _list(args):
    if not args.all and not args.config:
        experiment_manager = ExperimentManager(args.root)
        experiment_manager.list()
    if args.config:
        print(Config())
    if args.all:
        Config().list()


def _run(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.run(args.experiments, *args.append)


def _unlink(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.unlink(args.experiments)


def _relink(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.relink(args.experiments)


def _clean(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.clean()


def _note(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.note(args.message, args.delete)

def _rm(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.rm()

def _config(args):
    with Config() as config:
        if args.disable_prompt is not None:
            config.prompt_for_message = False
        elif args.enable_prompt is not None:
            config.prompt_for_message = True
        if args.add is not None:
            config.add_class(args.add)
        if args.add_path is not None:
            if args.add_path not in config.python_paths:
                config.python_paths.append(os.path.abspath(args.add_path))
        if args.header is not None:
            if os.path.exists(args.header):
                config.header = open(args.header, 'r').read()
            else:
                config.header = args.header
        if args.remove is not None:
            if args.remove in config.python_paths:
                config.python_paths.remove(args.remove)
            if args.remove in config.python_experiments:
                config.python_experiments.pop(args.remove)


parser = argparse.ArgumentParser(prog='xmen',
                                 description='A helper module for the quick setup and management of experiments')
subparsers = parser.add_subparsers()

# Config
config_parser = subparsers.add_parser('config')
config_parser.add_argument('--disable_prompt', action='store_false',
                           help='Turn purpose prompting off', default=None)
config_parser.add_argument('--enable_prompt', action='store_false',
                           help='Turn purpose prompting on', default=None)
config_parser.add_argument('--add_path', type=str, default=None, help='Add pythonpath to the global config')
config_parser.add_argument('--add', default=None, metavar='PATH',
                            help='Add an Experiment api python script (it must already be on PYTHONPATH)')
config_parser.add_argument('-r', '--remove', default=None, help='Remove a python path or experiment (passed by Name) '
                                                                'from the config.')
config_parser.add_argument('-H', '--header', type=str, help='Update the default header used when generating experiments')
config_parser.set_defaults(func=_config)

# Note parser
note_parser = subparsers.add_parser('note')
note_parser.add_argument('message', help='A note to add to the experiment manager')
note_parser.add_argument('-r', '--root', metavar='DIR', default='',
                         help='Path to the root experiment folder. If None then the current work directory will be '
                              'used')
note_parser.add_argument('-d', '--delete', default='', action='store_true',
                         help='Delete the note corresponding to message.')
note_parser.set_defaults(func=_note)


# Init
init_parser = subparsers.add_parser('init', help='Initialise an experiment.')
init_parser.add_argument('-d', '--defaults', metavar='PATH', default='',
                         help='Path to defaults.yml file. If None then a defaults.yml will be looked for in the current'
                              'work directory.')
init_parser.add_argument('-s', '--script', metavar='PATH', default='',
                         help="Path to a script.sh file. If None a script.sh will be searched for in the current "
                              "work directory.")
init_parser.add_argument('-r', '--root', metavar='DIR', default='',
                         help='Path to the root experiment folder. If None then the current work directory will be '
                              'used')
init_parser.add_argument('-n', '--name', metavar='NAME', default=None,
                         help='A name of a python experiment registered with the global configuration.')
init_parser.add_argument('--purpose', metavar='PURPOSE', default='',
                         help='A string giving the purpose of the experiment set (only used if message prompting is disabled).')
init_parser.set_defaults(func=_init)

# Register
register_parser = subparsers.add_parser('register', help='Register a set of experiments.')
register_parser.add_argument('updates', metavar='YAML_STR',
                             help='Defaults to update to register experiments passes as a yaml dict. The special character'
                                  '"|" is interpreted as an or operator. all combinations of parameters appearing '
                                  'either side of "|" will be registered.')
register_parser.add_argument('-H', '--header', metavar='PATH', help='A header file to prepend to each run script', default=None)
register_parser.add_argument('-p', '--purpose', metavar='STR', help='A string giving the purpose of the experiment.')
register_parser.add_argument('-r', '--root', metavar='DIR', default='',
                             help='Path to the root experiment folder. If None then the current work directory will be '
                                  'used')
register_parser.set_defaults(func=_register)
# List

list_parser = subparsers.add_parser('list', help='List all experiments to screen')
list_parser.add_argument('-r', '--root', metavar='DIR', default='',
                         help='Path to the root experiment folder. If None then the current work directory will be '
                              'used')
list_parser.add_argument('-a', '--all', action='store_true',
                         help='List all experiments roots initialised globally')
list_parser.add_argument('--config', action='store_true', help='List the global configuration.')
list_parser.set_defaults(func=_list)

# Run
run_parser = subparsers.add_parser('run', help='Run a set of experiments')
run_parser.add_argument('experiments', metavar='NAMES', help='A unix glob giving the experiments whos status '
                                                             'should be updated (relative to experiment manager root)')
run_parser.add_argument('append', metavar='FLAG', nargs=argparse.REMAINDER,
                        help='A set of run command options to prepend to the run.sh for each experiment '
                             '(eg. "sh", "srun", "sbatch" etc.)')
run_parser.add_argument('-r', '--root', metavar='DIR', default='',
                         help='Path to the root experiment folder. If None then the current work directory will be '
                              'used')
run_parser.set_defaults(func=_run)

# Reset
status_parser = subparsers.add_parser('reset', help='Reset an experiment to registered status')
status_parser.add_argument('experiments', metavar='NAME', help='A unix glob giving the experiments whos status '
                                                               'should be updated (relative to experiment manager root)')
status_parser.add_argument('-r', '--root', metavar='DIR', default='',
                           help='Path to the root experiment folder. If None then the current work directory will be '
                                'used')
status_parser.set_defaults(func=_reset)

# Clean
clean_parser = subparsers.add_parser('clean', help='(DESTRUCTIVE) Remove experiments no longer managed by the experiment manager')
clean_parser.add_argument('-r', '--root', metavar='DIR', default='',
                          help='Path to the root experiment folder. If None then the current work directory will be '
                               'used')
clean_parser.set_defaults(func=_clean)

# Removes
remove_parser = subparsers.add_parser('rm', help='(DESTRUCTIVE) Remove experiment manager')
remove_parser.add_argument('root', metavar='ROOT_DIR',
                            help='Path to the root experiment folder to be removed.')
remove_parser.set_defaults(func=_rm)

# Unlink
unlink_parser = subparsers.add_parser('unlink', help='Unlink experiments from the experiment manager')
unlink_parser.add_argument('experiments', metavar='NAMES', help='A unix glob giving the experiments to be unlinked')
unlink_parser.add_argument('-r', '--root', metavar='DIR', default='',
                           help='Path to the root experiment folder. If None then the current work directory will be '
                                'used')
unlink_parser.set_defaults(func=_unlink)

# Relink
relink_parser = subparsers.add_parser('relink', help='Relink experiments to the experiment manager')
relink_parser.add_argument('experiments', metavar='NAMES', help='A unix glob giving the experiments to be relinked '
                                                                '(relative to experiment manager root)')
relink_parser.add_argument('-r', '--root', metavar='DIR', default='',
                           help='Path to the root experiment folder. If None then the current work directory will be '
                                'used')
relink_parser.set_defaults(func=_relink)


class Config(object):
    """A helper class used to manage global configuration of the Experiment Manager"""
    def __init__(self):
        self.python_experiments = {}   # A dictionary of paths to python modules compatible with the experiment api
        self.python_paths = []         # A list of python paths needed to run each module
        self.prompt_for_message = True
        self.experiments = []          # A list of all experiments registered with an Experiment Manager
        self.header = ''
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

        params = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        with open(os.path.join(self._dir, 'config.yml'), 'w') as file:
            ruamel.yaml.dump(params, file, Dumper=ruamel.yaml.RoundTripDumper)

    def _from_yml(self):
        """Load the experiment config from an ``config.yml`` file"""
        with open(os.path.join(self._dir, 'config.yml'), 'r') as file:
            params = ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)
            for k, v in params.items():
                self.__dict__[k] = v

    def __str__(self):
        string = f'Prompt for Message: {self.prompt_for_message}\n'
        string += f'Python Path:\n'
        for p in self.python_paths:
            string += f'  - {p}\n'
        string += 'Python Experiments:\n'
        for k, v in self.python_experiments.items():
           string += f'  - {k}: {v}\n'
        string += 'Header:\n'
        string += self.header + '\n'
        string += 'Experiment Roots:\n'
        for e in self.experiments:
            string += f'  - {e}\n'
        return string

    def list(self):
        """List all experiments currently created with the experiment manager."""
        if self.experiments != []:
            table = {'purpose': [], 'created': [], 'root': [], 'class': [], 'commit': []}
            for root in self.experiments:
                em = ExperimentManager(root)
                table['purpose'] += [em.purpose]
                table['created'] += [em.created]
                table['root'] += [em.root]
                defaults = em.load_defaults()
                if '_version' in defaults:
                    version = defaults["_version"]
                    if 'class' in version:
                        table['class'] += [version['class']]
                    if 'git' in version:
                        table['commit'] += [version['git']['commit']]
            print(pd.DataFrame(table))

    def add_class(self, path):
        path = os.path.abspath(path)
        for p in self.python_paths:
            if p not in sys.path:
                sys.path.append(p)
        sys_paths = [p for p in sys.path if p in path]
        if len(sys_paths) == 0:
            print('ERROR: The module has not been added to the PYTHONPATH. Please add!')
        else:
            try:
                name = subprocess.check_output(['python', path, '--name'])
            except subprocess.CalledProcessError:
                exit()
            self.python_experiments.update({name.decode("utf-8").replace('\n', ''): path})
            self._to_yml()


class ExperimentManager(object):
    """A helper class with wrapped command line interface used to manage a set of experiments. It is compatible both
    with experiments generated by the ``Experiment.to_root`` method call as well as any experiment that can be
    represented as a bash `script.sh` which takes a set of parameters in a yaml file as input.

    Command Line Interface:

        * ``init`` - Initialise an experiment from a set of hyper parameters that can be used to specialise a run script.sh::

            exp init 'path/to/defaults.yml' 'path/to/script.sh'
            exp init
            #(see method initialise)

        * ``register`` - Register a set of experiments by overloading the default parameters::

            exp register '{a: 1., b: 2.}' 'initial_experiment'
            exp register '{a: 1 | 2, b: 2., c: a| b| c, d: , e: [a, b, c], f: {a: 1., b: 2., e:5}}' 'another_experiment'
            # (see method register)

        * ``list`` - List all experiments::

            exp list
            # (see method list)

          An example output will look something like::

                 overrides       purpose                  created      status messages                                    commit
            0     a:3__b:4  initial test  07:17PM September 13, 2019  created       {}  dda2f262db819e14900c78d807b2182bf6111aef
            1     a:3__b:5  initial test  07:17PM September 13, 2019  created       {}  dda2f262db819e14900c78d807b2182bf6111aef
            2     a:4__b:4  initial test  07:17PM September 13, 2019  created       {}  dda2f262db819e14900c78d807b2182bf6111aef
            3     a:4__b:5  initial test  07:17PM September 13, 2019  created       {}  dda2f262db819e14900c78d807b2182bf6111aef

            Defaults:
            - created 06:34PM September 13, 2019
            - from  NewNewExperiment at /Users/robweston/projects/rad2sim2rad/python/rad2sim2rad/prototypes/test_typed_experiment.py
            - git local /Users/robweston/projects/rad2sim2rad
            - commit 51ad1eae73a2082e7636056fcd06e112d3fbca9c
            - remote ssh://git@mrgbuild.robots.ox.ac.uk:7999/~robw/rad2sim2rad.git

        * ``unlink`` - Unlink all experiments from the experiment manager matching name string::

            exp unlink 'a:1__b:2'
            exp unlink '*'
            exp unlink 'a:1__*'
            #(see method unlink)

        * ``clean`` - Remove folders of experiments that are no longer managed by the experiment manager::

            exp clean
            #(see method clean)

        * ``run`` - Run a group of experiments matching pattern::

            exp run '*' 'sh'                                        # Run all
            exp run 'a:1__b:2' '[sbatch', --gres, gpu:1]'           # yaml list of strings
            #(see method run)

    More Info:
        At its core the experiment manager maintains a single `root` directory::

            root
            ├── defaults.yml
            ├── experiment.yml
            ├── script.sh
            ├── {param}:{value}__{param}:{value}
            │   └── params.yml
            ├── {param}:{value}__{param}:{value}
            │   └── params.yml
            ...

        In the above we have:

            * ``defaults.yml`` defines a set of parameter keys and there default values shared by each experiment. It
              generated from the ``Experiment`` class it will look like::

                  # Optional additional Meta parameters
                  _created: 06:58PM September 16, 2019    # The date the defaults were created
                  _version:
                      module: /path/to/module/experiment/that/generated/defaults/was/defined/in
                      class: TheNameOfTheExperiment
                      git:
                         local: /path/to/git/repo/defaults/were/defined/in
                         branch: some_branch
                         remote: /remote/repo/url
                         commit: 80dcfd98e6c3c17e1bafa72ee56744d4a6e30e80    # The git commit defaults were generatd at

                  # Default parameters
                  a: 3 #  This is the first parameter (default=3)
                  b: '4' #  This is the second parameter (default='4')

               The ``ExperimentManager`` is also compatible with generic experiments. In this the ``_version`` meta
               field can be added manually, replacing ``module`` and ``class`` with ``path``. The git information for
               each will be updated automatically provided ``path`` is within a git repository.

            * ``script.sh`` is a bash script. When run it takes a single argument ``'params.yml'`` (eg. ```script.sh params.yml```).

                .. note ::

                    ``Experiment`` objects are able to automatically generate script.sh files that look like this::

                        #!/bin/bash
                        # File generated on the 06:34PM September 13, 2019
                        # GIT:
                        # - repo /path/to/project/module/
                        # - remote {/path/to/project/module/url}
                        # - commit 51ad1eae73a2082e7636056fcd06e112d3fbca9c

                        export PYTHONPATH="${PYTHONPATH}:path/to/project"
                        python /path/to/project/module/experiment.py --execute ${1}

              Generic experiments are compatible with the ``ExperimentManager`` provided they can be executed with a
              shell script. For example a bash only experiment might have a ``script.sh`` script that looks like::

                   #!/bin/bash
                   echo "$(cat ${1})"

            * A set of experiment folders representing individual experiments within which each experiment has a
              ``params.yml`` with a set of override parameters changed from the original defaults. These overrides define the
              unique name of each experiment (in the case that multiple experiments share the same overrides each experiment
              folder is additionally numbered after the first instantiation). Additionally, each ``params.yml`` contains the
              following::

                    # Parameters special to params.yml
                    _root: /path/to/root  #  The root directory to which the experiment belongs (should not be set)
                    _name: a:10__b:3 #  The name of the experiment (should not be set)
                    _status: registered #  The status of the experiment (one of ['registered' | 'running' | 'error' | 'finished'])
                    _created: 07:41PM September 16, 2019 #  The date the experiment was created (should not be set)
                    _purpose: this is an experiment example #  The purpose for the experiment (should not be set)
                    _messages: {} #  A dictionary of messages which are able to vary throughout the experiment (should not be set)

                    # git information is updated at registration if ``_version['module']`` or `_version['path']``
                    # exists in the defaults.yml file and the path is to a valid git repo.
                    _version:
                        module: /path/to/module   # Path to module where experiment was generated
                        class: NameOfExperimentClass     # Name of experiment class params are compatible with
                        git: #  A dictionary containing the git history corresponding to the defaults.yml file. Only
                          local_path: path/to/git/repo/params/were/defined/in
                          remote_url: /remote/repo/url
                          hash: 80dcfd98e6c3c17e1bafa72ee56744d4a6e30e80
                                        # Parameters from the default (with values overridden)
                    a: 3 #  This is the first parameter (default=3)
                    b: '4' #  This is the second parameter (default='4')

            * ``experiment.yml`` preserves the experiment state with the following entries::

                    root: /path/to/root
                    defaults: /path/to/root/defaults.yml
                    script: /path/to/root/script.sh
                    experiments:
                    - /private/tmp/new-test/a:10__b:3
                    - /private/tmp/new-test/a:20__b:3
                    - /private/tmp/new-test/a:10__b:3_1
                    overides:
                    - a: 10
                      b: '3'
                    - a: 20
                      b: '3'
                    - a: 10
                      b: '3'
                    created: 07:41PM September 16, 2019   # The date the experiment manager was initialised

        The ``ExperimentManager`` provides the following public interface for managing experiments:

            * ``__init__(root)``:
                Link the experiment manager with a root directory and load the experiments.yml if it exists
            * ``initialise(script, defaults)``:
                Initialise an experiment set with a given script and default parameters
            * ``register(string_pattern)``:
                Register a number of experiments overriding parameters based on the particular ``string_pattern``
            * ``list()``:
                 Print all the experiments and their associated information
            * ``unlink(pattern)``:
                 Relieve the experiment manager of responsibility for all experiment names matching pattern
            * ``clean()``:
                 Delete any experiments which are no longer the responsibility of the experiment manager
            * ``run(string, options)``:
                 Run an experiment or all experiments (if string is ``'all'``) with options prepended.

        Example::

            experiment_manager = ExperimentManager(ROOT_PATH)   # Create experiment set in ROOT_PATH
            experiment_manager.initialise(PATH_TO_SCRIPT, PATH_TO_DEFAULTS)
            experiment_manager.register('parama: 1, paramb: [x, y]')   # Register a set of experiments
            experiment_manger.unlink('parama_1__paramb_y')                    # Remove an experiment
            experiment_manager.clean()                                    # Get rid of any experiments no longer managed
            experiment_run('parama:1__paramb:x', sh)                      # Run an experiment
            experiment_run('all')                                         # Run all created experiments
        """

    def __init__(self, root=""):
        """Link an experiment manager to root. If root already contains an ``experiment.yml`` then it is loaded.

        In order to link a new experiment with a defaults.yml and script.sh file then the initialise method must be
        called.

        Args:
            root: The root directory within which to create the experiment. If "" then the current working directory is
                used. If the root directory does not exist it will be made.

        Parameters:
            root: The root directory of the experiment manger
            defaults: A path to the defaults.yml file. Will be None for a fresh experiment manager (if experiments.yml
                has just been created).
            script: A path to the script.sh file. Will be None for a fresh experiment manager (if experiments.yml
                has just been created).
            # created: A string giving the date-time the experiment was created
            experiments: A list of paths to the experiments managed by the experiment manager
            overides: A list of dictionaries giving the names (keys) and values of the parameters overridden from the
                defaults for each experiment in experiments.
        """
        self.root = os.getcwd() if root == "" else os.path.abspath(root)
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.defaults = None
        self.script = None
        self.experiments = []
        self.overides = []
        self.created = None
        self.purpose = None
        self.notes = []
        self._specials = ['_root', '_name', '_status', '_created', '_purpose', '_messages', '_version']
        self._config = Config()

        # Load dir from yaml
        if os.path.exists(os.path.join(self.root, 'experiment.yml')):
            self._from_yml()

    def check_initialised(self):
        """Make sure that ``'experiment.yml'``, ``'script.sh'``, ``'defaults.yml'`` all exist in the directory"""
        all_exist = all(
            [os.path.exists(os.path.join(self.root, s)) for s in ['experiment.yml', 'script.sh', 'defaults.yml']])
        if not all_exist:
            print('The current work directory is not a valid experiment folder. It is either missing one of'
                             '"script.sh", "experiment.yml" or "defaults.yml" or the experiment.yml file is not valid')
            exit()

    def load_defaults(self):
        """Load the ``defaults.yml`` file into a dictionary"""
        with open(os.path.join(self.root, 'defaults.yml'), 'r') as file:
            defaults = ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)
        return defaults

    def save_params(self, params, experiment_name):
        """Save a dictionary of parameters at ``{root}/{experiment_name}/params.yml``

        Args:
            params (dict): A dictionary of parameters to be saved. Can also be a CommentedMap from ruamel
            experiment_name (str): The name of the experiment
        """
        experiment_path = os.path.join(self.root, experiment_name)
        with open(os.path.join(experiment_path, 'params.yml'), 'w') as out:
            yaml = ruamel.yaml.YAML()
            yaml.dump(params, out)

    def load_params(self, experiment_path, experiment_name=False):
        """Load parameters for an experiment. If ``experiment_name`` is True then experiment_path is assumed to be a
        path to the folder of the experiment else it is assumed to be a path to the ``params.yml`` file."""
        if experiment_name:
            experiment_path = os.path.join(self.root, experiment_path)
        with open(os.path.join(experiment_path, 'params.yml'), 'r') as params_yml:
            params = ruamel.yaml.load(params_yml, ruamel.yaml.RoundTripLoader)
        return params

    def _to_yml(self):
        """Save the current experiment manager to an ``experiment.yaml``"""
        params = {k: v for k, v in self.__dict__.items() if k[0] != '_' or k in self._specials}
        with open(os.path.join(self.root, 'experiment.yml'), 'w') as file:
            ruamel.yaml.dump(params, file, Dumper=ruamel.yaml.RoundTripDumper)

    def _from_yml(self):
        """Load an experiment manager from an ``experiment.yml`` file"""
        with open(os.path.join(self.root, 'experiment.yml'), 'r') as file:
            params = ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)
            self.root = params['root']
            self.defaults = params['defaults']
            self.script = params['script']
            self.created = params['created']
            self.experiments = params['experiments']
            self.overides = params['overides']
            try:
                self.purpose = params['purpose']
            except KeyError:
                pass
            try:
                self.notes = params['notes']
            except KeyError:
                pass

    def initialise(self, *, defaults="", script="", purpose="", name=None):
        """Link an experiment manager with a ``defaults.yml`` file and ``sript.sh``.

        Args:
            defaults (str): A path to a ``defaults.yml``. If "" then a ``defaults.yml`` is searched for in the current
                work directory.
            script (str): A path to a ``script.sh``. If ``""`` then a script.sh file is searched for in the current work
                directory.
        """
        print(name)
        if name is None:
            # Load defaults
            self.defaults = os.path.join(self.root, 'defaults.yml') if defaults == "" else os.path.abspath(defaults)
            print(f'Defaults from {self.defaults}')
            if os.path.exists(self.defaults):
                if defaults != "":
                    copyfile(self.defaults, os.path.join(self.root, 'defaults.yml'))
                    self.defaults = os.path.join(self.root, 'defaults.yml')
            else:
                raise ValueError(f"No defaults.yml file exists in {self.root}. Either use the root argument to copy "
                                 f"a default file from another location or add a 'defaults.yml' to the root directory"
                                 f"manually.")

            # Load script file
            self.script = os.path.abspath(os.path.join(self.root, 'script.sh')) if script == "" else os.path.abspath(script)
            print(f'Script from {self.script}')
            if os.path.exists(self.script):
                if script != "":
                    copyfile(self.script, os.path.join(self.root, 'script.sh'))
                    self.script = os.path.join(self.root, 'script.sh')
            else:
                raise ValueError(f"File {self.script} does not exist. Either use the script argument to copy "
                                 f"a script file from another location or add a 'script.sh' to the root directory"
                                 f"manually.")
            self.script = os.path.join(self.root, 'script.sh')
        else:
            if name not in self._config.python_experiments:
                print(f'Python Experiment {name} has not been registered with the global configuration. Aborting!')
                return

            # Add experiments to python path if they are not there already
            for p in self._config.python_paths:
                if p not in sys.path:
                    sys.path.append(p)
            subprocess.call(['python', self._config.python_experiments[name], '--to_root', self.root])
            self.script = os.path.join(self.root, 'script.sh')
            self.defaults = os.path.join(self.root, 'defaults.yml')

        # Meta Information
        self.created = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")

        # Save state to yml
        if os.path.exists(os.path.join(self.root, 'experiment.yml')):
            print(f"There already exists a experiment.yml file in the root directory {self.root}. "
                  f"To reinitialise an experiment folder remove the experiment.yml.")
            exit()
        print(f'Experiment root created at {self.root}')

        # Add experiment to global config
        with self._config:
            self._config.experiments.append(self.root)

        # Add purpose message
        if self._config.prompt_for_message:
            purpose = input('\nPlease enter the purpose of the experiments: ')

        self.purpose = purpose
        self._to_yml()

    def _generate_params_from_string_params(self, x):
        """Take as input a dictionary and convert the dictionary to a list of keys and a list of list of values
        len(values) = number of parameters specified whilst len(values[i]) = len(keys).
        """
        values = [[]]  # List of lists. Each inner list is of length keys
        keys = []

        for k, v in x.items():
            if type(v) is str:
                if '|' in v:
                    v = v.split('|')
                    v = [ruamel.yaml.load(e, Loader=ruamel.yaml.Loader) for e in v]
                else:
                    v = [v]
            else:
                v = [v]
            keys += [k]

            new_values = []
            # Generate values
            for val in values:  # val has d_type list
                for vv in v:  # vv has d_type string
                    # print(val, vv)
                    new_values += [val + [vv]]
            values = new_values
        return values, keys

    def note(self, msg, remove=False):
        """Add a note to the epxeriment manager. If remove is True msg is deleted instead."""
        self.check_initialised()
        if not remove:
            self.notes += [msg.strip()]
        else:
            self.notes = [n for n in self.notes if msg.strip() != n]
        self._to_yml()

    def register(self, string_params, purpose, header=None, shell='/bin/bash'):
        """Register a set of experiments with the experiment manager.

        Experiments are created by passing a yaml dictionary string of parameters to overload in the ``params.yml``
        file. The special symbol ``'|'`` can be thought of as an or operator. When encountered each of the parameters
        either side ``'|'`` will be created separately with all other parameter combinations.

        Args:
            string_params (str): A yaml dictionary of parameters to override of the form
                ``'{p1: val11 | val12, p2: val2, p3: val2 | p4: val31 | val32 | val33, ...}'``.
                The type of each parameter is inferred from its value in defaults. A ValueError will be raised if any
                of the parameter cannot be found in defaults. Parameters can be float (1.), int (1), str (a), None, and
                dictionaries {a: 1., 2.} or lists [1, 2] of these types. None parameters are specified using empty space.
                The length of list parameters must match the length of the parameter in default.
                Dictionary parameters may only be partially defined. Missing keys will be assumed
                to take there default value.

                The special character '|' is used as an or operator. All combinations of parameters
                either side of an | operator will be created as separate experiments. In the example above
                ``N = 2 * 2 * 3 = 12`` experiments will be generated representing all the possible values for
                parameters ``p1``, ``p3`` and ``p4`` can take with ``p2`` set to ``val2`` for all.
            purpose (str): An optional purpose message for the experiment.
            header (str): An optional header message prepended to each run script.sh

        .. note ::

            This function is currently only able to register list or dictionary parameters at the first level.
            ``{a: {a: 1.. b: 2.} | {a: 2.. b: 2.}}`` works creating two experiments with over-ridden dicts in each case
            but ``{a: {a: 1. | 2.,  b:2.}}`` will fail.

            The type of each override is inferred from the type contained in the defaults.yml file (ints will be cast
            to floats etc.) where possible. This is not the case when there is an optional parameter that can take a
            None value. If None (null yaml) is passed as a value it will not be cast. If a default.yml entry is
            given the value null the type of any overrides in this case will be inferred from the yaml string.
        """
        # TODO: This function is currently able to register or arguments only at the first level
        self.check_initialised()
        defaults = self.load_defaults()

        # Convert input string to dictionary
        p = ruamel.yaml.load(string_params, Loader=ruamel.yaml.Loader)
        values, keys = self._generate_params_from_string_params(p)

        # Add new experiments
        for elem in values:
            overides = {}
            for k, v in zip(keys, elem):
                if v is dict:
                    if defaults[k] is not dict:
                        raise ValueError(f'Attempting to update dictionary parameters but key {k} is not a dictionary'
                                         f'in the defaults.yml')
                    overides.update({k: defaults[k]})
                    for dict_k, dict_v in v.items():
                        if dict_k not in defaults[k]:
                            raise ValueError(f'key {dict_k} not found in defaults {k}')
                        overides[k].update(
                            {dict_k: type(defaults[k][dict_k])(dict_v)
                                if defaults[k][dict_k] is not None and dict_k is not None else dict_v})
                if v is list:
                    if defaults[k] is not list:
                        raise ValueError(f'Attempting to update a list of parameters but key {k} does not have '
                                         f'a list value')
                    if len(v) != len(defaults[k]):
                        raise ValueError(f'Override list length does not match default list length')
                    overides.update({k: [type(defaults[k][i])(v[i]) if defaults[k][i] is not None and v[i] is not None
                                         else v[i] for i in range(len(v))]})
                else:
                    overides.update({k: type(defaults[k])(v) if defaults[k] is not None and v is not None else v})

            # Check parameters are in the defaults.yml file
            if any([k not in defaults for k in overides]):
                raise ValueError('Some of the specified keys were not found in the defaults')

            experiment_name = '__'.join([k + '=' + str(v) for k, v in overides.items()])
            experiment_path = os.path.join(self.root, experiment_name)

            # Setup experiment folder
            if os.path.isdir(experiment_path):
                for i in range(1, 100):
                    if not os.path.isdir(experiment_path + f"_{i}"):
                        # logging.info(f"Directory already exists creating. The current experiment will be set up "
                        #              f"at {experiment_path}_{i}")
                        experiment_name = experiment_name + f"_{i}"
                        experiment_path = experiment_path + f"_{i}"
                        break
                    if i == 99:
                        raise ValueError('The number of experiments allowed with the same overides is limited to 100')
            os.makedirs(experiment_path)

            # Convert defaults to params
            # definition = defaults['definition'] if 'definition' in defaults else None
            if defaults['_version'] is not None:
                version = defaults['_version']
                if 'path' in version:
                    version = get_version(path=version['path'])
                elif 'module' in version and 'class' in version:
                    # We want to get version form the original class if possible
                    spec = importlib.util.spec_from_file_location("_em." + version['class'], version['module'])
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    cls = getattr(module, version['class'])
                    version = get_version(cls=cls)
            else:
                version = None
            extra_params = {
                '_root': self.root,
                '_name': experiment_name,
                '_status': 'registered',
                '_created': datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"),
                '_purpose': purpose,
                '_messages': {},
                '_version': version}

            params = copy.deepcopy(defaults)
            # Remove optional parameters from defaults
            for k in ['_created', '_version']:
                if k in params:
                    params.pop(k)

            # Add base parameters to params
            helps = get_attribute_helps(Experiment)
            for i, (k, v) in enumerate(extra_params.items()):
                h = helps[k].split(':')[1] if helps[k] is not None else helps[k]
                params.insert(i, k, v, h)

            # Generate a script.sh in each folder that can be used to run the experiment
            if header is not None and header != '':
                header_str = open(header).read()
            else:
                header_str = self._config.header
            script = f'#!{shell}\n{header_str}\n{shell} {os.path.join(self.script)} {os.path.join(experiment_path, "params.yml")}'
            with open(os.path.join(self.root, experiment_name, 'run.sh'), 'w') as f:
                f.write(script)

            # Update the overridden parameters
            params.update(overides)
            self.save_params(params, experiment_name)
            self.experiments.append(experiment_path)
            self.overides.append(overides)
            self._to_yml()

    def reset(self, pattern, status='registered'):
        """Update the status of the experiment. This is useful if you need to re-run an experiment
        from a latest saved checkpoint for example.

        Args:
            pattern: The experiment name
        """
        experiments = [p for p in glob.glob(os.path.join(self.root, pattern)) if p in self.experiments]
        for p in experiments:
            P = self.load_params(p)
            P['_status'] = status
            self.save_params(P, P['_name'])

    def list(self):
        """List all experiments currently created with the experiment manager."""
        self.check_initialised()

        if self.purpose is not None:
            print('Purpose: ' + self.purpose)
        print()

        # Construct dictionary
        if self.experiments != []:
            table = {'overrides': [], 'purpose': [], 'created': [], 'status': [], 'messages': [], 'commit': []}
            keys = ['_name', '_purpose', '_created', '_status', '_messages', '_version']
            for i, p in enumerate(self.experiments):
                P = self.load_params(p)
                for k_table, k_params in zip(table, keys):
                    if k_params == '_version':
                        if 'git' in P[k_params]:
                            v = P[k_params]['git']['commit']
                        else:
                            v = None
                        # print(P[k_params].keys())
                        # v = P[k_params]['comit']
                    else:
                        v = P[k_params]
                    table[k_table] += [v]
            table = pd.DataFrame(table)
            # with pd.option_context('display.max_rows', None, 'display.max_columns',
            #                        None):  # more options can be specified also
            print(table)
        else:
            print('No experiments currently registered!')
        print()

        # Print Notes
        if len(self.notes) > 0:
            print('Notes:')
            for n in self.notes:
                print('- ' + n)

        # Get defaults information
        defaults = self.load_defaults()
        if '_created' in defaults:
            print(f'Defaults: \n'
                  f'- created: {defaults["_created"]}')
        if '_version' in defaults:
            version = defaults["_version"]
            if 'path' in version:
                print(f'- path: {version["path"]}')
            else:
                print(f'- module: {version["module"]}')
                print(f'- class: {version["class"]}')
            if 'git' in version:
                git = version['git']
                print('- git:')
                print(f'   local: {git["local"]}')
                print(f'   commit: {git["commit"]}')
                print(f'   remote: {git["remote"]}')

    def clean(self):
        """Remove directories no longer linked to the experiment manager"""
        self.check_initialised()
        subdirs = [x[0] for x in os.walk(self.root) if x[0] != self.root and x[0] not in self.experiments]
        for d in subdirs:
            print(d)
            rmtree(d)

    def rm(self):
        self.check_initialised()
        if self._config.prompt_for_message:
            inp = input(f'This command will remove the whole experiment folder {self.root}. '
                        f'Do you wish to continue? [y | n]: ')
            if inp != 'y':
                print('Aborting!')
                return
        with self._config:
            self._config.experiments.remove(self.root)
            rmtree(self.root)
            print(f'Removed {self.root}')

    def run(self, pattern, *args):
        """Run all experiments that match the global pattern using the run command given by args."""
        experiments = [p for p in glob.glob(os.path.join(self.root, pattern)) if p in self.experiments]
        for p in experiments:
            P = self.load_params(p)
            if P['_status'] == 'registered':
                args = list(args)
                # subprocess_args = args + [self.script, os.path.join(p, 'params.yml')]
                subprocess_args = args + [os.path.join(p, 'run.sh')]
                print('\nRunning: {}'.format(" ".join(subprocess_args)))
                subprocess.call(subprocess_args)
                time.sleep(0.2)

    def unlink(self, pattern='*'):
        """Unlink all experiments matching pattern. Does not delete experiment folders simply deletes the experiment
        paths from the experiment folder. To delete call method ``clean``."""
        self.check_initialised()
        remove_paths = [p for p in glob.glob(os.path.join(self.root, pattern)) if p in self.experiments]
        if len(remove_paths) != 0:
            print("Removing Experiments...")
            for p in remove_paths:
                print(p)
                self.experiments.remove(p)
            self._to_yml()
            print("Note models are removed from the experiment list only. To remove the model directories run"
                  "experiment clean")
        else:
            print(f"No experiments match pattern {pattern}")

    def relink(self, pattern='*'):
        """Re-link all experiment folders that match ``pattern`` (and are not currently managed by the experiment
        manager)"""
        self.check_initialised()
        subdirs = [x[0] for x in os.walk(self.root)
                   if x[0] != self.root
                   and x[0] in glob.glob(os.path.join(self.root, pattern))
                   and x[0] not in self.experiments]
        if len(subdirs) == 0:
            print("No experiements to link that match pattern and aren't managed already")

        for d in subdirs:
            params, defaults = self.load_params(d, True), self.load_defaults()
            if any([k not in defaults and not k.startswith('_') for k in params]):
                print(f'Cannot re-link folder {d} as params are not compatible with defaults')
            else:
                print(f'Relinking {d}')
                self.experiments += [d]
                self.overides += [{pk: pv for pk, pv in params.items() if not pk.startswith('_') and defaults[pk] != pv}]
        self._to_yml()


if __name__ == "__main__":
    args = parser.parse_args()
    args.func(args)
