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
import fnmatch
import time
import importlib.util
import argparse
import copy
import re
import textwrap
from copy import deepcopy
from xmen.utils import *
import socket
import getpass

def _init(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.initialise(defaults=args.defaults, script=args.script, purpose=args.purpose, name=args.name)


def _register(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.register(args.updates, args.purpose, args.header)


def _reset(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.reset(args.experiments)


def _records(args):
    global_exp_manager = GlobalExperimentManager()
    results, special_keys = global_exp_manager.find(
        mode='set', pattern=args.pattern, param_match=args.param_match, types_match=args.type_match)
    records = global_exp_manager.find_to_records(results, display_git=args.display_git)
    print(records)


def _list(args):
    # pd.set_option('expand_frame_repr', True)
    pd.set_option('display.max_colwidth', args.max_width)
    pd.set_option('display.max_rows', args.max_rows)

    if len(args.pattern) > 1:
        print(f'ERROR: Only one pattern may be passed but got {args.pattern}')
    pattern = args.pattern[0]
    if pattern == '':
        pattern = os.path.join(os.getcwd() + '*')
    global_exp_manager = GlobalExperimentManager()

    if args.list:
        results = global_exp_manager.find(
            mode='set', pattern=pattern, param_match=args.param_match, types_match=args.type_match,
            load_defaults=True)
        notes = []
        for i, (r, e, p, n, d, t) in enumerate(
                zip(*[results[j] for j in ('_root', '_experiments', '_purpose', '_notes', '_created', '_type')])):
            k = 5
            i = str(i)
            note = ' ' * (k // 2 - len(str(i))) + str(i) + ' ' * (k // 2 - 1) + r + '\n' + ' ' * k
            if len(e) > 0:
                note += ('\n' + ' ' * k).join(['|- ' + ee[len(r) + 1:] for ee in e]) + '\n' + ' ' * k
            note += 'Purpose: ' + p + '\n' + ' ' * k
            note += 'Created: ' + d + '\n' + ' ' * k
            note += 'Type: ' + str(t)
            if len(n) > 0:
                note += '\n' + ' ' * k + 'Notes: ' + '\n' + ' ' * (k + 2)
                note += ('\n' + ' ' * (k + 2)).join(
                    ['\n'.join(textwrap.wrap(nn, width=1000, subsequent_indent=' ' * (k + 3))) for i, nn in
                     enumerate(n)])
            notes += [note]
        print('\n'.join(notes))
    else:
        results = global_exp_manager.find(
            mode='all', pattern=pattern, param_match=args.param_match, types_match=args.type_match,
            load_defaults=args.load_defaults)
        data_frame, root = global_exp_manager.find_to_dataframe(
            results,
            verbose=args.verbose,
            display_git=args.display_git,
            display_purpose=args.display_purpose,
            display_date=args.display_date,
            display_messages=args.display_messages,
            display_meta=args.display_meta,
            display_status=args.display_status)
        if data_frame.empty:
            print(f'No experiments found which match glob pattern {pattern}. With parameter filter = {args.param_match} '
                  f'and type filter = {args.type_match}.')
        else:
            print(data_frame)
            print(f'\nRoots relative to: {root}')


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
    with GlobalExperimentManager() as config:
        if args.disable_prompt is not None:
            config.prompt_for_message = False
        elif args.enable_prompt is not None:
            config.prompt_for_message = True
        if args.add is not None:
            config.add_class(args.add)
        if args.add_path is not None:
            if args.add_path not in config.python_paths:
                config.python_paths.append(os.path.abspath(args.add_path))
        if args.update_meta is not None:
            config.update_meta()
            # config.meta.update({args.update_meta[0]: args.update_meta[1]})

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
        if args.list is not None:
            print(config)
        if args.clean is not None:
            config.clean()


parser = argparse.ArgumentParser(prog='xmen',
                                 description='A helper module for the quick setup and management of experiments')
subparsers = parser.add_subparsers()

# Config
config_parser = subparsers.add_parser('config')
config_parser.add_argument('--disable_prompt', action='store_false',
                           help='Turn purpose prompting off', default=None)
config_parser.add_argument('--clean', action='store_false',
                           help='Remove experiments that have been corrupted from the global configuration.',
                           default=None)
config_parser.add_argument('--enable_prompt', action='store_false',
                           help='Turn purpose prompting on', default=None)
config_parser.add_argument('--add_path', type=str, default=None, help='Add pythonpath to the global config')
config_parser.add_argument('--add', default=None, metavar='PATH',
                            help='Add an Experiment api python script (it must already be on PYTHONPATH)')
config_parser.add_argument('--update_meta', default=None, action='store_true',
                           help='Update meta information in each experiment (both defaults.yml and params.yml). '
                                'WARNING: Overwrites information in the params.yml or defaults.yml')

config_parser.add_argument('-r', '--remove', default=None, help='Remove a python path or experiment (passed by Name) '
                                                                'from the config.')
config_parser.add_argument('-H', '--header', type=str, help='Update the default header used when generating experiments'
                                                            ' to HEADER (a .txt file)')
config_parser.add_argument('--list', default=None, help='Display the current configuration', action='store_false')
config_parser.set_defaults(func=_config)


# Note parser
note_parser = subparsers.add_parser('note')
note_parser.add_argument('message', help='Add note to experiment set')
note_parser.add_argument('-r', '--root', metavar='DIR', default='',
                         help='Path to the root experiment folder. If None then the current work directory will be '
                              'used')
note_parser.add_argument('-d', '--delete', default='', action='store_true',
                         help='Delete the note corresponding to message.')
note_parser.set_defaults(func=_note)

# Init
init_parser = subparsers.add_parser('init', help='Initialise an experiment set.')
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
register_parser.add_argument('-helps', '--header', metavar='PATH', help='A header file to prepend to each run script', default=None)
register_parser.add_argument('-p', '--purpose', metavar='STR', help='A string giving the purpose of the experiment.')
register_parser.add_argument('-r', '--root', metavar='DIR', default='',
                             help='Path to the root experiment folder. If None then the current work directory will be '
                                  'used')
register_parser.set_defaults(func=_register)

# List
list_parser = subparsers.add_parser('list', help='List (all) experiments to screen')
list_parser.add_argument('pattern', type=str, help='List experiments which match pattern.', default=[''], nargs='*')
list_parser.add_argument('-p', '--param_match', type=str, default=None, nargs='*',
                         help="List only experiments with certain parameter conditions of the form reg, reg==val or "
                              "val1==reg==val2. Here reg is a regex matching a set of parameters. "
                              "==, <, >, !=, >=, <=  are all supported with meaning defined as in python. Eg. "
                              "a.*==cat and 1.0<a.*<=2.0 will return any experiment that has parameters that match "
                              "a.* provided that each match satisfies the condition.")
list_parser.add_argument('-n', '--type_match', type=str, default=None, nargs='*',
                         help="List only experiments with this type (class).")
list_parser.add_argument('-v', '--verbose', action='store_true', default=None,
                         help="Display all information for each experiment")
list_parser.add_argument('-d', '--display_date', action='store_true', default=None,
                         help="Display created date for each experiment")
list_parser.add_argument('-g', '--display_git', action='store_true', default=None,
                         help="Display git commit for each experiment")
list_parser.add_argument('-P', '--display_purpose', action='store_true', default=None,
                         help="Display purpose for each experiment")
list_parser.add_argument('-s', '--display_status', action='store_true', default=None,
                         help="Display status for each experiment")
list_parser.add_argument('-m', '--display_messages', action='store_true', default=None,
                         help="Display messages for each experiment")
list_parser.add_argument('-M', '--display_meta', action='store_true', default=None,
                         help="Display messages for each experiment")
list_parser.add_argument('-l', '--list', action='store_true', default=None,
                         help="Display as list and not a table")
list_parser.add_argument('--load_defaults', action='store_true', default=None,
                         help='Infer parameters from defaults.yml and overides instead of params.yml. Potentially '
                              'faster but no messages are available.')
list_parser.add_argument('--max_width', default=60, help='The maximum width of an individual collumn. '
                                                           'If None then will print for ever', type=int)
list_parser.add_argument('--max_rows', default=None, help='Display tables with this number of rows.', type=int)

list_parser.set_defaults(func=_list)

# Run
run_parser = subparsers.add_parser('run', help='Run experiments matching glob in experiment set that have not yet'
                                               'been run.')
run_parser.add_argument('experiments', metavar='NAMES', help='A unix glob giving the experiments to be run in the set')
run_parser.add_argument('append', metavar='FLAG', nargs=argparse.REMAINDER,
                        help='A set of run command options to prepend to the run.sh for each experiment '
                             '(eg. "sh", "srun", "sbatch", "docker" etc.)')
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
clean_parser = subparsers.add_parser('clean', help='(DESTRUCTIVE) Remove unlinked experiments')
clean_parser.add_argument('-r', '--root', metavar='DIR', default='',
                          help='Path to the root experiment folder. If None then the current work directory will be '
                               'used')
clean_parser.set_defaults(func=_clean)

# Removes
remove_parser = subparsers.add_parser('rm', help='(DESTRUCTIVE) Remove an experiment set.')
remove_parser.add_argument('root', metavar='ROOT_DIR', help='Path to the root experiment folder to be removed.')
remove_parser.set_defaults(func=_rm)


# Unlink
unlink_parser = subparsers.add_parser('unlink', help='Unlink experiments from experiment set')
unlink_parser.add_argument('experiments', metavar='NAMES', help='A unix glob giving the experiments to be unlinked')
unlink_parser.add_argument('-r', '--root', metavar='DIR', default='',
                           help='Path to the root experiment folder. If None then the current work directory will be '
                                'used')
unlink_parser.set_defaults(func=_unlink)

# Relink
relink_parser = subparsers.add_parser('relink', help='Relink experiments to experiment set')
relink_parser.add_argument('experiments', metavar='NAMES', help='A unix glob giving the experiments to be relinked '
                                                                '(relative to experiment manager root)')
relink_parser.add_argument('-r', '--root', metavar='DIR', default='',
                           help='Path to the root experiment folder. If None then the current work directory will be '
                                'used')
relink_parser.set_defaults(func=_relink)


class NoMatchException(Exception):
    def __init__(self, path):
        """Experiment at ``path`` does not contain a match"""
        super(NoMatchException, self).__init__()
        self.path = path


class GlobalExperimentManager(object):
    """A helper class used to manage global configuration of the Experiment Manager"""
    def __init__(self):
        self.python_experiments = {}   # A dictionary of paths to python modules compatible with the experiment api
        self.python_paths = []         # A list of python paths needed to run each module
        self.prompt_for_message = True
        self.experiments = {}          # A list of all experiments registered with an Experiment Manager
        self.meta = get_meta()
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
        """Load the experiment config from a ``config.yml`` file"""
        with open(os.path.join(self._dir, 'config.yml'), 'r') as file:
            params = ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)
            to_yml = False
            for k, v in params.items():
                if k == 'experiments' and len(v) != 0 and type(v) is not CommentedMap:
                    experiments = {}
                    to_yml = True
                    for vv in v:
                        em = ExperimentManager(root=vv, headless=True)
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
            em = ExperimentManager(p)
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
                em = ExperimentManager(p)
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
        string = f'Prompt for Message: {self.prompt_for_message}\n'
        string += '\n'
        string += f'Python Path:\n'
        for p in self.python_paths:
            string += f'  - {p}\n'
        string += '\n'
        string += 'Python Experiments:\n'
        for k, v in self.python_experiments.items():
           string += f'  - {k}: {v}\n'
        string += '\n'
        string += 'Header:\n'
        string += self.header + '\n'
        string += '\n'
        string += 'Meta:\n'
        for k, m in self.meta.items():
            string += f'  - {k}: {m}\n'
        string += '\n'
        string += 'Experiments:\n'
        for k, e in self.experiments.items():
            e = dict(e)
            string += f'  - {e["created"]}: {k}\n'

        return string

    def find(self, mode='all', pattern="*", param_match=None, types_match=None, load_defaults=False, missing_entry=''):
        """Search through all experiments in the globally filtering by experiment location, parameters, and type.

        Args:
            mode (str): If 'all' then all experiments for each set will be concatenated and summarised individually
                whilst if 'set' then experiments will be summarised and searched for based on the default parameters.
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
        experiments = fnmatch.filter(self.experiments.keys(), pattern)
        encountered = []
        for root in experiments:
            try:
                em = ExperimentManager(root)
                if load_defaults:
                    defaults = em.load_defaults()
                if types_match is not None:
                    if em.type not in types_match:
                        raise NoMatchException(root)

                paths = em.experiments if mode == 'all' else [em.defaults]
                for i, path in enumerate(paths):
                    if mode == 'all':
                        if not load_defaults:
                            params = em.load_params(path)
                            if params is None:
                                print(path)
                        else:
                            params = deepcopy(defaults)
                            params.update(em.overides[i])
                    else:
                        params = em.load_defaults()

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
                            keys = [k for k in params if re.match(reg, k) is not None]
                            if len(keys) == 0:
                                raise NoMatchException(root)

                            keys += [k for k in params if k.startswith('_')]
                            for key in keys:
                                if valid(params[key]):
                                    extracted[key] = params[key]
                                else:
                                    raise NoMatchException(root)

                        if len(extracted) == 0:
                            raise NoMatchException(root)
                        else:
                            # If we get here add parameters to experiments (extracted must have at least one element)
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
                            if k.startswith('_') and k not in ['_root', '_purpose']:
                                if k not in table:
                                    table[k] = [v]
                                else:
                                    table[k] += [v]
                        if '_meta' not in params:
                            table['_meta'] += [get_meta()]

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
                          display_meta=None):
        """Convert the output of the `find` method to a formatted data frame configured by args. If verbose then all
        entries will be displayed (independent of the other args)"""

        special_keys = [k for k in table if k.startswith('_')]
        display = {
            'root': table.pop('_root')}

        if '_name' in special_keys:
            display['name'] = table.pop('_name')
        display.update({
            'created': table.pop('_created'),
            'type': table.pop('_type'),
            'purpose': table.pop('_purpose')})

        meta = table.pop('_meta')
        meta_dict = {}
        for m in meta:
            for k, v in m.items():
                if k in meta_dict:
                    meta_dict[k] += [m[k]]
                else:
                    meta_dict[k] = [m[k]]
        display.update(meta_dict)

        # Remove Notes
        table.pop('_notes')

        if '_status' in special_keys:
            display['status'] = table.pop('_status')
        # Add version information to table
        versions = table.pop('_version')
        display['commit'] = [version.get('git', {}).get('commit', None) for version in versions]

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
            if display_git:
                display_keys += ['commit']
            if display_status:
                display_keys += ['status']
            if display_date:
                display_keys += ['date']
            if display_meta:
                display_keys += meta_dict.keys()
            if display_messages and '_messages' in special_keys:
                display_keys += encountered
                display_keys += ['date']
            if display_purpose:
                display_keys += ['purpose']

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
        df = df.filter(items=display_keys)
        return df, prefix

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

    def __init__(self, root="", headless=False):
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
        self.type = None
        self._specials = ['_root', '_name', '_status', '_created', '_purpose', '_messages', '_version', '_meta']
        if not headless:
            self._config = GlobalExperimentManager()
        else:
            self._config = None

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

    def update_meta(self):
        """Save a dictionary of parameters at ``{root}/{experiment_name}/params.yml``

        Args:
            params (dict): A dictionary of parameters to be saved. Can also be a CommentedMap from ruamel
            experiment_name (str): The name of the experiment
        """
        defaults = self.load_defaults()
        if '_meta' not in defaults:
            defaults.insert(2, '_meta', get_meta())
        else:
            defaults['_meta'] = get_meta()
        # experiment_path = os.path.join(self.root, experiment_name)
        with open(os.path.join(self.root, 'defaults.yml'), 'w') as out:
            yaml = ruamel.yaml.YAML()
            yaml.dump(defaults, out)

        for p in self.experiments:
            params = self.load_params(p)
            if '_meta' not in params:
                params.insert(7, '_meta', get_meta())
            else:
                params['_meta'] = get_meta()
            self.save_params(params, os.path.basename(p))

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
            if 'purpose' in params:
                self.purpose = params['purpose']
            if 'notes' in params:
                self.notes = params['notes']
            if 'type' in params:
                self.type = params['type']

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
            self.type = name

        # Meta Information
        self.created = datetime.datetime.now().strftime(DATE_FORMAT)

        # Save state to yml
        if os.path.exists(os.path.join(self.root, 'experiment.yml')):
            print(f"There already exists a experiment.yml file in the root directory {self.root}. "
                  f"To reinitialise an experiment folder remove the experiment.yml.")
            exit()
        print(f'Experiment root created at {self.root}')

        # Add purpose message
        if self._config.prompt_for_message:
            purpose = input('\nPlease enter the purpose of the experiments: ')
        self.purpose = purpose

        # Add experiment to global config
        with self._config:
            self._config.experiments[self.root] = {
                'created': self.created,
                'type': self.type,
                'purpose': self.purpose,
                'notes': self.notes}
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

        # Add experiment to global config
        with self._config:
            self._config.experiments[self.root]['notes'] = self.notes
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
                dictionaries {a: 1., 2.} or lists [1, 2] of these types_match. None parameters are specified using empty space.
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
                        overides[k].update({dict_k: dict_v})
                if v is list:
                    if defaults[k] is not list:
                        raise ValueError(f'Attempting to update a list of parameters but key {k} does not have '
                                         f'a list value')
                    if len(v) != len(defaults[k]):
                        raise ValueError(f'Override list length does not match default list length')
                    overides.update({k: [v[i] for i in range(len(v))]})
                else:
                    overides.update({k: v})

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
                '_created': datetime.datetime.now().strftime(DATE_FORMAT),
                '_purpose': purpose,
                '_messages': {},
                '_version': version,
                '_meta': get_meta()}

            params = copy.deepcopy(defaults)
            # Remove optional parameters from defaults
            for k in ['_created', '_version', '_meta']:
                if k in params:
                    params.pop(k)

            # Add base parameters to params
            # helps = get_attribute_helps(Experiment)
            from xmen.experiment import Experiment
            for i, (k, v) in enumerate(extra_params.items()):
                h = Experiment._Experiment__params[k][2]
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
            self._config.experiments.pop(self.root)
            rmtree(self.root)

            print(f'Removed {self.root}')

    def run(self, pattern, *args):
        """Run all experiments that match the global pattern using the run command given by args."""
        experiments = [p for p in glob.glob(os.path.join(self.root, pattern)) if p in self.experiments]
        for p in experiments:
            P = self.load_params(p)
            if P['_status'] == 'registered':
                args = list(args)
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
