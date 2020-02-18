#!/usr/bin/env python3
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
import os
import pandas as pd

import argparse
import textwrap
import glob

from xmen.config import GlobalExperimentManager
from xmen.manager import ExperimentManager


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
    pd.set_option('display.max_colwidth', args.max_width)
    pd.set_option('display.max_rows', args.max_rows)
    if len(args.pattern) > 1:
        print(f'ERROR: Only one pattern may be passed but got {args.pattern}')
    pattern = os.path.abspath(os.path.expanduser(args.pattern[0]))
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


def _move(args):
    experiment_manager = ExperimentManager(args.src)
    experiment_manager.replant(args.dest)


def _run(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.run(args.experiments, *args.append)


def _unlink(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.unlink(args.experiments)


def _relink(args):
    config = GlobalExperimentManager()
    if args.root == '':
        args.root = os.getcwd()
    args.root = os.path.abspath(args.root)
    if args.experiments is None:
        # Perform global link
        if args.recursive:
            roots = [os.path.dirname(p) for p in glob.glob(args.root + '/**/experiment.yml', recursive=True)]
            if len(roots) == 0:
                print(f"No roots found for pattern {args.root + '/**/experiment.yml'}")
        else:
            roots = [args.root]
        for r in roots:
            if r not in config.experiments:
                experiment_manager = ExperimentManager(r)
                experiment_manager.replant(r)
            else:
                print(f'Experiment set {args.root} is already registered.')
    else:
        experiment_manager = ExperimentManager(args.root)
        experiment_manager.check_initialised()
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

# # Move
# move_parser = subparsers.add_parser('move', help='Move an experiment set from one location to another')
# move_parser.add_argument('src', help='The source directory to move', type=str)
# move_parser.add_argument('dest', help='The target destination folder', type=str)
# move_parser.set_defaults(func=_move)

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
relink_parser = subparsers.add_parser('relink', help='Relink experiments to global configuration or to a set root')
relink_parser.add_argument('-r', '--root', metavar='DIR', default='',
                           help='Path to the root experiment folder. If None then the current work directory will be '
                                'used.')
relink_parser.add_argument('--recursive', action='store_true', default=None,
                           help='If true recursively search recursively through subdirectories from root adding'
                                'all experiment sets found')
relink_parser.add_argument('-e', '--experiments', metavar='NAMES',
                           help='A unix glob giving the experiments to relink to a set (relative to experiment manager '
                                'root). If not passed then the experiment will be relinked globally.')
relink_parser.set_defaults(func=_relink)

# Enable command line interface
if __name__ == "__main__":
    args = parser.parse_args()
    args.func(args)
