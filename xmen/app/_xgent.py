#!/usr/bin/env experiments
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
import argparse
import textwrap
import glob

from xmen.config import GlobalExperimentManager, Config
from xmen.manager import ExperimentManager, InvalidExperimentRoot

DESCRIPTION = r"""
||||||||||||||||||||||||||| WELCOME TO |||||||||||||||||||||||||||
||                                                              ||
||          &@&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&@&%          ||    
||         *@&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&         ||    
||          &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&          ||    
||           &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&          ||    
||           &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#          ||    
||           &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&.          ||    
||           &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.          ||    
||           &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*          ||    
||           @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@          ||    
||   #&@@@@@&%&&&&&&&@@@@@@@@@@@@@@@@@@@@@@@@@&&&&&&&&@@@@@@&#  ||    
||  /#%%%%%%%%%&&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&&%%&%%%%%%#  ||    
||   &%&&&&&&&&&&&@@@@@@@@@@@@@@@@@@@@@@@@@@@&&@&&&&&&&&&&&&&   ||    
||     (@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&.    ||    
||      ...,*/#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&##(*,...      ||    
||                                                              ||
||    \\\  ///  |||\\        //|||  |||||||||  |||\\   |||      ||
||     \\\///   |||\\\      ///|||  |||        |||\\\  |||      ||
||      ||||    ||| \\\    /// |||  ||||||     ||| \\\ |||      ||
||     ///\\\   |||  \\\  ///  |||  |||        |||  \\\|||      ||
||    ///  \\\  |||   \\\///   |||  |||||||||  |||   \\|||      ||
||                                                              ||
||                      %@@,     (@@/                           ||
||                     @@@@@@@@@@@@@@@@@@@@@                    ||
||        @@        @@@@@@@@@@@@@@@@@@@@@@@@@@/        @#       ||
||       @@#     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#     @@       ||
||        @@@@@@@@@@@@@@@@@@@@@@@.@@@@@@@@@@@@@@@@@@@@@@.       ||
||           ,@@@@@@@@@@@@@@@%       @@@@@@@@@@@@@@@@           ||
||                                                              ||
|||||||||||||| FAST - REPRODUCIBLE - EXPERIMENTATION |||||||||||||
"""
parser = argparse.ArgumentParser(
    prog='xmen',
    description=DESCRIPTION,
    formatter_class=argparse.RawTextHelpFormatter)
subparsers = parser.add_subparsers()


#######################################################################################################################
#  add
#######################################################################################################################
python_parser = subparsers.add_parser('python', help='Python interface')
python_parser.add_argument('--list', '-l', action='store_true', default=None, help='List available python experiments')
python_parser.add_argument('--add', '-a', default=None, metavar=('MODULE', 'NAME'),
                           help='Add a python Experiment class or run script (it must already be on PYTHONPATH)',
                           nargs=2)
python_parser.add_argument('name', help='The name of the experiment to run', nargs='*', default=None)
python_parser.add_argument('--remove', '-r', help='Remove a python experiment (passed by Name)')
python_parser.add_argument('flags', help='Python flags (pass --help for more info)', nargs=argparse.REMAINDER, default=[])


def _python(args):
    config = Config()
    if args.list is not None:
        print(f'The following python experiments are currently linked')
        for k, v in config.python_experiments.items():
            print(f'{k}: {v}')
    if args.add is not None:
        try:
            with config as c:
                c.add_python(*args.add)
            print(f'Added experiment {args.add[-1]} from module {args.add[-2]}')
        except:
            print(f'ERROR: failed to add experiment {args.add[-1]} from module {args.add[-2]}')
    if args.remove is not None:
        with config as c:
            if args.remove in c.python_experiments:
                path = c.python_experiments.pop(args.remove)
                if '.xmen' in path:
                    os.remove(path)
                print(f'Successfully removed {args.remove}!')
    if args.name:
        import subprocess
        if args.name[0] not in config.python_experiments:
            print(f'No experiments found matching {args.name[0]}')
            exit()
        args = [config.python_experiments[args.name[0]]] + args.flags
        subprocess.call(args)


python_parser.set_defaults(func=_python)
#######################################################################################################################
#  config
#######################################################################################################################
config_parser = subparsers.add_parser('config', help='View / edit the global configuration')
config_parser.add_argument('--interactive', help='Interactively configure the configuration', default=None,
                           action='store_false')
config_parser.add_argument('--disable_prompt', action='store_false',
                           help='Turn purpose prompting off', default=None)
config_parser.add_argument('--clean', action='store_false',
                           help='Remove experiments that have been corrupted from the global configuration.',
                           default=None)
config_parser.add_argument('--register_user', default=None, action='store_false',
                           help='Register the user with the remote server')
config_parser.add_argument('--enable_prompt', action='store_false',
                           help='Turn purpose prompting on', default=None)
config_parser.add_argument('--disable_save_conda', action='store_false',
                           help='Turn conda environment saving off', default=None)
config_parser.add_argument('--enable_save_conda', action='store_false',
                           help='Turn conda environment saving on', default=None)
config_parser.add_argument('--disable_stdout_to_txt', action='store_false',
                           help='Disable logging to text file in each experiment folder', default=None)
config_parser.add_argument('--enable_stdout_to_txt', action='store_false',
                           help='Enable saving to text file in each experiment folder', default=None)

config_parser.add_argument('--disable_requeue', action='store_false',
                           help='Disable automatic requeue of experiments on timeout', default=None)
config_parser.add_argument('--enable_requeue', action='store_false',
                           help='Enable automatic requeue of experiments on timeout', default=None)

config_parser.add_argument('--change_password',  help='Change the password for the current user', default=None,
                           action='store_false')
config_parser.add_argument('--sync', help='Synchronise experiments in the cwd with the remote server', default=None,
                           action='store_false')
config_parser.add_argument('--sync_all', help='Synchronise ALL experiments with the remote server', default=None,
                           action='store_false')
config_parser.add_argument('--update_meta', default=None, action='store_true',
                           help='Update meta information in each experiment (both defaults.yml and params.yml). '
                                'WARNING: Overwrites information in the params.yml or defaults.yml')
config_parser.add_argument('-H', '--header', type=str, help='Update the default header used when generating experiments'
                                                            ' to HEADER (a .txt file)')
config_parser.add_argument('--list', default=None, help='Display the current configuration', action='store_false')
config_parser.add_argument('--server_host',  type=str, help='Update the default host used by xmen', nargs='?')
config_parser.add_argument('--server_port', type=int, help='Update the default port used by xmen', nargs='?')


def _config(args):
    with Config() as config:
        if args.interactive is not None:
            config.setup()
        if args.disable_prompt is not None:
            config.prompt = False
        elif args.enable_prompt is not None:
            config.prompt = True
        if args.disable_save_conda is not None:
            config.save_conda = False
        elif args.enable_save_conda is not None:
            config.save_conda = True
        if args.register_user is not None:
            from getpass import getpass
            user = input('username: ')
            password = getpass(prompt='password: ')
            check = getpass(prompt='confirm password: ')
            if check != password:
                print('ERROR: Passwords do not match')
            else:
                config.register_user(user, password)
        if args.change_password is not None:
            from getpass import getpass
            password = getpass(prompt='Current Password: ')
            new_password = getpass(prompt='New Password: ')
            check = getpass(prompt='Confirm new Password: ')
            if new_password != check:
                print('ERROR: Passwords do not match')
            else:
                config.change_password(password, new_password)
        if args.disable_stdout_to_txt is not None:
            config.redirect_stdout = False
        elif args.enable_stdout_to_txt is not None:
            config.redirect_stdout = True

        if args.disable_requeue is not None:
            config.requeue = False
        elif args.enable_requeue is not None:
            config.requeue = True
        if args.header is not None:
            if os.path.exists(args.header):
                config.header = open(args.header, 'r').read()
            else:
                config.header = args.header
        if args.server_host:
            print(f'Updating server host to {args.server_host}')
            config.server_host = args.server_host
        if args.server_port:
            print(f'Updating server port to {args.server_port}')
            config.server_port = args.server_port
        if args.list is not None:
            print(config)
        if args.clean is not None:
            config.clean()
        if args.sync is not None:
            roots = config.filter(os.getcwd() + '.*')
            print('Synchronising...')
            for r in roots:
                print(r)
            config.sync(roots)
        if args.sync_all is not None:
            print('Synchronising...')
            for r in config.linked:
                print(r)
            config.sync()


config_parser.set_defaults(func=_config)


#######################################################################################################################
#  init
#######################################################################################################################
init_parser = subparsers.add_parser('init', help='Initialise an experiment set')
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
                         help='A name of a experiments experiment registered with the global configuration.')
init_parser.add_argument('--purpose', metavar='PURPOSE', default='',
                         help='A string giving the purpose of the experiment set (only used if message prompting is '
                              'disabled).')


def _init(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.initialise(defaults=args.defaults, script=args.script, purpose=args.purpose, name=args.name)


init_parser.set_defaults(func=_init)

#######################################################################################################################
#  register
#######################################################################################################################
register_parser = subparsers.add_parser('register', help='Register a set of experiments')
register_parser.add_argument('-n', '--name', help='Name of the experiment', default=None)
register_parser.add_argument('-u', '--updates', metavar='YAML_STR',
                             help='Defaults to update to register experiments passes as a yaml dict. '
                                  'The special character'
                                  '"|" is interpreted as an or operator. all combinations of parameters appearing '
                                  'either side of "|" will be registered.', default=None)
register_parser.add_argument('-H', '--header', metavar='PATH', help='A header file to prepend to each run script',
                             default=None)
register_parser.add_argument('-p', '--purpose', metavar='STR', help='A string giving the purpose of the experiment.')
register_parser.add_argument('-r', '--root', metavar='DIR', default='',
                             help='Path to the root experiment folder. If None then the current work directory will be '
                                  'used')
register_parser.add_argument('-x', '--repeats', metavar='DIR', default=1, type=int,
                             help='Repeat experiment(s) this number of times')


def _register(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.register(args.name, args.updates, args.purpose, args.header, repeats=args.repeats)


register_parser.set_defaults(func=_register)


#######################################################################################################################
#  run
#######################################################################################################################
run_parser = subparsers.add_parser('run', help='Run experiments matching glob in experiment set that have not yet'
                                               'been run.')
run_parser.add_argument('experiments', metavar='NAMES', help='A unix glob giving the experiments to be run in the set')
run_parser.add_argument('append', metavar='FLAG', nargs=argparse.REMAINDER,
                        help='A set of run command options to prepend to the run.sh for each experiment '
                             '(eg. "sh", "srun", "sbatch", "docker" etc.)')
run_parser.add_argument('-r', '--root', metavar='DIR', default='',
                        help='Path to the root experiment folder. If None then the current work directory will be '
                              'used')


def _run(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.run(args.experiments, *args.append)


run_parser.set_defaults(func=_run)


#######################################################################################################################
#  note
#######################################################################################################################
note_parser = subparsers.add_parser('note', help='add notes to an experiment')
note_parser.add_argument('pattern', help='Add messages to these experiments')
note_parser.add_argument('message', help='Message to add to the experiments')
note_parser.add_argument('-r', '--root', metavar='DIR', default='',
                         help='Path to the root of the experiment. If None then the current work directory will be '
                              'used')
note_parser.add_argument('-d', '--delete', default='', action='store_true',
                         help='Delete the note corresponding to message.')


def _note(args):
    if not args.root:
        args.root = os.getcwd()
    print(f'Leaving note with experiment {args.root}')
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.note(args.pattern, args.message, args.delete)


note_parser.set_defaults(func=_note)


#######################################################################################################################
#  reset
#######################################################################################################################
status_parser = subparsers.add_parser('reset', help='Reset an experiment to registered status')
status_parser.add_argument('experiments', metavar='NAME', help='A unix glob giving the experiments whos status '
                                                               'should be updated (relative to experiment manager root)')
status_parser.add_argument('-r', '--root', metavar='DIR', default='',
                           help='Path to the root experiment folder. If None then the current work directory will be '
                                'used')
status_parser.add_argument('-s', '--status', metavar='DIR', default='registered',
                           help='Path to the root experiment folder. If None then the current work directory will be '
                                'used')

def _reset(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.reset(args.experiments, args.status)


status_parser.set_defaults(func=_reset)

#######################################################################################################################
#  unlink
#######################################################################################################################
unlink_parser = subparsers.add_parser('unlink', help='Unlink experiments from experiment set')
unlink_parser.add_argument('experiments', metavar='NAMES', help='A unix glob giving the experiments to be unlinked')
unlink_parser.add_argument('-r', '--root', metavar='DIR', default='',
                           help='Path to the root experiment folder. If None then the current work directory will be '
                                'used')


def _unlink(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.unlink(args.experiments)


unlink_parser.set_defaults(func=_unlink)


#######################################################################################################################
# clean
#######################################################################################################################
clean_parser = subparsers.add_parser('clean', help='Remove unlinked experiments (DESTRUCTIVE)')
clean_parser.add_argument('-r', '--root', metavar='DIR', default='',
                          help='Path to the root experiment folder. If None then the current work directory will be '
                               'used')
clean_parser.add_argument('--all', help='If passed experiment sets not corresponding to ')


def _clean(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.clean()


clean_parser.set_defaults(func=_clean)

#######################################################################################################################
#  rm
#######################################################################################################################
remove_parser = subparsers.add_parser('rm', help='Remove an experiment set (DESTRUCTIVE)')
remove_parser.add_argument('root', metavar='ROOT_DIR', help='Path to the root experiment folder to be removed.')


def _rm(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.rm()


remove_parser.set_defaults(func=_rm)


#######################################################################################################################
#  relink
#######################################################################################################################
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


def _relink(args):
    from xmen.utils import load_param, save_param
    config = Config()
    if args.root == '':
        args.root = os.getcwd()
    args.root = os.path.abspath(args.root)
    if args.experiments is None:
        # relink experiment managers
        if args.recursive:
            managers = [os.path.dirname(p) for p in glob.glob(args.root + '/**/experiment.yml', recursive=True)]
            if len(managers) == 0:
                print(f"No roots found for pattern {args.root + '/**/experiment.yml'}")
        else:
            managers = [args.root]

        for manager in managers:
            experiment_manager = ExperimentManager(manager)
            experiment_manager.replant(manager)

        # relink experiments
        if args.recursive:
            roots = [os.path.dirname(p) for p in glob.glob(args.root + '/**/params.yml', recursive=True)]
            if len(managers) == 0:
                print(f"No roots found for pattern {args.root + '/**/params.yml'}")
        else:
            roots = [args.root]

        roots = [r for r in roots if r not in config.linked]
        config.link(roots)
        config.clean()

        for r in roots:
            params = load_param(r)
            if params['_root'] != r:
                params['_root'] = r
                save_param(params, r)
    else:
        experiment_manager = ExperimentManager(args.root)
        experiment_manager.check_initialised()
        experiment_manager.relink(args.experiments)


relink_parser.set_defaults(func=_relink)

#######################################################################################################################
#  list
#######################################################################################################################
config = Config()
list_parser = subparsers.add_parser('list', help='list experiments to screen')
# ---- Modes
list_parser.add_argument('-i', '--interval', type=float, default=None, const=1., nargs='?',
                         help='Run in interactive mode. If interval is passed then the table will be updated '
                              'every n seconds (defaults to 1.).')
list_parser.add_argument(
    '-N', '--list',
    action='store_true', default=None,
    help="Display experiments in notebook mode")
# --- FILTERS ---
list_parser.add_argument(
    '--pattern',
    type=str,
    help='Display only experiments matching pattern. In normal or notebook mode will be just a simple regex. '
         'In interactive mode it should be set as {user}@{host}:{pattern} where user, host and pattern can all '
         'be regex.  Defaults to "{CWD}.*" where CWD is the current work directory.',
    default=None,
    nargs='?')
list_parser.add_argument(
    '-S', '--status_filter',
    nargs="?",
    default="[^(deleted)]",
    help="Consider only experiments matching status_filter with the server (used only in interactive mode.)")

list_parser.add_argument(
    '-f', '--filters',
    nargs="*",
    default=['^$'],
    help="Apply a filter to the experiments. Can be in the form '(?!_)a.*' or "
         "'_version.*' to display all information matching regex. Can also be used to filter experiments eg. "
         "a.*=='Hello', sleep<2.0 and _version_module=='xmen.*'. Predicates ==, <=, <, >, >=, != are all valid. "
         "If mutlitple filters with predicates are passed then the results in the table will be combined as an OR "
         "operation. An AND operation can be mimicked using De Morgans theorem. "
         " Strict parameter regexes can be set using a negative look ahead of the form "
         "(?!_) eg. (?!_).* for all parameters. "
         "Special keys '_date.*', '_status.*', '_messages_.*', '_version_.*', '_meta.*', "
         "'_purpose.*' are all available. Nested dictionaries are flattened eg. "
         "'_version_git_commit'")
# --- DISPLAY TOGGLES ---
list_parser.add_argument(
    '-v', '--display_version',
    default=r'^$',
    const='_version',
    action='store_const',
    help="Display version information for each experiment")
list_parser.add_argument(
    '-P', '--display_purpose',
    default=r'^$',
    const=r'_purpose$',
    action='store_const',
    help="Display purpose for each experiment")
list_parser.add_argument(
    '-d', '--display_date',
    default=r'^$',
    const="_timestamps|_created",
    action='store_const',
    help="Display created date for each experiment")
list_parser.add_argument(
    '-s', '--display_status',
    default=r'^$',
    const=r'_status$',
    action='store_const',
    help="Display status for each experiment")
list_parser.add_argument(
    '-m', '--display_messages',
    action='store_const',
    default='^$',
    const='_messages_(last$|e$|s$|wall$|end$|next$|.*step$|.*load$)',
    help="Display messages for each experiment")
list_parser.add_argument(
    '-M', '--display_meta',
    action='store_const',
    default="^$",
    const='_meta_(root$|name$|mac$|host$|user$|home$)',
    help="Display meta information for each experiment. The regex "
         "'' gives basic meta information "
         "logged with every experiment. Other information is separated into groups including "
         "'network.*', 'gpu.*', 'cpu.*', 'system.*', 'virtual.*', 'swap.*'")
# --- CONFIG ---
list_parser.add_argument('--max_width', default=60, help='The maximum width of an individual collumn. '
                                                         'If None then will print for ever', type=int)
list_parser.add_argument('--max_rows', default=None, help='Display tables with this number of rows.', type=int)
list_parser.add_argument('--max_n', default=40, help='Display tables with this number of rows.', type=int)
list_parser.add_argument('--csv', action='store_true', help='Display the table as csv.', default=None)


def _curses_list(args):
    from xmen.list import NotEnoughRows
    _list(None, args)
    # if args.interval is not None:
    #     import curses
    #     try:
    #         args.param_match = "^$"
    #         curses.wrapper(_list, args)
    #     except NotEnoughRows:
    #         print('WARNING: Not enough rows to display the table in interactive mode. Find a bigger terminal')
    #     except KeyboardInterrupt:
    #         pass
    # else:
    #     _list(None, args)


def _list(stdscr, args):
    import pandas as pd
    from xmen.utils import load_params
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_colwidth', args.max_width)
    pd.set_option('display.max_rows', args.max_rows)

    if args.pattern is None:
        pattern = os.getcwd()
    else:
        pattern = os.path.abspath(os.path.expanduser(args.pattern[0]))

    params = os.path.join(pattern, 'params.yml')
    if os.path.exists(params):
        # print the params.yml file to screen
        print(f'Content of {params}')
        import ruamel.yaml
        from xmen.utils import recursive_print_lines
        with open(os.path.join(params), 'r') as params_yml:
            params = ruamel.yaml.load(params_yml, ruamel.yaml.RoundTripLoader)
            lines = recursive_print_lines(params)
            for l in lines:
                print(l)
    else:
        # print(args.pattern)
        # global_exp_manager = GlobalExperimentManager()
        config = Config()
        paths = config.filter(pattern)
        params = load_params(paths)
        if args.list:
            if args.pattern is None:
                pattern += '.*'
                args.pattern = pattern
            from xmen.list import notebook_display, args_to_filters
            args.filters += ['_notes', '_created|_timestamps', '_purpose', '_version']
            notebook_display(params, *args_to_filters(args))
        elif args.interval is None:
            if args.pattern is None:
                pattern += '.*'
                args.pattern = pattern
            from xmen.utils import load_params
            from xmen.list import args_to_filters, visualise_params
            data_frame, root = visualise_params(params, *args_to_filters(args))
            print(f'Roots relative to {root}')
            print(data_frame)
        else:
            if args.pattern is None:
                pattern += '.*'
                args.pattern = f'{config.local_user}@{config.local_host}:{pattern}'
            from xmen.utils import load_params
            from xmen.list import interactive_display, test
            test(args)
            # interactive_display(stdscr, args)

list_parser.set_defaults(func=_curses_list)

#######################################################################################################################
#  main
#######################################################################################################################
def setup():
    from xmen.config import Config
    print(DESCRIPTION)
    Config().setup()


def invalid_experiment_root_hook(exctype, value, traceback):
    import sys
    if exctype == InvalidExperimentRoot:
        print("ERROR: The current repository is not a valid experiment root. Try init?")
    else:
        sys.__excepthook__(exctype, value, traceback)


def main():
    import sys
    sys.excepthook = invalid_experiment_root_hook
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        setup()
        # parser.print_help()


# Enable command line interface
if __name__ == "__main__":
   main()
