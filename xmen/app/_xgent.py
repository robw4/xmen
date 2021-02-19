#!/usr/bin/env python
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

from xmen.config import GlobalExperimentManager
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
    import subprocess
    global_exp_manager = GlobalExperimentManager()
    if args.list is not None:
        print(f'The following python experiments are currently linked')
        for k, v in global_exp_manager.python_experiments.items():
            print(f'{k}: {v}')
    if args.add is not None:
        try:
            with global_exp_manager as config:
                config.add_experiment(*args.add)
            print(f'Added experiment {args.add[-1]} from module {args.add[-2]}')
        except:
            print(f'ERROR: failed to add experiment {args.add[-1]} from module {args.add[-2]}')
    if args.remove is not None:
        with global_exp_manager as config:
            if args.remove in config.python_experiments:
                path = config.python_experiments.pop(args.remove)
                if '.xmen' in path:
                    os.remove(path)
                print(f'Successfully removed {args.remove}!')
    if args.name:
        import subprocess
        if args.name[0] not in global_exp_manager.python_experiments:
            print(f'No experiments found matching {args.name[0]}')
            exit()
        args = [global_exp_manager.python_experiments[args.name[0]]] + args.flags
        subprocess.call(args)



python_parser.set_defaults(func=_python)

#######################################################################################################################
#  config
#######################################################################################################################
config_parser = subparsers.add_parser('config', help='View / edit the global configuration')
config_parser.add_argument('--disable_prompt', action='store_false',
                           help='Turn purpose prompting off', default=None)
config_parser.add_argument('--clean', action='store_false',
                           help='Remove experiments that have been corrupted from the global configuration.',
                           default=None)
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

config_parser.add_argument('--update_meta', default=None, action='store_true',
                           help='Update meta information in each experiment (both defaults.yml and params.yml). '
                                'WARNING: Overwrites information in the params.yml or defaults.yml')
config_parser.add_argument('-H', '--header', type=str, help='Update the default header used when generating experiments'
                                                            ' to HEADER (a .txt file)')
config_parser.add_argument('--list', default=None, help='Display the current configuration', action='store_false')
config_parser.add_argument('--host',  type=str, help='Update the default host used by xmen')
config_parser.add_argument('--port', default=2030, type=int, help='Update the default port used by xmen')


def _config(args):
    with GlobalExperimentManager() as config:
        if args.disable_prompt is not None:
            config.prompt_for_message = False
        elif args.enable_prompt is not None:
            config.prompt_for_message = True

        if args.disable_save_conda is not None:
            config.save_conda = False
        elif args.enable_save_conda is not None:
            config.save_conda = True

        if args.disable_stdout_to_txt is not None:
            config.redirect_stdout = False
        elif args.enable_stdout_to_txt is not None:
            config.redirect_stdout = True

        if args.disable_requeue is not None:
            config.requeue = False
        elif args.enable_requeue is not None:
            config.requeue = True

        if args.update_meta is not None:
            config.update_meta()
        if args.header is not None:
            if os.path.exists(args.header):
                config.header = open(args.header, 'r').read()
            else:
                config.header = args.header
        if args.host:
            config.host = args.host
        if args.port:
            args.port = args.port

        if args.list is not None:
            print(config)
        if args.clean is not None:
            print('Cleaning config. If you have a lot of experiment roots registered then this might take a second...')
            config.clean()


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
                         help='A name of a python experiment registered with the global configuration.')
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
note_parser.add_argument('message', help='Add note to experiment set')
note_parser.add_argument('-r', '--root', metavar='DIR', default='',
                         help='Path to the root experiment folder. If None then the current work directory will be '
                              'used')
note_parser.add_argument('-d', '--delete', default='', action='store_true',
                         help='Delete the note corresponding to message.')


def _note(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.note(args.message, args.delete)


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


def _reset(args):
    experiment_manager = ExperimentManager(args.root)
    experiment_manager.reset(args.experiments)


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


relink_parser.set_defaults(func=_relink)

#######################################################################################################################
#  list
#######################################################################################################################
list_parser = subparsers.add_parser('list', help='list experiments to screen')
list_parser.add_argument('pattern', type=str, help='List experiments which match pattern.', default=[''], nargs='*')
list_parser.add_argument('-p', '--param_match', type=str, default=None,
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
list_parser.add_argument('-m', '--display_messages', default=None,
                         const='^last$|^e$|^s$|^wall$|^end$|^next$|^.*step$|^.*load$',
                         type=str, help="Display messages for each experiment", nargs='?')
list_parser.add_argument('-M', '--display_meta', default=None,
                         const='^root$|^name$|^mac$|^host$|^user$|^home$',
                         type=str,
                         help="Display meta information for each experiment. The regex "
                              "'^root$|^name$|^mac$|^host$|^user$|^home$' gives basic meta information "
                              "logged with every experiment. Other information is separated into groups including "
                              "'network.*', 'gpu.*', 'cpu.*', 'system.*', 'virtual.*', 'swap.*' ", nargs='?')
list_parser.add_argument('-l', '--list', action='store_true', default=None,
                         help="Display as list and not a table")
list_parser.add_argument('--load_defaults', action='store_true', default=None,
                         help='Infer parameters from defaults.yml and overides instead of params.yml. Potentially '
                              'faster but no messages are available.')
list_parser.add_argument('--max_width', default=60, help='The maximum width of an individual collumn. '
                                                           'If None then will print for ever', type=int)
list_parser.add_argument('--max_rows', default=None, help='Display tables with this number of rows.', type=int)
list_parser.add_argument('--csv', action='store_true', help='Display the table as csv.', default=None)
list_parser.add_argument('-i', '--interval', type=float, default=None, const=1., nargs='?',
                         help='If set then the table will be updated every this number of seconds')


def _curses_list(args):
    import curses
    if args.interval is not None:
        try:
            args.param_match = "^$"
            curses.wrapper(_list, args)
        except NotEnoughRows:
            print('WARNING: Not enough rows to display the table in interactive mode. Find a bigger terminal')
        except KeyboardInterrupt:
            pass
    else:
        _list(None, args)


def updates_client(q, args):
    from xmen.utils import commented_to_py
    import os
    import socket
    import time
    # get the hostname
    import struct
    import ruamel.yaml
    from xmen.experiment import IncompatibleYmlException, HOST, PORT

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((HOST, PORT))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(100)

    while True:
        try:
            updates = {}
            last = time.time()
            while time.time() - last < args.interval:
                conn, address = server_socket.accept()
                # print("Connection from: " + str(address))
                length = conn.recv(struct.calcsize('Q'))
                if length:
                    length, = struct.unpack('Q', length)
                    # print('length = ', length)
                    params = conn.recv(length, socket.MSG_WAITALL).decode()
                    yaml = ruamel.yaml.YAML()
                    try:
                        params = yaml.load(params)
                    except:
                        raise IncompatibleYmlException
                    params = {k: commented_to_py(v) for k, v in params.items()}
                    # print(params)
                    updates[os.path.join(params['_root'], params['_name'])] = params
                conn.close()  # close the connection
            if updates:
                q.put((updates, last))
        except Exception as m:
            q.put((None, m))
            with open('/data/engs-robot-learning/kebl4674/usup/tmp/xmen-error-log.txt', 'w') as f:
                f.write(m)
            break


class NotEnoughRows(Exception):
    pass


def interactive_display(stdscr, results, args):
    import curses
    import multiprocessing as mp
    import queue
    import time
    import copy

    global root

    stdscr.refresh()
    global_experiment_manager = GlobalExperimentManager()
    rows, cols = stdscr.getmaxyx()

    if rows < 9:
        raise NotEnoughRows

    pos_x, pos_y = 0, 0
    last_time = time.time()
    meta = []
    message = []
    params = []
    user_message = []
    if args.display_meta is None:
        args.display_meta = ''

    args.display_messages = ''
    args.param_match = "^$"

    # initialise colours
    if curses.has_colors():
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

    WHITE = curses.color_pair(1)
    RED = curses.color_pair(2)
    CYAN = curses.color_pair(3)
    YELLOW = curses.color_pair(4)
    GREEN = curses.color_pair(5)
    MAGNETA = curses.color_pair(6)

    global window
    window = curses.newwin(rows, cols, 0, 0)

    def generate_table(results, args, return_root=False):
        import pandas as pd
        data_frame, root = global_experiment_manager.find_to_dataframe(
            copy.deepcopy(results),
            verbose=args.verbose,
            display_git=args.display_git,
            display_purpose=args.display_purpose,
            display_date=args.display_date,
            display_messages=args.display_messages if not args.display_messages == '' else False,
            display_meta=args.display_meta,
            display_status=args.display_status,
            display_params=args.param_match if not args.param_match == '' else '^$')
        return data_frame, root

    def update_meta(string):
        if string in meta:
            meta.remove(string)
        else:
            meta.append(string)
        args.display_meta = '|'.join(meta)

    def update_message(string=None):
        if string is not None:
            if string in message:
                message.remove(string)
            else:
                message.append(string)
        args.display_messages = '|'.join(message + user_message)

    def update_params(string=None):
        if string is not None:
            if string in params:
                params.remove(string)
            else:
                params.append(string)
        args.param_match = '|'.join(params)

    def display_row(i, pad, table=None, x=None):
        from tabulate import tabulate
        if x is None:
            x = tabulate(table, headers='keys', tablefmt='github').split('\n')
        x0 = x[0]
        xi = x[i]
        # stdscr.addstr(i + 2, 1, xx)
        status = [ii for ii, hh in enumerate(x0.split('|')) if hh.strip() == 'status']
        if i == 0:
            xi = xi.replace('|', '')
            pad.addstr(i, 1, xi, curses.A_BOLD)
        else:
            componenents = xi.split('|')
            if status:
                for j, c in enumerate(componenents):
                    if j == status[0]:
                        col = {
                            'registered': WHITE, 'timeout': YELLOW,
                            'running': GREEN, 'error': RED}
                        cc = col.get(c.strip(), None)
                        if cc is None:
                            offset = sum(len(_) for _ in componenents[:j]) + 1
                            pad.addstr(i, offset, c)
                        else:
                            pad.addstr(i, sum(len(_) for _ in componenents[:j]) + 1, c, cc)
                    else:
                        pad.addstr(i, sum(len(_) for _ in componenents[:j]) + 1, c)
            else:
                pad.addstr(i, 1, ''.join(componenents))

    def display_table(table, root):
        global pad, window
        import curses
        rows, cols = stdscr.getmaxyx()
        # generate
        import curses
        from curses.textpad import Textbox
        # if rows is None:
        window.erase()

        from tabulate import tabulate
        x = tabulate(table, headers='keys', tablefmt='github').split('\n')
        x.pop(1)

        # display_time(window)

        # if rows is None:
        import curses
        window.addstr(0, 0, f'Experiments matching {args.pattern}')
        window.addstr(1, 0, f'Roots relative to {root}')
        window.addstr(2, 0, f'Last update @ {time.time() - last_time:.3f} seconds ago', curses.A_DIM)

        import curses

        window.addstr(rows - 3, 0, 'Toggles:', curses.A_BOLD)
        window.addstr(rows - 2, 0, f'meta = {meta}, messages={message + user_message}, params={params}')
        window.addstr(rows - 1, 0,
                      'd=date s=status g=git p=purpose m=specify-message t=monitor-message M=meta G=gpu S=slurm c=cpu n=network v=virtual w=swap o=os D=disks')
        if len(x) > rows - 5:
            filler = '|'.join([' ...'.ljust(len(xx), ' ') if xx else ''
                                 for ii, xx in enumerate(x[1].split('|'))])
            filler = filler.replace('...', '   ', 1)
            x = [x[0], filler] + x[-(rows - 3):]

        n = 10
        pad = curses.newpad(len(x), len(x[0]))
        for i, xx in enumerate(x):
            display_row(i, pad, x=x)

        # for y in range(0, n * rows - 1):
        #     for x in range(0, n * cols - 1):
        #         pad.addch(y, x, ord('a') + (x * x + y * y) % 26)
        window.noutrefresh()
        pad.noutrefresh(pos_x, 0, 3, 0, rows - 5, cols - 1)
        curses.doupdate()
        pass

    def visualise_results(results, args):
        data_frame, root = generate_table(results, args)
        display_table(data_frame, root)

    visualise_results(results, args)
    rows, cols = stdscr.getmaxyx()

    try:
        import multiprocessing
        q = mp.Queue(maxsize=1)
        p = multiprocessing.Process(target=updates_client, args=(q, args))
        p.start()

        while True:
            try:
                updates, last_time = q.get(False)
                if updates is None:
                    raise RunTimeError(last_time)

                paths = [os.path.join(results['_root'][i], results['_name'][i])
                         for i in range(len(results['_root']))]
                idx = [(i, n) for i, n in enumerate(paths) if n in updates]
                if idx:
                    for i, n in idx:
                        for k, v in updates[n].items():
                            results[k][i] = v
                    visualise_results(results, args)

            except queue.Empty:
                pass

            import sys
            import time

            if stdscr is not None:
                stdscr.timeout(1000)
                c = stdscr.getch()
                # if c == curses.KEY_RIGHT:
                #     pos_x += rows
                #     visualise_results(results, args)
                # if c == curses.KEY_LEFT:
                #     pos_x -= rows
                #     visualise_results(results, args)

                if c == ord('d'):
                    args.display_date = not args.display_date
                    visualise_results(results, args)
                if c == ord('g'):
                    args.display_git = not args.display_git
                    visualise_results(results, args)
                if c == ord('s'):
                    args.display_status = not args.display_status
                    visualise_results(results, args)
                if c == ord('p'):
                    args.display_purpose = not args.display_purpose
                    visualise_results(results, args)
                if c == ord('M'):
                    update_meta('slurm_job|^root$|^name$|^mac$|^host$|^user$|^home$')
                    visualise_results(results, args)
                if c == ord('v'):
                    update_meta('virtual.*')
                    visualise_results(results, args)
                if c == ord('w'):
                    update_meta('swap.*')
                    visualise_results(results, args)
                if c == ord('o'):
                    update_meta('system.*')
                    visualise_results(results, args)
                if c == ord('D'):
                    update_meta('disks.*')
                    visualise_results(results, args)
                if c == ord('c'):
                    update_meta('cpu.*')
                    visualise_results(results, args)
                if c == ord('G'):
                    update_meta('gpu.*')
                    visualise_results(results, args)
                if c == ord('n'):
                    update_meta('network.*')
                    visualise_results(results, args)
                if c == ord('S'):
                    update_meta('slurm.*')
                    visualise_results(results, args)
                if c == ord('t'):
                    update_message('last|^e$|^s$|^wall$|^end$|^next$|^m_step$|^m_load$')
                    visualise_results(results, args)

                if c == ord('m'):
                    from curses.textpad import Textbox, rectangle
                    rows, cols = stdscr.getmaxyx()
                    win = curses.newwin(1, cols, rows - 1, 0)
                    stdscr.addstr(rows - 3, 0, 'Message:                                          ', curses.A_BOLD)
                    stdscr.addstr(rows - 2, 0, '(ctrl-h=backspace, "" to delete user message)                      ')
                    stdscr.addstr(rows - 1, 9, '' * 15)
                    # rectangle(stdscr, rows - 1, 0, rows, cols - 1)
                    stdscr.refresh()
                    text_box = Textbox(win)
                    text = text_box.edit().strip()
                    if text != '':
                        user_message += [text]
                    else:
                        user_message = []
                    update_message()
                    visualise_results(results, args)

                if c == ord('P'):
                    from curses.textpad import Textbox, rectangle
                    rows, cols = stdscr.getmaxyx()
                    win = curses.newwin(1, cols, rows - 1, 0)
                    stdscr.addstr(rows - 3, 0, 'Params:                                          ', curses.A_BOLD)
                    stdscr.addstr(rows - 2, 0, '(ctrl-h=backspace, "" to delete user message)                      ')
                    stdscr.addstr(rows - 1, 9, '' * 15)
                    # rectangle(stdscr, rows - 1, 0, rows, cols - 1)
                    stdscr.refresh()
                    text_box = Textbox(win)
                    text = text_box.edit().strip()
                    if text != '':
                        params += [text]
                    else:
                        params = []
                    update_params()
                    visualise_results(results, args)
    except KeyboardInterrupt:
        p.terminate()
        raise KeyboardInterrupt


            # elif c == curses.KEY_UP:
            #     mypad_pos -= 1
            #     mypad.refresh(mypad_pos, 0, 5, 5, 10, 60)


def notebook_display(results):
    from xmen.utils import recursive_print_lines
    notes = []
    for i, (r, e, p, n, d, t, v) in enumerate(
            zip(*[results[j] for j in ('_root', '_experiments', '_purpose',
                                       '_notes', '_created', '_type', '_version')])):
        k = 5
        i = str(i)
        note = ' ' * (k // 2 - len(str(i))) + str(i) + ' ' * (k // 2 - 1) + r + '\n' + ' ' * k
        if len(e) > 0:
            note += ('\n' + ' ' * k).join(['|- ' + ee[len(r) + 1:] for ee in e]) + '\n' + ' ' * k
        note += 'purpose: ' + p + '\n' + ' ' * k
        note += 'created: ' + d + '\n' + ' ' * k
        if len(v) > 0:
            note += 'version: ' + '\n' + ' ' * (k + 2) + ('\n' + ' ' * (k + 2)).join(
                [l for l in recursive_print_lines(v)])
        if len(n) > 0:
            note += '\n' + ' ' * k + 'notes: ' + '\n' + ' ' * (k + 2)
            note += ('\n' + ' ' * (k + 2)).join(
                ['\n'.join(textwrap.wrap(nn, width=1000, subsequent_indent=' ' * (k + 3))) for i, nn in
                 enumerate(n)])
        notes += [note]
    print('\n'.join(notes))


def _list(stdscr, args):
    import pandas as pd

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_colwidth', args.max_width)
    pd.set_option('display.max_rows', args.max_rows)
    if len(args.pattern) > 1:
        print(f'ERROR: Only one pattern may be passed but got {args.pattern}')
    pattern = os.path.abspath(os.path.expanduser(args.pattern[0]))

    if pattern == '':
        pattern = os.path.join(os.getcwd())

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
        if args.pattern[0] == '':
            pattern += '*'
            args.pattern = pattern

        global_exp_manager = GlobalExperimentManager()
        if args.list:
            results, last = global_exp_manager.find(
                mode='set', pattern=args.pattern, param_match=[args.param_match], types_match=args.type_match,
                load_defaults=True)
            notebook_display(results)
        elif args.interval is None:
            print('args.param_match', args.param_match)
            # print(pattern, end='\r')
            import copy
            import time
            results = global_exp_manager.find(
                mode='all',
                pattern=args.pattern,
                param_match=args.param_match,
                types_match=args.type_match,
                load_defaults=args.load_defaults)
            data_frame, root = global_exp_manager.find_to_dataframe(
                copy.deepcopy(results),
                verbose=args.verbose,
                display_git=args.display_git,
                display_purpose=args.display_purpose,
                display_date=args.display_date,
                display_messages=args.display_messages if not args.display_messages == '' else False,
                display_meta=args.display_meta,
                display_status=args.display_status,
                display_params=args.param_match if not args.param_match == '' else '^$')
            print(data_frame)
        else:
            results = global_exp_manager.find(
                mode='all',
                pattern=args.pattern,
                param_match=[".*"],
                types_match=args.type_match,
                load_defaults=args.load_defaults)
            interactive_display(stdscr, results, args)




list_parser.set_defaults(func=_curses_list)

#######################################################################################################################
#  main
#######################################################################################################################


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
        parser.print_help()


# Enable command line interface
if __name__ == "__main__":
   main()
