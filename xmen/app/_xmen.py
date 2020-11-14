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

from xmen.config import GlobalExperimentManager
from xmen.manager import InvalidExperimentRoot

DESCRIPTION = [r'||||||||||||||||||||||||| WELCOME TO ||||||||||||||||||||||||||',
               r'||                                                           ||',
               r'||    \\\  ///  |||\\        //|||  |||||||||  |||\\   |||   ||',
               r'||     \\\///   |||\\\      ///|||  |||        |||\\\  |||   ||',
               r'||      ||||    ||| \\\    /// |||  ||||||     ||| \\\ |||   ||',
               r'||     ///\\\   |||  \\\  ///  |||  |||        |||  \\\|||   ||',
               r'||    ///  \\\  |||   \\\///   |||  |||||||||  |||   \\|||   ||',
               r'||                                                           ||',
               r'|||||||||||| FAST - REPRODUCIBLE - EXPERIMENTATION ||||||||||||']

#######################################################################################################################
#  py
#######################################################################################################################
py_parser = argparse.ArgumentParser(
    prog='xman',
    description='\n'.join(DESCRIPTION),
    formatter_class=argparse.RawTextHelpFormatter
)

py_parser.add_argument('name', help='The name of the experiment to run', nargs='*', default=None)
py_parser.add_argument('--list', '-l', action='store_true', default=None, help='List available python experiments')
py_parser.add_argument('--add', default=None, metavar='MODULE NAME',
                       help='Add a python Experiment class or run script (it must already be on PYTHONPATH)', nargs=2)
py_parser.add_argument('--remove', '-r',  help='Remove a python experiment (passed by Name)')
py_parser.add_argument('flags', help='Python flags (pass --help for more info)', nargs=argparse.REMAINDER, default=[])


def _py(args):
    import subprocess

    global_exp_manager = GlobalExperimentManager()
    if args.list is not None:
        for k, v in global_exp_manager.python_experiments.items():
            print(f'{k}: {v}')
    elif args.add is not None:
        with global_exp_manager as config:
            config.add_experiment(*args.add)
    elif args.remove is not None:
        with global_exp_manager as config:
            if args.remove in config.python_experiments:
                path = config.python_experiments.pop(args.remove)
                if '.xmen' in path:
                    os.remove(path)
    else:
        if len(args.name) == 0:
            py_parser.print_help()
            return
        if args.name[0] not in global_exp_manager.python_experiments:
            print(f'No experiments found matching {args.name[0]}')
            exit()
        args = [global_exp_manager.python_experiments[args.name[0]]] + args.flags
        subprocess.call(args)


py_parser.set_defaults(func=_py)

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
    args = py_parser.parse_args()
    args.func(args)


# Enable command line interface
if __name__ == "__main__":
   main()