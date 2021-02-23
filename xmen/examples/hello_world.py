"""Basic Examples.
"""
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
import xmen


@xmen.autodoc
def hello_world(
    root: xmen.Root,   # experiments are assigned a root before being executed
    a: str = 'Hello',  # the first
    # argument
    b: str = 'World',   # the second argument
):
    """A hello world experiment designed to demonstrate
    defining experiments through the functional experiment api"""
    print(f'{a}  {b}')

    ...     # whatever other experiment code you want

    with open(root.directory + '/out.txt', 'w') as f:
        f.write(f'{a} {b}')
    root.message({'a': a, 'b': b})


class HelloWorld(xmen.Experiment):
    """A hello world experiment designed to demonstrate
    defining experiments through the class experiment api"""
    # Parameters
    a: str = 'Hello'  # @p The first argument
    b: str = 'World'  # @p The second argument

    def run(self):
        print(f'{self.a} {self.b}!')
        self.message({'a': self.a, 'b': self.b})


if __name__ == '__main__':
    # optionally expose the command line interface if you
    # would like to run the experiment as a python script
    from xmen.functional import functional_experiment
    # generate experiment from function definition if defined
    # using the functional experiment (this step is not needed if
    # the experiment is defined as a class)
    Exp = functional_experiment(hello_world)
    # every experiment inherits main() allowing the experiment
    # to be configured and run from the command line.
    Exp().main()
    # try...
    # >> python -m xmen.examples.hello_world --help
