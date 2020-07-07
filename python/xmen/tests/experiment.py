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
from xmen.experiment import Experiment
import os
import time
from typing import List


class BaseExperiment(Experiment):
    """A basic python experiment demonstrating the features of the xmen api."""

    # Parameters are defined as attributes in the class body with the
    # @p identifier

    t = 'cat'    # @p
    w = 3        # @p parameter w has a help message whilst t does not
    h: int = 10  # @p h declared with typing is very concise and neat

    # Parameters can also be defined in the __init__ method
    def __init__(self, *args, **kwargs):
        super(BaseExperiment, self).__init__(*args, **kwargs)

        self.a: str = 'h'  # @p A parameter
        self.b: int = 17   # @p Another parameter

        # Normal attributes are still allowed
        self.c: int = 5    # This is not a parameter


class AnotherExperiment(BaseExperiment):
    m: str = 'Another value'  # @p Multiple inheritance example
    p: str = 'A parameter only in Another Experiment '  # @p


class AnExperiment(BaseExperiment):
                    #     |
                    # Experiments can inherit from other experiments
                    # parameters are inherited too
    """An experiment testing the xmen experiment API. The __docstring__ will
    appear in both the docstring of the class __and__ as the prolog in the
    command line interface."""

    # Feel free to define more parameters
    x: List[float] = [3., 2.]  # @p Parameters can be defined cleanly as class attributes
    y: float = 5  # @p This parameter will have this
    # Parameters can be overridden
    a: float = 0.5  # a's default and type will be changed. Its help will be overridden
    b: int = 17  # @p b's help will be changed

    m: str = 'Defined in AnExperiment'  # @p m is defined in AnExperiment

    def run(self):
        # Experiment execution is defined in the run method
        print(f'The experiment state inside run is {self.status}')

        # recording messaging is super easy
        self.message({'time': time.time()})

        # Each experiment has its own unique directory. You are encourage to
        # write out data accumulated through the execution (snapshots, logs etc.)
        # to this directory.
        with open(os.path.join(self.directory, 'logs.txt'), 'w') as f:
            f.write('This was written from a running experiment')

    def debug(self):
        self.a = 'In debug mode'
        return self

    @property
    def h(self):
        return 'h has been overloaded as propery and will no longer' \
               'considered as a parameter'


if __name__ == '__main__':
    # Documentation is automatically added to the class
    help(AnExperiment)

    # to run an experiment first we initialise it
    # the experiment is initialised in 'default' status
    print('\nInitialising')
    print('---------------------------')
    exp = AnExperiment()
    print(exp)

    # whilst the status is default the parameters of the
    # experiment can be changed
    print('\nConfiguring')
    print('------------ ')
    exp = AnExperiment()
    exp.a = 'hello'
    exp.update({'t': 'dog', 'w': 100})

    # Note parameters are copied from the class during
    # instantiation. This way you don't need to worry
    # about accidentally changing the mutable class
    # types across the entire class.
    exp.x += [4.]
    print(exp)
    assert AnExperiment.x == [3., 2.]
    # If this is not desired (or neccessary) initialise
    # use exp = AnExperiment(copy=False)

    # Experiments can inheret from multiple classes:
    class MultiParentsExperiment(AnotherExperiment, AnExperiment):
        pass
    print('\nMultiple Inheritance')
    print('----------------------')
    print(MultiParentsExperiment())
    print('\n Parameters defaults, helps and values are '
          'inherited according to python method resolution order '
          '(i.e left to right). Note that m has the value '
          'defined in Another Experiment')

    print('\nRegistering')
    print('-------------')
    # Before being run an experiment needs to be registered
    # to a directory
    exp.register('/tmp/an_experiment', 'first_experiment',
                 purpose='A bit of a test of the xmen experiment api')
    print(exp, end='\n')
    print('\nGIT, and system information is automatically logged\n')

    # The parameters of the experiment can no longer be changed
    try:
        exp.a = 'cat'
    except AttributeError:
        print('Parameters can no longer be changed!', end='\n')
        pass

    # An experiment can be run either by...
    # (1) calling it
    print('\nRunning (1)')
    print('-------------')
    exp()

    # (2) using it as a context. Just define a main
    #     loop like you normally would
    print('\nRunning (2)')
    print('-------------')
    with exp as e:
        # Inside the experiment context the experiment status is 'running'
        print(f'Once again the experiment state is {e.status}')
        # Write the main loop just as you normally would"
        # using the parameters already defined
        results = dict(sum=sum(e.x), max=max(e.x), min=min(e.x))
        # Write results to the expeirment just as before
        e.message(results)

    # All the information about the current experiment is
    # automatically saved in the experiments root directory
    # for free
    print(f'\nEverything ypu might need to know is logged in {exp.directory}/params.yml')
    print('-----------------------------------------------------------------------------------------------')
    print('Note that GIT, and system information is automatically logged\n'
          'along with the messages')
    with open('/tmp/an_experiment/first_experiment/params.yml', 'r') as f:
        print(f.read())

    print('\nRegsitering and Configuring from the command line')
    print('---------------------------------------------------')
    # Alternatively configuring and registering an experiment
    # can be done from the command line as:
    exp = AnExperiment()
    args = exp.parse_args()

    print('\nRegsitering, Configuring and running in a single line!')
    print('--------------------------------------------------------')
    # Or alternatively
    # the experiment can be _configured_, _registered_
    # and _run_ by including a single line of code:
    AnExperiment().main()
    # See ``python xmen.tests.experiment --help`` for
    # more information
