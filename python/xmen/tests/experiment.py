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


class AnExperiment(Experiment):
    def __init__(self):
        """A basic python experiment demonstrating the features of the xmen api."""
        super(AnExperiment, self).__init__()
        # Lets define some parameters. This is done using the @p identifier.
        # By using Python typing you are able to define the name, type, default
        # And help for a parameter all in a single line. The TypedMeta metaclass
        # automatically takes care of identifying parameters and managing
        # doucmentation even before the class is instantiated!
        self.a: str = 'h'  # @p A parameter
        self.b: int = 17  # @p Another parameter

        # Normal attributes are still allowed
        self.c: int = 5  # This is not a parameter

    def run(self):
        # The execution of an experiement is created by overloading the run method.
        print(f'a = {self.a}, b = {self.b}')

        # In order to be executed an experiment must be linked with a directory
        # in which a params.yml file is created recording designed to allow the
        # execution environement to be reproduced. The experiments status is updated
        # to running within the experiment.
        print(f'The experiment state inside run is {self.status}')

        # The experiment class ``message`` facilitates experiment communication.
        # The time is now added to the _messages dictionary in params.yml.
        self.message({'time': time.time()})

        # Each experiment has its own unique directory. You are encourage to
        # write out data acumulated through the execution (snapshots, logs etc.)
        # to this directory.
        with open(os.path.join(self.directory, 'logs.txt'), 'w') as f:
            f.write('This was written from a running experiment')


if __name__ == '__main__':
    # Now we expose the command line interface
    exp = AnExperiment()
    args = exp.parse_args()
    exp.main(args)
