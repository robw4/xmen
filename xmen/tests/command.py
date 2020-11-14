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
"""A script to test the command line interface:
> python -m xmen.tests.command.py --update "{w: 10, h: 20}" --execute /tmp/initial
Updating parameters {'w': 10, 'h': 20}
The experiment state inside run is running
python -m xmen.tests.command.py --update /tmp/initial/params.yml --execute /tmp/from-params
python -m xmen.tests.command.py --execute /tmp/from-params/params.yml
"""

import sys
# sys.path.append('/home/robw/xmen/python')
# from xmen.tests.experiment import AnExperiment
from xmen.tests.experiment import AnExperiment

if __name__ == '__main__':
    AnExperiment().main()
