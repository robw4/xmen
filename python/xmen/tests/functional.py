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
from typing import Tuple


class Monitor(Experiment):
    x: Tuple[float, float] = (3., 2.)  # @p Parameters can be defined cleanly as class atrributes
    y: float = 5                       # @p This parameter will also be available


if __name__ == '__main__':
    m = Monitor()
    m.parse_args()

    with m:



        print('(3) ----------------------')
        print(m)
        print()
        m.message(dict(sum=sum(m.x), max=max(m.x), min=min(m.x)))

    print('(4) ----------------------')
    print(m)
    print()
