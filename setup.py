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

import setuptools
from setuptools import setup

setup(
    name='xmen',
    version='0.3.0',
    description='Experiment management libraries for fast, reproducible, remote and local experimentation',
    author='Rob Weston',
    author_email='robw@robots.ox.ac.uk',
    entry_points={
        'console_scripts':
            [
                # 'xmen=xmen.app._xmen:main',
                'xmen=xmen.app._xgent:main'
            ]},
    install_requires=[
            'ruamel.yaml',
            'pandas>=1.1.1',
            'GitPython>=3.1.7',
            'psutil',
            'GPUtil',
            'tabulate'
    ],
    packages=setuptools.find_packages())
