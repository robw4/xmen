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
import sys
import re

import xmen.manager
from xmen.utils import get_meta


class NoMatchException(Exception):
    def __init__(self, path):
        """Experiment at ``path`` does not contain a match"""
        super(NoMatchException, self).__init__()
        self.path = path


class GlobalExperimentManager(object):
    """A helper class used to manage global configuration of the Experiment Manager"""
    def __init__(self):
        self.python_experiments = {}   # A dictionary of paths to python modules compatible with the experiment api
        self.python_paths = []         # A list of python paths needed to run each module
        self.prompt_for_message = True
        self.experiments = {}          # A list of all experiments registered with an Experiment Manager
        self.meta = get_meta()
        self.header = ''
        self._dir = os.path.join(os.getenv('HOME'), '.xmen')

        if not os.path.isdir(self._dir):
            os.makedirs(self._dir)
            self._to_yml()
        else:
            self._from_yml()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._to_yml()

    def _to_yml(self):
        """Save the current config to an ``config.yaml``"""
        import ruamel.yaml
        params = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        with open(os.path.join(self._dir, 'config.yml'), 'w') as file:
            ruamel.yaml.dump(params, file, Dumper=ruamel.yaml.RoundTripDumper)

    def _from_yml(self):
        """Load the experiment config from a ``config.yml`` file"""
        with open(os.path.join(self._dir, 'config.yml'), 'r') as file:
            import ruamel.yaml
            from ruamel.yaml.comments import CommentedMap
            params = ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)
            to_yml = False
            for k, v in params.items():
                if k == 'experiments' and len(v) != 0 and type(v) is not CommentedMap:
                    experiments = {}
                    to_yml = True
                    for vv in v:
                        em = xmen.manager.ExperimentManager(root=vv, headless=True)
                        experiments[em.root] = {
                            'created': em.created, 'type': em.type, 'purpose': em.purpose, 'notes': em.notes}
                    self.__dict__[k] = experiments
                else:
                    self.__dict__[k] = v
            if 'meta' not in params:
                to_yml = True
        if to_yml:
            self._to_yml()

    def update_meta(self):
        for p in self.experiments:
            em = xmen.manager.ExperimentManager(p)
            em.update_meta()

    def clean(self):
        """Iteratively search through experiment sets and remove any that no longer exist.

        Note:
            This is performed by checking that experiment folder is a valid set using
            using ``ExperimentManager(p).check_initialised()``

        """
        corrupted = []
        for p in self.experiments:
            try:
                em = xmen.manager.ExperimentManager(p)
                em.check_initialised()
            except:
                corrupted.append(p)
        if len(corrupted) != 0:
            print('The following experiments were removed from the global configuration... ')
            for c in corrupted:
                self.experiments.pop(c)
                print('  - ' + c)

    def __repr__(self):
        """The current configuration as a string"""
        string = f'Prompt for Message: {self.prompt_for_message}\n'
        string += '\n'
        string += f'Python Path:\n'
        for p in self.python_paths:
            string += f'  - {p}\n'
        string += '\n'
        string += 'Python Experiments:\n'
        for k, v in self.python_experiments.items():
           string += f'  - {k}: {v}\n'
        string += '\n'
        string += 'Header:\n'
        string += self.header + '\n'
        string += '\n'
        string += 'Meta:\n'
        for k, m in self.meta.items():
            string += f'  - {k}: {m}\n'
        string += '\n'
        string += 'Experiments:\n'
        for k, e in self.experiments.items():
            e = dict(e)
            string += f'  - {e["created"]}: {k}\n'

        return string

    def find(self, mode='all', pattern="*", param_match=None, types_match=None, load_defaults=False, missing_entry=''):
        """Search through all experiments in the globally filtering by experiment location, parameters, and type.

        Args:
            mode (str): If 'all' then all experiments for each set will be concatenated and summarised individually
                whilst if 'set' then experiments will be summarised and searched for based on the default parameters.
            pattern (str): A unix glob pattern of experiments to consider
            param_match (list): A list of strings providing a condition to filter experiments by their parameters.
                Compatible string formats are "regex", "regex op val" or "val1 op1 regex op2 val2" where op is in
                ``[<=,==,>=,<,>,!=]``. If just a regex is supplied then only experiments with parameters matching the
                regex will be returned. If an op is supplied then only experiments with all parameters which match the
                regex and satisfy the condition will be returned.
            types_match (list): A list of types of experiments to search for
            load_defaults (bool): If True then ``defaults.yml`` files will be loaded and the parameters of each
                experiment inferred from the overides defined in the ``experiment.yml`` file for each experiment set.
                This is potentially faster (less io) but means that the messages for each experiment will not be
                returned.
            missing_entry: Assign this parameter to missing values (see notes).

        Returns:
            matches (dict): A dict with possible keys ``['_root', '_purpose', '_type', '_notes',  '_name', '_created',
                '_version', '_status', '_messages', '_experiments', *param_keys]`` where  ``param_keys`` correspond to
                any parameters that satisfy the conditions in param_match. The keys
                ``['_root', '_purpose', '_type', '_notes', '_created']`` will always be present,
                ``['_name', '_version']`` will be added if ``load_defaults==True`` and ``['_name', '_version',
                '_status', '_messages]`` if ``load_defaults==False``. If [mode == 'set'] then
                the key  '_experiments' will also be added (giving a list of paths to the experiments in each set).
                The created dates in this case correspond to the date the set was created
                (given in defaults.yml).
            special_keys (dict): A list of keys starting with '_' present in the dictionary.

        Note:
            Due to the use of regex and conditions it is possible for parameters to match in one experiment which are
                not present in another. In this case missing parameters are given a value equal to ``missing_string``.
        """
        import fnmatch
        import re
        def _comparison(*args):
            arg = ''.join(args)
            try:
               out = eval(arg)
            except:
                out = False
            return out

        # Will use same pattern to search for  parameters and messages
        # All special parameters are recorded
        # From experiment.yml
        table = {'_root': [], '_purpose': [], '_type': [], '_notes': [], '_meta': []}
        # Additional
        if load_defaults:
            table.update({'_name': [], '_created': [], '_version': []})
        else:
            table.update({'_name': [], '_created': [], '_status': [], '_messages': [], '_version': []})
        if mode != 'all':
            table.update({'_experiments': []})

        # Filter experiment by name
        experiments = fnmatch.filter(self.experiments.keys(), pattern)
        encountered = []
        for root in experiments:
            try:
                em = xmen.manager.ExperimentManager(root)
                if load_defaults:
                    defaults = em.load_defaults()
                if types_match is not None:
                    if em.type not in types_match:
                        raise NoMatchException(root)

                paths = em.experiments if mode == 'all' else [em.defaults]
                for i, path in enumerate(paths):
                    if mode == 'all':
                        if not load_defaults:
                            params = em.load_params(path)
                            if params is None:
                                print(path)
                        else:
                            from copy import deepcopy
                            params = deepcopy(defaults)
                            params.update(em.overides[i])
                    else:
                        params = em.load_defaults()

                    # Find experiments who's parameters match all conditions
                    if param_match is not None:
                        extracted = {}
                        # Search through al param matches
                        for m in param_match:
                            # Condition from param string
                            splits = re.split(r'(<=|==|>=|<|>|!=)', m.replace(' ', ''))
                            if len(splits) == 1:
                                reg = splits[0]
                                valid = lambda x: True
                            elif len(splits) == 3:
                                reg, op, y = splits
                                valid = lambda x: _comparison(str(x), op, y)
                            elif len(splits) == 5:
                                y1, op1, reg, op2, y2 = splits
                                valid = lambda x: _comparison(y1, op1, str(x), op2, y2)
                            else:
                                if len(splits) > 5:
                                    print('ERROR: Only strings of the form (3.0 | cat) < param <= (4.0 | dog) can be set but'
                                          'got match_keys')
                                    exit()

                            # Keep keys if match or special (_...)
                            keys = [k for k in params if re.match(reg, k) is not None]
                            if len(keys) == 0:
                                raise NoMatchException(root)

                            keys += [k for k in params if k.startswith('_')]
                            for key in keys:
                                if valid(params[key]):
                                    extracted[key] = params[key]
                                else:
                                    raise NoMatchException(root)

                        if len(extracted) == 0:
                            raise NoMatchException(root)
                        else:
                            # If we get here add parameters to experiments (extracted must have at least one element)
                            # Infer current length of table (same length for every entry)
                            table_length = max([len(v) for v in table.values()])
                            if any([len(next(iter(table.values()))) != len(v) for v in table.values()]):
                                raise RuntimeError('This should not happen')

                            for k, v in extracted.items():
                                if k not in ['_root', '_purpose']:
                                    if k not in table:
                                        encountered += [k]
                                        table[k] = [missing_entry] * table_length + [v]
                                    else:
                                        table[k] += [v]

                            if '_meta' not in extracted:
                                table['_meta'] += [get_meta()]
                            for k in encountered:
                                if k not in extracted:
                                    table[k] += [missing_entry]
                    else:
                        for k, v in params.items():
                            if k.startswith('_') and k not in ['_root', '_purpose']:
                                if k not in table:
                                    table[k] = [v]
                                else:
                                    table[k] += [v]
                        if '_meta' not in params:
                            table['_meta'] += [get_meta()]

                    # Add data to table from experiment.yml
                    if load_defaults:
                        table['_name'] += [path.replace(em.root, '')[1:]]
                    if mode != 'all':
                        table['_experiments'] += [em.experiments]

                    table['_root'] += [em.root]
                    table['_purpose'] += [em.purpose]
                    table['_type'] += [em.type]
                    table['_notes'] += [em.notes]
            except NoMatchException as m:
                continue

        if mode != 'all':
            table.pop('_name')
        return table

    def find_to_dataframe(self, table, verbose=True, display_git=None,
                          display_purpose=None, display_date=None, display_messages=None, display_status=None,
                          display_meta=None):
        """Convert the output of the `find` method to a formatted data frame configured by args. If verbose then all
        entries will be displayed (independent of the other args)"""
        import pandas as pd
        special_keys = [k for k in table if k.startswith('_')]
        display = {
            'root': table.pop('_root')}

        if '_name' in special_keys:
            display['name'] = table.pop('_name')
        display.update({
            'created': table.pop('_created'),
            'type': table.pop('_type'),
            'purpose': table.pop('_purpose')})

        if '_origin' in special_keys:
            table.pop('_origin')

        meta = table.pop('_meta')
        meta_dict = {}
        for m in meta:
            for k, v in m.items():
                if k in meta_dict:
                    meta_dict[k] += [m[k]]
                else:
                    meta_dict[k] = [m[k]]
        display.update(meta_dict)

        # Remove Notes
        table.pop('_notes')

        if '_status' in special_keys:
            display['status'] = table.pop('_status')
        # Add version information to table
        versions = table.pop('_version')
        display['commit'] = [version.get('git', {}).get('commit', None) for version in versions]

        # Add messages to table
        encountered = []
        if '_messages' in special_keys:
            messages = table.pop('_messages')  # List of dict
            message_dict = {}
            for i, m in enumerate(messages):
                for k, v in m.items():
                    if k not in message_dict:
                        message_dict[k] = [''] * i + [v]
                    else:
                        message_dict[k] += [v]
                for k in message_dict:
                    if k not in m:
                        message_dict[k] += ['']
            display.update(message_dict)
        if verbose:
            display_keys = list(display.keys())
        else:
            display_keys = [v for v in ('root', 'name') if v in display]
            if display_git:
                display_keys += ['commit']
            if display_status:
                display_keys += ['status']
            if display_date:
                display_keys += ['date']
            if display_meta:
                display_keys += list(meta_dict.keys()) + ['origin']
            if (isinstance(display_messages, str) or display_messages) and '_messages' in special_keys:
                display_keys += [k for k in message_dict.keys() if re.match(display_messages if isinstance(
                    display_messages, str) else '.*', k)]
                    # list(message_dict.keys())
                display_keys += ['date']
            if display_purpose:
                display_keys += ['purpose']

        display_keys += list(table.keys())
        # Add other params
        display.update(table)
        df = pd.DataFrame(display)
        # Shorten roots
        roots = [v["root"] for v in df.transpose().to_dict().values()]
        prefix = os.path.dirname(os.path.commonprefix(roots))
        if prefix != '/':
            roots = [r[len(prefix) + 1:] for r in roots]
        df.update({'root': roots})
        # Finally filter
        df = df.filter(items=display_keys)
        return df, prefix

    def add_class(self, module, name):
        import importlib.util
        import stat
        mod = importlib.import_module(module)

        # Get experiment from module
        X = getattr(mod, name)
        x = X()
        script = x.get_run_script()

        # Make executable run script
        path = os.path.join(self._dir, 'experiments')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '.'.join([module, name]))
        # Note old experiments will be written over
        open(path, 'w').write(script)
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)
        self.python_experiments.update({name: path})
        self._to_yml()
