"""A module containing several utilitity functions, classes and Meta Classes used by the ExperimentManager
and the Experiment classes."""
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
import inspect
import logging
DATE_FORMAT = "%y-%m-%d-%H-%M-%S-%f"


def load_param(root, file='params.yml'):
    """Load parameters from a params.yml file."""
    import ruamel.yaml
    with open(os.path.join(root, file), 'r') as params_yml:
        params = ruamel.yaml.load(params_yml, ruamel.yaml.RoundTripLoader)
    return params


def save_param(params, root, file='params.yml'):
    """Save a dictionary of parameters at ``{root}/params.yml``

    Args:
        params (dict): A dictionary of parameters to be saved. Can also be a CommentedMap from ruamel.yaml
        root (str): The root of the experiment
    """
    import ruamel.yaml
    with open(os.path.join(root, file), 'w') as out:
        yaml = ruamel.yaml.YAML()
        yaml.dump(params, out)


def load_params(roots):
    """Load params.yml files into a list of dictionaries from a list of paths"""
    import ruamel.yaml
    from xmen.utils import commented_to_py
    out = []
    for path in roots:
        with open(os.path.join(path, 'params.yml'), 'r') as params_yml:
            params = ruamel.yaml.load(params_yml, ruamel.yaml.RoundTripLoader)
        params = {k: commented_to_py(v) for k, v in params.items()}
        out.append(params)
    return out


class IncompatibleYmlException(Exception):
    pass


def flatten(d, parent_key='', sep='_'):
    """Flatten a nested dictionary to a single dictionary. The keys of nested entries will be joined using sep"""
    import collections
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dic_to_json(dic):
    import json
    return json.dumps(dic)


# Convert to yaml
def dic_to_yaml(dic, typ='rt', default_flow_style=False):
    """Convert dictionary to a yaml string (``dic`` can also be a CommentedMap)"""
    import ruamel.yaml
    from ruamel.yaml import StringIO

    stream = StringIO()
    ruamel.yaml.round_trip_dump(dic, stream)
    # yaml = ruamel.yaml.YAML(typ=typ)
    # yaml.default_flow_style = default_flow_style
    # yaml.dump(dic, stream)
    string = stream.getvalue()
    return string


def dic_from_yml(*, string=None, path=None):
    """Load from either a yaml string or path to a yaml file"""
    assert (string is None) != (path is None), 'One of string and path must be set'
    import ruamel.yaml
    yaml = ruamel.yaml.YAML()
    try:
        if path is not None:
            with open(path, 'r') as file:
                params = yaml.load(file)
        else:
            params = ruamel.yaml.round_trip_load(string, preserve_quotes=True)
    except:
        raise IncompatibleYmlException
    return params


def recursive_print_lines(dic, helps=None, start=''):
    import collections
    from ruamel.yaml.comments import CommentedMap
    lines = []
    dic = commented_to_py(dic)
    for k, v in dic.items():
        if type(v) is dict or type(v) is collections.OrderedDict or type(v) is CommentedMap:
            lines += [f'{k}:']
            lines += ['  ' + start + l for l in recursive_print_lines(v)]
        elif v is not None:
            h = ''
            if helps is not None:
                h = helps.get(k, None)
                if h is not None:
                    h = helps[k].split(":")[1].strip()
                else:
                    h = ''
            if h != '':
                lines += [f'{start}{k}: {v}   # {h}']
            else:
                lines += [f'{start}{k}: {v}']
    return lines


class MultiOut(object):
    def __init__(self, *args):
        self.handles = args

    def write(self, s):
        for f in self.handles:
            f.write(s)

    def flush(self):
        for f in self.handles:
            f.flush()


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_meta(get_platform=False, get_cpu=False, get_memory=False, get_disk=False,
             get_slurm=False, get_conda=False, get_network=False, get_gpu=False,
             get_environ=False, live=False):
    """Get Meta information for the system"""
    import uuid
    import socket
    import getpass
    import os
    import platform
    meta = {'mac': hex(uuid.getnode()),
            'host': socket.getfqdn(),
            'user': getpass.getuser(),
            'home': os.path.expanduser("~")}

    if get_platform:
        try:
            # system information
            uname = platform.uname()
            system = {
                'system': uname.system,
                'node': uname.node,
                'release': uname.release,
                'version': uname.version,
                'machine': uname.machine,
                'processor': uname.processor}
            meta.update({'system': system})
        except:
            pass
    if get_cpu:
        try:
            import psutil
            cpufreq = psutil.cpu_freq()
            cpu = {
                'physical': psutil.cpu_count(logical=False),
                'total':  psutil.cpu_count(logical=True),
                'max_freq': f"{cpufreq.max:.2f}Mhz",
                'min_freq': f"{cpufreq.min:.2f}Mhz",
                'cur_freq': f"{cpufreq.current:.2f}Mhz"}
            if live:
                cpu.update({'usage': {str(i): f'{percentage}%' for i, percentage in
                            enumerate(psutil.cpu_percent(percpu=True, interval=1))}})
            meta.update({'cpu': cpu})
        except:
            pass

    if get_memory:
        try:
            import psutil
            svmem = psutil.virtual_memory()
            virtual = {
                'total': f"{get_size(svmem.total)}",
                'free': f"{get_size(svmem.available)}",
                'used': f"{get_size(svmem.used)}",
                'percentage': f"{svmem.percent}%"}
            swap = psutil.swap_memory()
            swap = {
                "total": f"{get_size(swap.total)}",
                "free": f" {get_size(swap.free)}",
                "used": f"{get_size(swap.used)}",
                "percentage": f"{swap.percent}"}
            meta.update({'virtual': virtual, 'swap': swap})
        except:
            pass

    if get_disk:
        try:
            # Disk Information
            # get all disk partitions
            partitions = psutil.disk_partitions()
            disks = {}
            for partition in partitions:
                info = {
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                }
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    info.update({
                        'total': f"{get_size(partition_usage.total)}",
                        'used': f"{get_size(partition_usage.used)}",
                        "free": f"{get_size(partition_usage.free)}",
                        'percent': f"{partition_usage.percent}%"})
                except PermissionError:
                    # this can be catched due to the disk that
                    # isn't ready
                    continue
                disks.update({partition.device: info})

            # get IO statistics since boot
            disk_io = psutil.disk_io_counters()
            disks.update({
                'read': f"{get_size(disk_io.read_bytes)}",
                'write': f"{get_size(disk_io.write_bytes)}"})
            meta.update(
                {'disks': disks})
        except:
            pass

    if get_network:
        try:
            if_addrs = psutil.net_if_addrs()
            network = {}
            for interface_name, interface_addresses in if_addrs.items():
                interface = {}
                for address in interface_addresses:
                    interface.update(
                        {interface_name: {
                            str(address.family).split('.')[-1]: {
                                'address': f"{address.address}",
                                'netmask': f"{address.netmask}",
                                'broadcast': f"{address.broadcast}"}}})
                network.update(interface)
            net_io = psutil.net_io_counters()
            network.update({
                'sent': f"{get_size(net_io.bytes_sent)}",
                'received': f"{get_size(net_io.bytes_recv)}"})
            meta.update({'network': network})
        except:
            pass

    if get_gpu:
        try:
            import GPUtil
            device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if device is not None:
                if not isinstance(device, list):
                    device = [device]
                device = [int(d) for d in device]
                gpus = {}
                for gpu in GPUtil.getGPUs():
                    count = 0
                    if gpu.id in device:
                        gpus.update({
                            str(count): {
                                'name': gpu.name,
                                'id': f"{gpu.id}",
                                'stats': f"{gpu.temperature}°C, {gpu.load*100}%, {gpu.memoryUsed}/{gpu.memoryTotal}MB",
                                'uuid': f"{gpu.uuid}"}})
                        count += 1
                meta.update({'gpu': gpus})
        except Exception as m:
            print(m)
            pass

    if get_slurm:
        try:
            import os
            import sys
            import subprocess
            id = os.environ.get('SLURM_JOB_ID')
            if id is not None:
                slurm = {'id': id}
                if get_slurm:
                    out = subprocess.run(
                        ['/usr/bin/scontrol', 'show', 'jobid', str(id)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    variables = ' '.join(out.stdout.decode(sys.getdefaultencoding()).replace('\n', '  ').split())

                    for v in variables.split(' '):
                        args = v.split('=')
                        if len(args) == 2 and args[0] != '':
                            slurm.update({args[0]: args[1]})
                    meta.update({'slurm': slurm})
        except:
            pass

    if get_environ:
        meta.update({'environ': {k: v for k, v in os.environ.items()}})

    try:
        if 'CONDA_EXE' in os.environ and get_conda:
            conda = subprocess.run(
                [os.environ['CONDA_EXE'], 'env', 'export'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = conda.stdout.decode(sys.getdefaultencoding())
            from ruamel.yaml import YAML
            yaml = YAML(typ="safe")
            out = yaml.load(out)
            meta['conda'] = out
    except:
        pass

    return meta


def get_attribute_helps(cls):
    """Get all help files for class ``cls`` from doc strings from all inherited classes."""
    helps = {}
    if cls is object:
        return helps
    else:
        atts = cls().__dict__.items()
        att_keys = [k for k, v in atts if k[0:2] != '__' and not hasattr(v, '__call__')]
        docs = [d for d in [cls.__doc__, cls.__init__.__doc__] if d is not None]
        for d in docs:
            if 'Parameters' in d:
                candidates = d.split('Parameters')[1].splitlines()
                candidates = [c.strip() for c in candidates]
                for a in att_keys:
                    h = [s.strip() for s in candidates if s.startswith(a + ' ')]
                    if len(h) > 1:
                        # Be even more restrictive
                        h = [s for s in h if ':' in s]
                    if len(h) == 0:
                        helps.update({a: None})
                    elif len(h) == 1:
                        helps.update({a: h[0]})
                    else:
                        raise ValueError('Multiple _helps found for attribute {}. Helps found: {}'.format(a, h))
        else:
            helps.update(get_attribute_helps(cls.__base__))
            return helps


def get_docs(cl):
    """Return all docs for every inherited class"""
    if cl is object:
        return [cl.__doc__]
    else:
        return [cl.__doc__] + get_docs(cl.__base__)


def get_parameters(lines, name):
    params = {}
    helps = []
    for l in lines:
        # if l.startswith(indent * ' ' + 'self.')  and 'self._' not in l:
        # All parameters will have a comment
        if not l.replace(' ', '').startswith('#') and '#@p' in l.replace(' ', ''):
            l = l.strip()
            # default always appears after = and before comment
            default = l.split('=')[1].split('#')[0].strip() if '=' in l else None
            # comment always appears after #@p
            comment = l.split('@p')[1].strip() if len(l.split('@p')) > 1 else None
            if comment == '':
                comment = None
            # New line
            l = l.split('#')[0]
            l = l.split('=')[0]
            ty = None
            if ':' in l:  # self.a: int ...
                attr = l.split(':')[0].replace('self.', '').strip()
                ty = l.split(':')[1].strip()
            else:  # self.a (already stripped)
                attr = l.replace('self.', '').strip()
            # Generate attribute lines
            help_string = f'    {attr}'
            if ty is not None:
                help_string += f' ({ty}):'
            else:
                help_string += ':'
            if comment is not None:
                help_string += f' {comment.strip()}'
            if default is not None:
                help_string += f' (default={default})'
            # Log parameters
            params.update({attr.strip(' '): (default, ty, comment, help_string, name)})
            helps += [help_string]
        return params, helps


class TypedMeta(type):
    """A meta class helper used to generate automatic doc strings generated from typed class definitions. This allows
    the docstrings to be defined inside the ``__init__`` body instead of inside the doc string iradicating the need
    to type attributes twice::

        MyClass(metaclass=TypedMeta):
            '''This class is bound to do something...'''

            a int: 3   # @p My first attribute
            b int: 5   # @p My second attribute

         m = MyClass()
         print(m.__init.__.__doc__)

    Now if we call::

        >>> ob = print(MyClass.__doc__)

         This class is bound to do something...

         Parameters:
             a (int) : My first attribute (default=3)
             b (int) : My second attribute (default=5)

    The doc string has automatically been updated.
    """

    def __init__(cls, name, bases, attr_dict):
        super(TypedMeta, cls).__init__(name, bases, attr_dict)
        import copy

        # Add _params attribute to cls if it does
        # not already have one.
        if '_params' not in dir(cls):
            cls._params = {}
        else:
            # As each cls is inspected the parameters of each are
            # added to the base class. If multiple objects inherit
            # from the same base who's metaclass is TypedMeta then
            # parameters from one subclass will be available to
            # to another which is counter-intuitive.
            # To avoid this parameters are deep copied down the hierarchy
            # ensuring each class has a unique set of parameters.
            cls._params = copy.deepcopy(cls._params)
            # Parameters are inherited following the experiments inheritance
            # order. In order to make this work the inheritance of
            # _params no longer is the same as the first _params in
            # the method resolution order but will be a merge of
            # all the parameters encountered in all superclasses with
            # merging following the method resolution order instead
            for sup in reversed(cls.mro()):
                sup_params = getattr(sup, '_params', None)
                if sup_params is not None and sup is not cls:
                    cls._params = {**cls._params, **copy.deepcopy(sup_params)}
            # It is possible for users to update the class attribute defaults
            # and type in subclasses which is not currently reflected
            # in the _param default values. To counter this
            # the current default and type are updated dynamically from the current
            # value of each in the current class.
            pops = []
            for k in cls._params:
                try:
                    val = getattr(cls, k)
                    # Subclasses could override a parameter with a
                    # new instance. In which cases they should be removed
                    # from the parameters. These are added to pops and
                    # removed after the loop
                    if isinstance(val, property) or callable(val):
                        pops.append(k)
                        raise AttributeError
                except AttributeError:
                    continue
                # Some work is needed to convert __annotations__
                # into a nice printable string...
                ty = cls.__annotations__.get(k, cls._params[k][1])
                if not isinstance(ty, str):
                    string = getattr(ty, '__name__', None)
                    if string is not None:
                        string = str(string).replace('.typing', '')
                    ty = string
                help = cls._params[k][2]
                helpstring = f'    {k}{f" ({str(ty)})" if ty is not None else ""}: {help} (default={val})'
                # Update parameters
                cls._params[k] = (val, ty, help, helpstring, cls._params[k][-1])

            # Remove parameters that have since been
            # overridden
            for p in pops:
                cls._params.pop(p)

        # Inspect the cls body for parameter definitions
        helps = []

        try:
            cls_source = inspect.getsource(cls)
        except OSError:
            # If the cls has no source code than all of the above cannot be executed
            return

        if cls.__doc__ is not None:
            # Remove the doc string
            cls_source = cls_source.replace(cls.__doc__, "")
        lines = [l.strip() for l in cls_source.splitlines()]

        # Note any attribute which is private is not a
        # valid parameter candidate.
        candidates = [c for c, p in cls.__dict__.items() if
                      not isinstance(p, property) and not c.startswith('_')]
        # This allows both parameters in the class body and in the
        # __init__ method to be treated the same.
        lines = [''.join(['self.', l]) for l in lines if any(l.startswith(c) for c in candidates)]

        # Add parameters from the __init__ method.
        # Note that in the case that an experiment inherits
        # from another it does not need to define an __init__
        # method. It is therefore a waste of effort to re-look
        # up these parameters as all superclass __init__'s will
        # already have been inspected. To avoid this we check to
        # see if the cls defines a new __init__. This is done
        # by inspecting the cls.__dict__ attribute.
        if '__init__' in cls.__dict__:
            code = inspect.getsource(cls.__init__)
            lines += code.splitlines()

        for l in lines:
            # if l.startswith(indent * ' ' + 'self.')  and 'self._' not in l:
            # All parameters will have a comment
            if not l.replace(' ', '').startswith('#') and '#@p' in l.replace(' ', ''):
                l = l.strip()
                # default always appears after = and before comment
                default = l.split('=')[1].split('#')[0].strip() if '=' in l else None
                # comment always appears after #@p
                comment = l.split('@p')[1].strip() if len(l.split('@p')) > 1 else None
                if comment == '':
                    comment = None
                # New line
                l = l.split('#')[0]
                l = l.split('=')[0]
                ty = None
                if ':' in l:   # self.a: int ...
                    attr = l.split(':')[0].replace('self.', '').strip()
                    ty = l.split(':')[1].strip()
                else:   # self.a (already stripped)
                    attr = l.replace('self.', '').strip()
                # Generate attribute lines
                help_string = f'    {attr}'
                if ty is not None:
                    help_string += f': {ty}'
                if default is not None:
                    help_string += f'={default}'
                if comment is not None:
                    help_string += f' ~ {comment.strip()}'

                # Log parameters
                cls._params.update({attr.strip(' '): (default, ty, comment, help_string, cls.__name__)})
                helps += [help_string]

        if cls.__doc__ is None:
            cls.__doc__ = ""

        # Note this will always override new parameters as they are found.
        lines = []
        for sup in reversed(cls.mro()):
            l = len(lines)
            for n, (_, _, _, h, c) in cls._params.items():
                if c == sup.__name__ and not n.startswith('_'):
                    lines += [' ' + h]
            if len(lines) - l > 0:
                lines.insert(l, '  ' + '-' * len(sup.__name__))
                lines.insert(l, '  ' + sup.__name__)
                lines.insert(l, '  ')


        if len(lines) > 0:
            cls.__doc__ += '\n\nParameters:\n'
            cls.__doc__ += '\n'.join(lines)

    def __iter__(self):
        """Iterating over the class simply returns itself once, None once and then ends.
        This allows functional experiments to be defined using the new signature::

            Exp = functional_experiment(func)

        whilst also being backward compatible with the previous signatuew::

            Exp, _ = functional_experiment(func)

        In the latter case ``_`` will be assigned the value of None.
        """
        self.count = 0
        return self

    def __next__(self):
        self.count += 1
        if self.count == 1:
            return self
        elif self.count == 2:
            return None
        else:
            raise StopIteration


def get_git(path):
    """Get git information for the given path.

    Returns:
         (dict): Empty if status == False otherwise with keys:
            * remote: A url to the remote repository
            * hash: A git hash to the current commit
        status (bool): False if either git is not available, the repo is not in a git repository or if there are
            uncommited changes in the current repository. Otherwise true
    """
    # Get the directory path which trigerred the call to get_git
    import git
    try:
        git_repo = git.Repo(path, search_parent_directories=True)
    except:    # If something goes wrong we just assume that git is not available TODO(robw): Is there a better way?
        logging.info(f'Could not load git repo for path {path}')
        return {}
    info = {
        'local': git_repo.git.rev_parse("--show-toplevel"),
        'remote': git_repo.remotes.origin.url,
        'commit': next(git_repo.iter_commits()).hexsha,
        'branch': git_repo.active_branch.name}
    return info


def get_version(*, path=None, cls=None, fn=None):
    """Get version information for either a path to a directory or a class. Git version information is loaded
    if available.

    Args:
        path (str): A path to a repository which is inspected for version information
        cls (Class): A Class object to be inspected for version information

    Returns:
        version (dict): A dictionary containing at least one of the following:
            * ``if path is not None:``
                * ``'path'``: A copy of the path
            * ``if cls is not None:``
                * ``'module'``: A path to the module in which the class was defined
                * ``'class'``: The name of the class
            * ``if git != {}``: (i.e if path or module is in a git repository):
                * ``'git_local'``: The root of the local git repository
                * ``'git_commit'``: The hash of the commit
                * ``'git_remote'``: The remote url if has remote else ``None``
    """
    if (path is None) == (cls is None) == (fn is None):
        raise ValueError('Exactly one of path or class must be set!')

    if cls is not None:
        # Note: inspecting cls.__init__ is compatible with ipython whilst inspecting cls directly is not
        module = os.path.realpath(inspect.getfile(cls))
        path = os.path.dirname(module)
        version = {'module': cls.__module__, 'class': cls.__name__, 'path': path}
    elif fn is not None:
        import importlib
        mod, name = fn
        path = importlib.import_module(mod).__file__
        version = {'module': mod, 'function': name, 'path': path}
    else:
        version = {'path': path}

    git = get_git(path)  # Try and get git information
    if git != {}:
        version.update({'git': git})
    return version


def commented_to_py(x, seq=tuple):
    from ruamel.yaml.comments import CommentedSeq, CommentedMap
    if type(x) is CommentedMap:
        return {k: commented_to_py(v) for k, v in x.items()}
    if type(x) is CommentedSeq:
        return seq(commented_to_py(v) for v in x)
    else:
        return x


def get_run_script(module, name, shell='/usr/bin/env python3', comment='#'):
    """Generate a run script for a particular experiment.

    Args:
        module (str): the module to look in
        name (str): the name of the experiment in the module. If name corresponds to
            a function it will be converted to an Experiment class
    """
    import sys
    import xmen
    import datetime
    import importlib
    sh = [f'#!{shell}']
    mod = importlib.import_module(module)
    X = getattr(mod, name)
    if type(X) is not xmen.utils.TypedMeta:
        from xmen.functional import functional_experiment
        X, _ = functional_experiment(X)
        version = xmen.utils.get_version(path=mod.__file__)
    else:
        version = xmen.utils.get_version(cls=X.__class__)

    sh += [f'# File generated on the {datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")}']
    if 'git' in version:
        sh += [f'{comment} GIT:']
        sh += [f'{comment} - repo {version["git"]["local"]}']
        sh += [f'{comment} - branch {version["git"]["branch"]}']
        sh += [f'{comment} - remote {version["git"]["remote"]}']
        sh += [f'{comment} - commit {version["git"]["commit"]}']
        sh += ['']
    possible_roots = sorted([p for p in sys.path if p in version.get('module', version.get('path'))])
    root = None
    if len(possible_roots) > 0:
        root = possible_roots[0]

    sh += ['import sys']
    sh += ['import importlib']
    sh += ['import xmen']
    if root is not None:
        sh += [f'sys.path.append("{root}")']
    sh += ['import logging']
    sh += ['logger = logging.getLogger()']
    sh += ['logger.setLevel(logging.INFO)']
    sh += ['']
    sh += [f'mod = importlib.import_module("{module}")']
    sh += [f'X = getattr(mod, "{name}")']
    sh += ['if type(X) is not xmen.utils.TypedMeta:']
    sh += ['    from xmen.functional import functional_experiment']
    sh += ['    X = functional_experiment(X)']
    sh += ['X().main()']
    return '\n'.join(sh)





def dics_to_pandas(dics, reg):
    """Convert a list of nested dictionaries into a pandas data frame keeping only keys that match reg
    (after flattening)"""
    import re
    frames = []
    import pandas as pd
    from xmen.utils import flatten
    for dic in dics:
        dic = flatten(dic)
        dic = {k: v for k, v in dic.items() if re.match(reg, k)}
        dic = {k: [v] for k, v in dic.items()}
        frames.append(pd.DataFrame(dic))
    return pd.concat(frames, axis=0, sort=False, ignore_index=True)



if __name__ == '__main__':
    from xmen.experiment import Experiment

    class TestExperiment(Experiment):
        n_epochs: int = 10   # @p Some help
        n_steps: int = 1  # @p Some other help
        nn_c0: int = 8  # @p Another piece of help

    exp = TestExperiment()
    print(exp)



