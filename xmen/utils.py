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
DATE_FORMAT = "%Y-%m-%d-%H-%M-%S"


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


def get_meta():
    """Get Meta information for the system"""
    import uuid
    import socket
    import getpass
    return {'mac': hex(uuid.getnode()), 'host': socket.getfqdn(), 'user': getpass.getuser(), 'home': os.path.expanduser("~")}


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
            # Parameters are inherited following the python inheritance
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
                lines.insert(l, '    ' + sup.__name__)

        if len(lines) > 0:
            cls.__doc__ += '\n\nParameters:\n'
            cls.__doc__ += '\n'.join(lines)


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
    sh += ['    X, _ = functional_experiment(X)']
    sh += ['X().main()']
    return '\n'.join(sh)


if __name__ == '__main__':
    from xmen.experiment import Experiment

    class TestExperiment(Experiment):
        n_epochs: int = 10   # @p Some help
        n_steps: int = 1  # @p Some other help
        nn_c0: int = 8  # @p Another piece of help

    exp = TestExperiment()
    print(exp)


