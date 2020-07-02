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


def get_parameters(cls):
    code = inspect.getsource(cls.__init__)
    lines = code.splitlines()
    new_lines = []
    # new_lines = ['', '', 'Attributes:']
    for l in lines:
        if 'self.' in l and 'self._' not in l:
            l = l.strip()
            default = l.split('=')[1] if '=' in l else None  # default always appears after = and before comment
            comment = l.split('#')[1] if '#' in l else None  # comment always appears after #
            if comment is not None and default is not None:
                default = default.split('#')[0].strip()
            ty = None
            if ':' in l:  # self.a: int ...
                attr = l.split(':')[0].replace('self.', '')
                if '=' in l:
                    ty = l.split(':')[1].split('=')[0].strip()
                elif '#' in l:
                    ty = l.split(':')[1].split('#')[0].strip()
                else:
                    ty = l.split(':')[1].strip()
            elif '=' in l:  # self.a = 3 ....
                attr = l.split('=')[0].replace('self.', '')
            elif '#' in l:  # self.a # Some docs
                attr = l.split('#')[0].replace('self.', '')
            else:  # self.a (already stripped)
                attr = l.replace('self.', '')
            # Generate attribute lines
            new_line = f'    {attr}'
            if ty is not None:
                new_line += f' ({ty}):'
            else:
                new_line += ':'
            if comment is not None:
                new_line += f' {comment.strip()}'
            if default is not None:
                new_line += f' (default={default})'
            new_lines += [new_line]


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
            # added to the base class. If multiple objects inheret
            # from the same base who's metaclass is TypedMeta then
            # parameters from one subclass will be available to
            # to another which is counter-intuitive.
            # To avoid this parameters are deep copied down the hierarchy
            # ensuring each class has a unique set of parameters.
            cls._params = copy.deepcopy(cls._params)

        # Inspect the cls body for parameter definitions
        helps = []
        cls_source = inspect.getsource(cls)
        if cls.__doc__ is not None:
            # Remove the doc string
            cls_source = cls_source.replace(cls.__doc__, "")
        lines = [l.strip() for l in cls_source.splitlines()]

        # Note any attribute which is private will is not a
        # valid parameter candidate.
        candidates = [c for c, p in cls.__dict__.items() if
                      not isinstance(p, property) and not c.startswith('_')]
        # This allows both parameters in the class body and in the
        # __init__ method to be trated the same.
        lines = [''.join(['self.', l]) for l in lines if any(l.startswith(c) for c in candidates)]

        # Add parameters from the __init__ method.
        # Note that in the case that an experiment inherits
        # from another it does not need to define an __init__
        # method. It is therefore a waste of effort to re-look
        # up these parameters as all superclass __init__'s will
        # already have been inspected. To avoid this check to
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
                new_line = f'    {attr}'
                if ty is not None:
                    new_line += f' ({ty}):'
                else:
                    new_line += ':'
                if comment is not None:
                    new_line += f' {comment.strip()}'
                if default is not None:
                    new_line += f' (default={default})'
                # Log parametes
                cls._params.update({attr.strip(' '): (default, ty, comment, new_line)})
                helps += [new_line]

        if cls.__doc__ is None:
            cls.__doc__ = ""

        # Note this will always override new parameters as they are found.
        cls.__doc__ += '\n'.join(['', '', f'Parameters:'] + [
            cls._params[k][-1] for k in cls._params if not k.startswith('_')])


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


def get_version(*, path=None, cls=None):
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
    if (path is None) == (cls is None):
        raise ValueError('Exactly one of path or class must be set!')

    if cls is not None:
        # Note: inspecting cls.__init__ is compatible with ipython whilst inspecting cls directly is not
        module = os.path.realpath(inspect.getfile(cls.__init__))
        path = os.path.dirname(module)
        version = {'module': module, 'class': cls.__name__}
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


if __name__ == '__main__':
    from xmen.experiment import Experiment

    class TestExperiment(Experiment):
        n_epochs: int = 10  #@p Some help
        n_steps: int = 1    #@p Some other help
        nn_c0: int = 8      #@p Another piece of help

    exp = TestExperiment()
    print(exp)



