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
import git


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
                    h = [s.strip() for s in candidates if s[:len(a)] == a]
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


class TypedMeta(type):
    """A meta class helper used to generate automatic doc strings generated from typed class definitions. This allows
    the docstrings to be defined inside the ``__init__`` body instead of inside the doc string iradicating the need
    to type attributes twice::

        MyClass(metaclass=TypedMeta):
            '''This class is bound to do something...'''
            a int: 3   # My first attribute
            b int: 5   # My second attribute

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
        code = inspect.getsource(cls.__init__)
        lines = code.splitlines()
        new_lines = []
        # new_lines = ['', '', 'Attributes:']
        for l in lines:
            if 'self.' in l and 'self._' not in l:
                l = l.strip()
                default = l.split('=')[1] if '=' in l else None    # default always appears after = and before comment
                comment = l.split('#')[1] if '#' in l else None    # comment always appears after #
                if comment is not None:
                    default = default.split('#')[0].strip()
                ty = None
                if ':' in l:   # self.a: int ...
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
                else:   # self.a (already stripped)
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
        # Add to __class__.__init__.__doc__
        if cls.__init__.__doc__ is None:
            cls.__init__.__doc__ = ''
        cls.__init__.__doc__ += '\n'.join(['', '', 'Parameters:'] + new_lines)
        # Add to __class__.__doc__
        if cls.__doc__ is None:
            cls.__doc__ = ""
        cls.__doc__ += '\n'.join(['', '', f'Parameters:'] + new_lines)

    # def __new__(mcs, name, bases, attr_dict):
    #     x = super(TypedMeta, mcs).__new__(mcs, name, bases, attr_dict)
    #     code = inspect.getsource(x)
    #     if 'def' in code:
    #         # Make searching space smaller
    #         code = code.split('def')[0]
    #
    #     lines = code.splitlines()
    #     new_lines = ['', '', 'Attributes:']
    #     for a, t in x.__annotations__.items():
    #         matches = [c for c in lines if a + ':' in c]
    #         if len(matches) != 1:
    #             raise ValueError("Syntax for typed experiment is not correct. Make sure that stirng patterns"
    #                              "like '{attribute_name}:' do not appear in comments.")
    #         definition, comment = matches[0].split('#') if '#' in matches[0] else (matches[0], None)
    #         attr, type_default = definition.replace(" ", "").split(':')
    #         ty, default = type_default.split('=') if '=' in type_default else (type_default, None)
    #         new_line = f'    {attr} ({ty}):'
    #         if comment is not None:
    #             new_line += f' {comment.strip()}'
    #         if default is not None:
    #             new_line += f' (default={default})'
    #         new_lines += [new_line]
    #         lines.remove(matches[0])
    #     docs = '\n'.join(new_lines)
    #     x.__doc__ = x.__doc__ + docs
    #     return x

    #
    # lines = code.split(name)[1].split('##')[1].splitlines()
    # new_lines = ['', '', 'Attributes:']
    # for l in lines:
    #     if l == '' or l.isspace():
    #         continue
    #     # Get attribute and default types
    #     definition, comment = l.split('#') if '#' in l else (l, None)
    #     attr, type_default = definition.replace(" ", "").split(':')
    #     ty, default = type_default.split('=') if '=' in l else (type_default, None)
    #     new_line = f'    {attr} ({ty}):'
    #     if comment is not None:
    #         new_line += f' {comment.strip()}'
    #     if default is not None:
    #         new_line += f' (default={default})'
    #     new_lines += [new_line]
    # docs = '\n'.join(new_lines)
    # x.__doc__ = x.__doc__ + docs
    # return x


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
    try:
        git_repo = git.Repo(path, search_parent_directories=True)
    except:    # If something goes wrong we just assume that git is not available TODO(robw): Is there a better way?
        print(f'Could not load git repo for path {path}')
        return {}
    info = {
        'local': git_repo.git.rev_parse("--show-toplevel"),
        'remote': git_repo.remotes.origin.url,
        'commit': next(git_repo.iter_commits()).hexsha,
        'branch': git_repo.active_branch.name
    }
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
