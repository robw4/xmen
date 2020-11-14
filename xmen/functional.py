from xmen import Experiment


def functional_experiment(fn):
    """Convert a class inheriting from experiment from the given func.

    - func must have a signature ``fn(ex, ...)`` where ``ex`` has a base class
        experiment can be unused in which case use ``_`` and can be named whatever
    - the parameters of the returned Exp class are added from the function signature.
    - the run method of the Exp class is defined by fn()
    """
    import inspect
    # Generate new class instance with name of the function inheriting from Experiment
    cls = type(fn.__name__, (Experiment,), {})
    # Add parameters and get helps from the function definition
    signature = inspect.signature(fn)
    src = '('.join(inspect.getsource(fn).split('(')[1:])

    lines = []
    obj = None
    for i, k in enumerate(signature.parameters):
        p = signature.parameters[k]

        if i > 0:
            ty = p.annotation
            if ty == inspect.Parameter.empty:
                ty = None

            default = p.default
            if default == inspect.Parameter.empty:
                default = None

            if ty is not None:
                if not isinstance(ty, str):
                    string = getattr(ty, '__name__', None)
                    if string is not None:
                        string = str(string).replace('.typing', '')
                    ty = string

            # find first comment
            comments = []
            for ii, l in enumerate(src.splitlines()):
                l = l.split('#')[0]
                l = l.replace(' ', '')
                if ':' in l:
                    l = l.split(':')[0]
                elif '=' in l:
                    l = l.split('=')[0]
                if l == p.name:
                    comments = src.splitlines()[ii:]
                    break

            # comments = p.name.join(src.split(p.name)[1:]).split('\n')

            help = None
            for k, c in enumerate(comments):
                c = c.split('#')
                if k == 0 and len(c) == 2:
                    help = c[-1].strip()
                elif k > 0 and c[0].strip() == '' and len(c) == 2:
                    help += ' ' + c[1].strip()
                else:
                    break

            # Generate attribute lines
            help_string = f'{p.name}'
            if ty is not None:
                help_string += f': {ty}'
            if default is not None:
                help_string += f'={default}'
            if help is not None:
                help_string += f' ~ {help.strip()}'

            # wrap text
            import textwrap
            help_string_wrapped = textwrap.wrap(help_string, break_long_words=False)
            for i in range(len(help_string_wrapped)):
                help_string_wrapped[i] = textwrap.indent(help_string_wrapped[i], ' ' * (4 + (i > 0) * 2))

            lines += ['\n'.join(help_string_wrapped)]

            # Add attribute to class
            setattr(cls, p.name, default)
            # Update parameter information
            cls._params.update({p.name: (default, ty, help, help_string, fn.__name__)})
        else:
            obj = p.name

    # Add parameters to __doc__ of the function
    cls.__doc__ = ''
    if fn.__doc__ is not None:
        cls.__doc__ = fn.__doc__
    if len(lines) > 0:
        cls.__doc__ += '\n\nParameters:\n'
        cls.__doc__ += '\n'.join(lines)
    cls.fn = (fn.__module__, fn.__name__)

    # generate run method from the function
    def run(self):
        params = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return fn(self, **params)

    cls.run = run
    return cls, obj


# def functional(func):
#     cls, X = functional_experiment(func)
#
#     def _func(*args, **kwargs):
#         if len(args) < 2 and len(kwargs) == 0:
#             exp = cls()
#             exp.parse_args()
#             params = {k: v for k, v in exp.__dict__.items() if not k.startswith('_')}
#             if X is not None:
#                 params.update({X: exp})
#
#             if exp.status not in ['default']:
#                 with exp:
#                     x = func(**params)
#                 return x
#             else:
#                 print('To run an experiment please register first. See --help for more options')
#         else:
#             return func(*args, **kwargs)
#     return _func
