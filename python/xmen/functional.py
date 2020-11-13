from xmen import Experiment
import xmen


def functional_experiment(func):
    import inspect
    Exp = type(func.__name__, (Experiment,), {})
    signature = inspect.signature(func)
    src = '('.join(inspect.getsource(func).split('(')[1:])
    lines = []
    X = None
    for k in signature.parameters:
        p = signature.parameters[k]
        if p.default != xmen.X:
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

            comment = p.name.join(src.split(p.name)[1:]).split('\n')[0].split('#')
            if len(comment) == 2:
                comment = comment[-1]
            else:
                comment = None
            # Generate attribute lines
            help_string = f'    {p.name}'
            if p.annotation is not None:
                help_string += f' ({ty}):'
            else:
                help_string += ':'
            if comment is not None:
                help_string += f' {comment.strip()}'
            if default is not None:
                help_string += f' (default={default})'
            lines += [help_string]
            # cls._params.update({attr.strip(' '): (default, ty, comment, help_string, cls.__name__)})
            setattr(Exp, p.name, default)
            Exp._params.update({p.name: (default, ty, comment, help_string, func.__name__)})

        else:
            X = p.name

    Exp.__doc__ = ''
    if func.__doc__ is not None:
        Exp.__doc__ = func.__doc__

    if len(lines) > 0:
        Exp.__doc__ += '\n\nParameters:\n'
        Exp.__doc__ += '\n'.join(lines)

    Exp.fn = (func.__module__, func.__name__)

    def run(self):
        params = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        if X is not None:
            params.update({X: self})
        return func(**params)

    Exp.run = run
    return Exp, X


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
