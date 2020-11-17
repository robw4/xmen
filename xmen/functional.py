"""A collection of function and variables used to define experiments using the functional api"""

from xmen import Experiment


class Root(Experiment):
    """The first argument passed to a functional experiment and principally used to root an experiment instance to a
    particular directory::


        def functional_experiment(root: Root, ...):
            with open(root.directory, 'w') as f:
                f.write('Running experiment)

            root.message({'time': time.time()})

    Note:
        Root is nothing more than Experiment with a different name. Whilst principally offering exactly the same
        functionality, primarily the purpose of Root is to expose the directory property and messaging protocol of
        the Experiment class to functional experiment definitions. However, there is nothing stopping the user
        form using the full functionality of the Experiment class if they wish. Please consult the Experiment class
        documentation in this case.
    """


def read_comments(fn):
    """A helper function for reading comments from the function definition. This should not be generally needed as
    xmen takes care of this for you.

    Args:
        fn: A functional experiment definition conforming to the xmen api

    Returns:
        docs (str): A help string generated for each function argument.
    """
    import inspect
    signature = inspect.signature(fn)
    src = '('.join(inspect.getsource(fn).split('(')[1:])

    lines = []
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
    return '\n'.join(lines)


def functional_experiment(fn):
    """Convert a functional experiment to a class definition. Generally this should not be needed
    as xmen takes care of this for you. Specifically:

        - The parameters of the experiment are added from the argument of the function
        - Comments next to each argument will be automatically added to the doc string of the experiment
        - The experiments run method will be set to ``fn``

    Args:
        fn (xmen.Root): An experiment definition conforming to the xmen functioanl api (the function must take as its
            first argument an object inheriting from experiment)

    Returns:
        Exp (class): A class equivalent definition of `fn`
    """
    import inspect
    # Generate new class instance with name of the function inheriting from Experiment
    cls = type(fn.__name__, (Experiment,), {})
    # Add parameters and get helps from the function definition
    signature = inspect.signature(fn)
    for i, k in enumerate(signature.parameters):
        p = signature.parameters[k]

        if i > 0:
            default = p.default
            if default == inspect.Parameter.empty:
                default = None

            # Add attribute to class
            setattr(cls, p.name, default)

    # Add parameters to __doc__ of the function
    cls.__doc__ = ''
    if fn.__doc__ is not None:
        cls.__doc__ = fn.__doc__
    if not hasattr(fn, 'autodocs'):
        cls.__doc__ += '\n\nParameters:\n' + read_comments(fn)
    cls.fn = (fn.__module__, fn.__name__)

    # generate run method from the function
    def run(self):
        params = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return fn(self, **params)

    cls.run = run
    return cls, None


def autodoc(func):
    """A decorator used to add parameter comments to the docstring of func."""
    _docs = read_comments(func)
    if func.__doc__ is None:
        func.__doc__ = ''
    func.__doc__ += '\n\nParameters:\n'
    func.__doc__ += _docs
    setattr(func, 'autodocs', _docs)
    return func
