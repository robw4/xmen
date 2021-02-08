#  Copyright (C) 2019  Robert J Weston, Oxford Robotics Institute
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
import time
import datetime
import glob
import re
import inspect
import copy
import warnings
import os
import logging
logging.basicConfig(format='%(message)s')

TRIGGERS = ['step', 'epoch', 'era', 'eon', 'supereon']
BRIEF = {"step": "s", "epoch": "e", "era": "er", "eon": "eo", "supereon": "se"}
IDX = {c: i for i, c in enumerate(TRIGGERS)}


class Spec(object):
    def __init__(self, spec):
        assert isinstance(spec, str), 'invalid specification found'
        regex, steps, trigger = None, None, None
        if '@' in spec:
            parts = spec.split('@')
            regex, string = parts
            if regex == '':
                regex = None
            steps = re.search('[0-9]+', string)
            assert steps is not None, f'No steps found in {spec}'
            steps = steps.group()
            trigger = {}
            for k, v in BRIEF.items():
                trigger[k] = k
                trigger[v] = k
            trig_key = string.split(steps)[1]
            trigger = trigger.get(trig_key, None)
            assert trigger is not None, f'Invalid or no trigger specified in {spec}'
        else:
            regex = spec
        self.regex = regex
        self.modulo = int(steps) if steps is not None else None
        self.trigger = trigger

    def __str__(self):
        return f'{self.regex if self.regex is not None else ""}@{self.modulo}{self.trigger}'


def read_modulo_string(string):
    assert isinstance(string, str)
    parts = string.split('@')
    regex, string = parts
    steps = re.search('[0-9]+', string)
    assert steps is not None
    steps = steps.group()
    trigger = {}
    for k, v in BRIEF.items():
        trigger[k] = k
        trigger[v] = k
    trigger = trigger[string.split(steps)[1]]
    return regex, int(steps), trigger


class LastStep(Exception):
    def __init__(self, trigger, step):
        self.trigger = trigger
        self.step = step


class Hook(Spec):
    """A base class defining a variable passing protocol with the Monitor class. Ever time the monitors count is
    divisible by ``modulo`` for a particular ``trigger`` then the hook is passed all the variables matching ``regex``
    from the current stack. Users therefore define hooks by overloading the hooks __call__ method (with the same
    call signature).
    """
    def __call__(self, var_dict, monitor):
        raise NotImplementedError('All methods must implement the call method')


class EarlyStop(object):
    def __init__(self, condition='max', memory=5):
        """An early stop implementation. Returns true if the score in most distant memory is <= / >= all other scores
        in memory. Example:

        ::
            import numpy as np
            import matplotlib.pyplot as plt
            plt.style.use('ggplot')

            es = EarlyStop('max', 36)

            X = np.linspace(0, 10, 100)
            Y = 1 - np.sin(X) * X
            y = []
            for x in X:
                y.append(1 - np.sin(x) * x)
                es.append(y[-1], an_example=True, step=1)
                if es:
                    print(f'Breaking at {x}')
                    break
            print(max(es))
            best, count, tags = max(es)
            plt.plot(X, Y)
            plt.plot(X[:len(y)], np.array(y))
            plt.scatter(X[count], best, color='red')
            plt.show()

        """
        assert condition in ['max', 'min']
        assert memory >= 1

        from collections import deque
        self.condition = condition
        self.count = 0
        self.history = deque([None] * memory, maxlen=memory)
        self.counts = deque([None] * memory, maxlen=memory)
        self.tags = deque([None] * memory, maxlen=memory)

    def append(self, value, **tags):
        self.history.append(value)
        self.counts.append(self.count)
        self.tags.append(tags)
        self.count += 1

    def __bool__(self):
        import operator
        op = {'max': operator.le, 'min': operator.ge}[self.condition]
        first = self.history[0]
        return all(op(h, first) for h in self.history) if first is not None else False

    def __iter__(self):
        for h, s, t in zip(self.history, self.counts, self.tags):
            yield h, s, t

    def __repr__(self):
        best, count, tags = dict((("max", max), ("min", min)))[self.condition](self)
        return f'EarlyStop({bool(self)}, memory={len(self.history)}, condition={self.condition}, ' \
               f'best={best}, count={count}, tags={tags})'


class EarlyStopper(Hook):
    def __init__(self, metric, condition='max', memory=5, tags='', modulo=None, trigger='step'):
        if tags not in ['', '^$', None]:
            metric = tags + '|' + metric
        super(EarlyStopper, self).__init__(metric, modulo, trigger)
        self.early_stop = EarlyStop(condition, memory)
        self.metric = read_modulo_string(metric)[0]

    def __call__(self, tags, monitor):
        metric = tags.pop([k for k in tags if re.match(self.metric, k) is not None][0], None)
        if metric is None:
            raise RuntimeError(f'No Metric was found matching {self.metric}')
        self.early_stop.append(metric, **tags)
        if self.early_stop:
            monitor.stop(f'----- {self.early_stop} ----')


class Checkpointer(Hook):
    def __init__(self, spec, to_keep=None, expand=True):
        super(Checkpointer, self).__init__(spec)
        self.to_keep = to_keep
        self._checkpoint_buffer = {}
        self.expand = expand

    def __call__(self, var_dict, monitor):
        import torch
        from xmen.utils import get_version
        saved = []
        if monitor.directory is None:
            monitor.log(f'WARNING: Cannot checkpoint {list(var_dict.keys())} as monitor does not have a directory')
        else:
            if self.expand:
                pops = []
                updates = {}
                for k, v in var_dict.items():
                    if isinstance(v, dict):
                        pops.append(k)
                        updates.update({k + '_' + kk: vv for kk, vv in v.items()})
                for p in pops:
                    var_dict.pop(p)
                    var_dict.update(updates)

            for k, v in var_dict.items():
                if hasattr(v, 'state_dict'):
                    # Save directory
                    save_dir = os.path.join(monitor.directory, 'checkpoints', k)

                    # Create directory if it doesn't exist
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # Get PATHS in file if buffer is not loaded
                    if k not in self._checkpoint_buffer:
                        files = glob.glob(os.path.join(save_dir, '*'))
                        # Order files in ascending order by step (note sort must be used on int)
                        if len(files) != 0:
                            _, files = zip(*sorted([(int(os.path.splitext(os.path.split(ff)[-1])[0]), ff) for ff in files]))
                        self._checkpoint_buffer[k] = list(files)

                    if len(self._checkpoint_buffer[k]) == self.to_keep:
                        file = self._checkpoint_buffer[k].pop(0)
                        os.remove(file)
                    self._checkpoint_buffer[k].append(os.path.join(save_dir, f'{monitor.step}.pt'))
                    save_dict = {k: v for k, v in monitor.triggers.items() if v != 0.}
                    save_dict.update({
                        'version': get_version(cls=type(v)),
                        'date': datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"),
                        'state_dict': v.state_dict()})
                    torch.save(save_dict, self._checkpoint_buffer[k][-1])
                    saved.append(k)
                else:
                    warnings.warn(f'Value for key = {k} does not have a state_dict')
            monitor.log(f'saved {saved} at {monitor.directory}')


class XmenMessenger(Hook):
    """A hook for logging messages with an ``xmen.Experiment`` object.

        Example 1::

            from xmen import Experiment
            from xmen.monitor import Monitor

            messenger = XmenMessenger('y.*->ex.*@10s')   # log all variables matching the loss to experiments matching ex
            m = Monitor(hooks=[messenger])
            y1, y2 = 0, 0
            ex1, ex2 = Experiment(), Experiment()
            ex1.register('/tmp', 'ex1')
            ex2.register('/tmp', 'ex2')
            for i in m(range(40)):
                y1 += 1
                y2 += 2
                if i % 10 == 1:
                    print(', '.join(
                        [f"ex1: {k} = {ex1.messages.get(k, None)}" for k in ('y1', 'y2')] +
                        [f"ex2: {k} = {ex1.messages.get(k, None)}" for k in ('y1', 'y2')]))

            # Output
            # ex1: y1 = None, ex1: y2 = None, ex2: y1 = None, ex2: y2 = None
            # ex1: y1 = 10, ex1: y2 = 20, ex2: y1 = 10, ex2: y2 = 20
            # ex1: y1 = 20, ex1: y2 = 40, ex2: y1 = 20, ex2: y2 = 40
            # ex1: y1 = 30, ex1: y2 = 60, ex2: y1 = 30, ex2: y2 = 60


        Example 2::

            from xmen import Experiment
            from xmen.monitor import Monitor

            ex = Experiment()
            ex.register('/tmp', 'ex')
            m = Monitor(
                hooks=[
                    XmenMessenger('^y$|^i$->^ex$@10s', keep='min', leader='^y$')])

            x = -50
            for i in m(range(100)):
                x += 1
                y = x ** 2
                if i % 10 == 1:
                    print([ex.messages.get(k, None) for k in ('i', 'y')])

            # Output
            # [None, None]
            # [9, 1600]
            # [19, 900]
            # [29, 400]
            # [39, 100]
            # [49, 0]
            # [49, 0]
            # [49, 0]
            # [49, 0]
            # [49, 0]

        Example 3::

            from xmen import Experiment
            from xmen.monitor import Monitor

            ex = Experiment()
            ex.register('/tmp', 'ex')
            m = Monitor(
                hooks=[
                    XmenMessenger('z->^ex$@10s', keep='min', leader='y', expand=True)])

            z = {'x': 5, 'y': 10}
            for i in m(range(100)):
                z['i'] = i
                z['x'] += 1
                z['y'] = z['x'] ** 2
                if i % 10 == 1:
                    print([ex.messages.get(k, None) for k in ('i', 'y', 'x')])

            # [None, None, None]
            # [9, 225, 15]
            # [9, 225, 15]
            # [9, 225, 15]
            # [9, 225, 15]
            # [9, 225, 15]
            # [9, 225, 15]
            # [9, 225, 15]
            # [9, 225, 15]
            # [9, 225, 15]

        Example 4::

            from xmen import Experiment
            from xmen.monitor import Monitor

            ex = Experiment()
            ex.register('/tmp', 'ex')
            m = Monitor(
                hooks=[
                    XmenMessenger('z->^ex$@10s', keep='min', leader='z_y', expand=True, prepend=True)])


            z = {'x': 5, 'y': 10}
            for i in m(range(100)):
                z['i'] = i
                z['x'] += 1
                z['y'] = z['x'] ** 2
                if i % 10 == 1:
                    print([ex.messages.get(k, None) for k in ('z_i', 'z_y', 'z_x')])

            # same output as above
    """

    def __init__(self, spec, keep='latest', leader=None, expand=False, prepend=None):
        """
        Args:
            spec (Spec): hook specification in the form either ``{log_regex}->{exp_regex}@{modulo}{trigger}`` or
                ``{exp_regex}@{modulo}{trigger}`` where ``exp_regex`` is the experiments to message and ``log_regex`` are
                additional messages to log with the experiment. In the second case only timing and step information
                will be logged
            keep (str): One of ['latest', 'max', 'min'] in the case of message collision with each experiment
            leader (regex): If leader is not None then messages will be logged according to ``keep`` as judged by
                the variable in var_dict which matches ``leader``.
            expand: If a dictionary variable with K keys matches ``log_regex`` then it is converted to K variables with
                names corresponding to the keys in dict if ``expand==True``.
            prepend: If prepend is ``True`` then in the case above the name of each variable will be
                prepended by the dict variable name. eg. each variable will be called '{name}_{k}'.
        """
        assert keep in ['max', 'min', 'latest']
        self.keep = keep
        self.expand = expand
        self.leader = leader
        self.prepend = prepend
        self.log = r''

        if '->' in spec:
            self.log, spec = spec.split('->')
            spec = self.log + '|' + spec

        super(XmenMessenger, self).__init__(spec)

    def __call__(self, var_dict, monitor):
        """Leave messages, timing and step information with experiments"""
        from xmen.experiment import Experiment

        # if monitor.directory is None:
        #     monitor.log(f'WARNING: Cannot log {list(var_dict.keys())} as monitor does not have a directory')
        # else:
        # get and remove experiments from var_dict
        results = zip(*[(k, var_dict[k]) for k in var_dict if re.match(self.log, k) is None])
        results = list(results)
        if len(results) == 2:
            names, experiments = results
        else:
            return

        for k in names:
            var_dict.pop(k)

        if self.expand:
            # Expand dictionaries in var_dict
            pop, add = [], {}
            for k, v in var_dict.items():
                if isinstance(v, dict):
                    p = '' if self.prepend is None else k + '_'
                    add.update({p + kk: vv for kk, vv in v.items()})
                    pop.append(k)
            for p in pop:
                var_dict.pop(p)
            var_dict.update(add)

        for e, name in zip(experiments, names):
            # Get leader
            leader, best_leader = None, False

            if self.leader is not None:
                leader = [k for k in var_dict if re.match(self.leader, k) is not None][0]
            if isinstance(e, Experiment):
                e.message(monitor.summary(verbose=1))
                e.message(var_dict, keep=self.keep, leader=leader)
                keys = list(var_dict.keys())
                monitor.log(f'Left messages {keys if keys != [] else ""} with {name} at {e.directory}')


class Timer(Hook):
    """A simple timing hook used to log any timers setup by the experiment monitor"""
    def __call__(self, var_dict, monitor):
        s = monitor.summary(verbose=1)
        keys = [k for k in s if k not in BRIEF.values() and k != 'last']
        monitor.log(' '.join(f'{k}={s[k]}' for k in keys))


class Probe(Hook):
    """A simple probing hook used to retrieve and log a system snapshot with the experiment"""
    def __call__(self, var_dict, monitor):
        for k, v in var_dict.items():
            try:
                v.update_meta(get_cpu=True, get_gpu=True, save=True)
                string = ''
                cpu = v._meta.get('cpu', None)
                if cpu is not None:
                    cpu_use = sum(float(c.replace('%', '')) for c in cpu['usage']) / len(cpu['usage'])
                    string += f"cpu={cpu_use}%"
                gpu = v._meta.get('gpu', None)
                if gpu is not None:
                    string += '   '
                    string += '   '.join(f"{kk} = {vv['name']} {vv['load']} {vv['memory']} {vv['temperature']}"
                                       for kk, vv in gpu.items())
                    string += '   '
                monitor.log(string)
            except AttributeError:
                pass


class Logger(Hook):
    """A simple logging hook used to log variables to stdout.

    Example::

        from xmen.monitor import Monitor, Logger

        m = Monitor(
            hooks=[
                Logger('x@2s', process_func=lambda x: '|'.join(x)),
                Logger('y@1e', format='.5f')])

        x = ['cat', 'dog']
        y = 0.
        for _ in m(range(3)):
            for i in m(range(5)):
                y += i

        # [01:17PM 18/11/20 0/3 2/15]: x = cat|do
        # [01:17PM 18/11/20 0/3 4/15]: x = cat|do
        # [01:17PM 18/11/20 1/3]: y = 10.0000
        # [01:17PM 18/11/20 1/3 6/15]: x = cat|do
        # [01:17PM 18/11/20 1/3 8/15]: x = cat|do
        # [01:17PM 18/11/20 1/3 10/15]: x = cat|do
        # [01:17PM 18/11/20 2/3]: y = 20.0000
        # [01:17PM 18/11/20 2/3 12/15]: x = cat|do
        # [01:17PM 18/11/20 2/3 14/15]: x = cat|do
        # [01:17PM 18/11/20 3/3]: y = 30.0000

    """
    def __init__(self, spec, format='', process_func=None):
        """
        Args:
            spec (Spec): a specification string of the form "{regex}@{modulo}{trigger}". Variables matching
                ``regex`` will be logged when ``trigger`` % ``modulo`` == 0.
            format (str): a format string used as f"{var:format}" for logging string variables
            process_func (callable): used to convert variables to a string for logging to stdout. Format will be
                applied after.
        """
        super(Logger, self).__init__(spec)
        self.format = format
        self.process_func = process_func

    def __call__(self, var_dict, monitor, *args, format=None, process_func=None):
        """Log variables to standard out"""
        s = monitor.summary(verbose=1)
        triggers = [k for k in s if k in BRIEF.values()]
        elems = [f'{s["last"]}']
        elems += [f'{s[k]}' for k in triggers]
        string = '[' + ' '.join(elems) + ']:'
        # string = ' '.join(elems) + ':'
        # Log other arguments
        process_func = self.process_func if process_func is None else process_func
        format = self.format if format is None else format
        for k, v in var_dict.items():
            if process_func is not None:
                v = process_func(v)
            string += f' {k} = {v:{format}}'
        if len(var_dict) != 0:
            string = string[:-1]
        for v in args:
            if process_func is not None:
                v = process_func(v)
            string += f' {v:{format}}'
        print(string)


class TensorboardLogger(Hook):
    """A hook for logging tensorboard summaries. Currently ``image``, ``scalar``, ``histogram``, ``figure``,
    ``video``, ``text``, ``pr_curve``, ``mesh`` are all supported.

    Before being passed to the summary writer each variable is processes as follows:

        1. First all variables are passed to ``fn`` (if supplied).
        2. Variables of type list or dict with length K will be expanded to give K variables. The name of each variable
           will be the list name postpended with its index or the dictionary name postpended with its key.
        3. If the summary type is image or scalar then some additional processing will be performed:

            1. For scalars if variables are not already scalar variables they will be converted by calling var.mean()
            2. For images the variable must be either a 3D [C, H, W] or 4D [B, C, H, W] tensor. Tensors are converted
               to images of shape [C, H, W]:
            3. if the variable is 3D it will be converted to 4D
            4. the variable is then converted to an image using ``torchvision.utils.make_grid``. Additional options
               can be passed to ``torchvision.utils.make_grid`` using the ``options`` parameter. Available options
               include:

                - ``'nrow' # images per row``
                - ``'padding'``
                - ``'normalize'``
                - ``'range'``
                - ``'scale_each'``
                - ``'pad_value'``
                (see torchvision.utils.make_grid for more info)


    Note:
        Tensorboard does not allow summaries to have the same name. If you want to leave to different types
        of summary for the same variable then you will need to use the `prefix` argument.

    """
    def __init__(self, type, spec, fn=None, prefix='', prepend=True, **options):
        """
        Args:
            type (str): The type of tensorboard summary to log. Should be one of
                ['image', 'scalar', 'histogram', 'figure', 'video', 'text', 'pr_curve', 'mesh']
            spec (str): A string in the form ``"{regex}@{modulo}{trigger}"``. tensorbaord summaries will be logged
                for all variables matching ``regex`` when ``monitor.{trigger} % modulo == 0`
            fn (callable): A function used to convert each variable to a summary
            prefix (str): prepend all summaries with this string
            prepend (bool): If true prepend dictionary variable names with the dictionary name.
            **options: Keyword arguments passed to ``torchvision.utils.make_grid``
        """
        super(TensorboardLogger, self).__init__(spec)
        self.fn = fn
        self.type = type
        self.prefix = prefix
        self.prepend = prepend
        if type not in ['image', 'scalar', 'histogram', 'figure', 'video', 'text', 'pr_curve', 'mesh']:
            raise NotImplementedError("Only image and scalar summaries are currently supported.")
        self.options = options

    def __call__(self, var_dict, monitor):
        if len(var_dict) > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning, append=True)
                import torch.utils.tensorboard as tb

            if monitor.directory is None:
                monitor.log(f'WARNING: Cannot log {list(var_dict.keys())} to tensorboard as '
                            f'monitor does not have a directory')
            else:
                if not os.path.join(monitor.directory, 'summaries'):
                    os.makedirs(os.path.join(monitor.directory, 'summaries'))
                from torch.utils.tensorboard import SummaryWriter
                tb_writer = tb.SummaryWriter(os.path.join(monitor.directory, 'summaries'))
                monitor.log(f'saved tb {self.type} summaries for {list(var_dict.keys())} at {monitor.directory}')
                # print(monitor.directory, 'summaries')
                for k, v in var_dict.items():
                    k = self.prefix + k
                    if v is not None:
                        if self.fn is not None:
                            v = self.fn(v)
                        add_summary = getattr(tb_writer, 'add_' + self.type)
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                vv = self.make_compatible(vv)
                                p = k + '_' if self.prepend else ''
                                add_summary(p + kk, vv, global_step=monitor.step)
                        elif isinstance(v, (list, tuple)):
                            for i, vv in enumerate(v):
                                vv = self.make_compatible(vv)
                                add_summary(k + '_' + str(i), vv, global_step=monitor.step)
                        else:
                            v = self.make_compatible(v)
                            add_summary(k, v, global_step=monitor.step)
                tb_writer.flush()
                tb_writer.close()

    def make_compatible(self, v):
        """Convert the variable v to a valid input for the summary writer"""
        import torch
        import numpy as np
        from torchvision.utils import make_grid
        if hasattr(v, 'detach') and hasattr(v, 'clone'):
            v = v.detach().clone()
        if self.type == 'image':
            if isinstance(v, np.ndarray):
                v = torch.as_tensor(v)
            options = {'nrow', 'padding', 'normalize', 'range', 'scale_each', 'pad_value'}
            make_grid_params = {'normalize': True, 'scale_each': True}
            make_grid_params.update(
                {k: v for k, v in self.options.items() if k in options})
            if len(v.shape) == 4:
                v = make_grid(v.float(), **make_grid_params)
            elif len(v.shape) == 3:
                # Make images a grid if batched and normalise each
                v = make_grid([v], **make_grid_params)
        if self.type == 'scalar':
            if isinstance(v, (torch.Tensor, np.ndarray)) and len(v.shape) > 0:
                v = v.mean()
        return v


class StopWatch(object):
    """A simple timer class"""
    def __init__(self, name, length=None, time_format='.4f', date_format='%j %H:%M:%S'):
        """
        Args:
            name (str): the name of the stopwatch
            length (int): the number of steps to time for (can be None)
            time_format (str): display timings in this format
            date_format (str): display date in this format
        """
        self.name = name
        self.start_time = time.time()
        self.average = 0.
        self.delta = 0.
        self.length = length
        self.since_start = 0.
        self.n = 0.
        self._t_fmt = time_format
        self._date_fmt = date_format
        self.reference = 0.

    def start(self):
        """Start the stop watch"""
        self.reference = time.time()
        return self

    def stop(self):
        """Stop the stopwatch"""
        now_time = time.time()
        self.delta = now_time - self.reference
        self.average += (self.delta - self.average) / (self.n + 1)
        self.n += 1.
        return self

    def projected(self, n_more=None):
        """Get projected time to completion. If ``n_more is None`` then the time to completion will be inferred
        from ``length``. As a result at least of one of ``n_more`` or ``length`` must be set"""
        assert n_more is not None or self.length is not None, 'One of n_more or length must be set'
        if n_more is None:
            n_more = (self.length - self.n)
        return self.average * n_more

    def wall_time(self):
        """The time since the experiment started in seconds"""
        return time.time() - self.start_time

    def __repr__(self):
        """String representation"""
        string = f'{self.name} {self.delta:{self._t_fmt}}secs ({self.average:{self._t_fmt}} avg)'
        if self.length is not None:
            string += ' wall ' + time.strftime(self._date_fmt,  time.gmtime(time.time() - self.start_time))
            string += ' finish ' + time.strftime(self._date_fmt, time.gmtime(self.projected()))
        return string


def load(load_dir, step=None, attempt=True, **modules):
    """Load torch objects from a checkpoint object. If ``Step is None`` then the most recent checkpoint is loaded.
    Else the checkpoint is loaded at the specified step. If attempt == False then load will raise a ValueError
    if either one of the modules was not found in modules or if no checkpoints were found in load_dir.

    Note:
        No check to ensure the meta information in each file is the same. The meta information returned corresponds
        to the first module encountered in modules.
    """
    import torch
    if load_dir.endswith('/'):
        load_dir = load_dir[:-1]
    folders = glob.glob(load_dir + '/*')
    meta = {}
    found = []
    steps = []
    load_step = step
    if len(folders) != 0:
        for i, f in enumerate(folders):
            module_key = f.split('/')[-1]
            if module_key in modules:
                files = glob.glob(os.path.join(f, '*'))
                # Order files in ascending order by step (note sort must be used on int)
                _, files = zip(*sorted([(int(os.path.splitext(os.path.split(ff)[-1])[0]), ff) for ff in files]))
                if not len(files):
                    msg = f'No checkpoint found for {f}'
                    if attempt:
                        logging.warning(msg)
                        break
                    else:
                        raise RuntimeWarning(msg)
                if step is None:
                    file = files[-1]     # Take most recent
                    load_step = int(file.split('/')[-1].split('.')[0])
                else:
                    file = [f for f in files if str(step) in f]
                    if file:
                        file = file[0]
                    else:
                        msg = f'Could not find checkpoint at step {step} for {f}'
                        if attempt:
                            logging.warning(msg)
                            break
                        else:
                            raise RuntimeWarning(msg)
                checkpoint = torch.load(file)
                if i == 0:
                    meta.update({k: checkpoint.get(k, 0) for k in TRIGGERS})
                if 'state_dict' in checkpoint:
                    if hasattr(modules[module_key], 'load_state_dict'):
                        modules[module_key].load_state_dict(checkpoint['state_dict'])
                        found += [module_key]
                        steps += [load_step]
        logging.info(f"Loaded {found} at step {steps}")
        missing = set(modules.keys()).difference(found)
        if len(missing) != 0:
            msg = f'Keys {missing} were not found in the checkpoint'
            if attempt:
                logging.warning(msg)
            else:
                raise ValueError(msg)
    else:
        msg = f'No checkpoints exist in {load_dir}'
        if attempt:
            logging.warning(msg)
        else:
            raise RuntimeWarning(msg)
    return modules, meta


class Monitor(object):
    """Automate tracking and logging of experiments.

    Configured through arguments of the form ``f'{regex}@{modulo}{trigger}'``.
    Any variables in the local scope of the monitor which match the
    specified regular expression will be logged every modulo steps of the trigger. Triggers are
    incremented either manually (using the ``inc`` method) or automatically using nested iterators
    (see example). Supported triggers include ["step", "epoch", "era", "eon", "supereon"] or their
    abbreviations ["s", "e", "er", "eo", "se"].

    If a hook is passed with modulo == None it can instead be triggered manually
    as ``monitor.to_hooks()``.

    Example::

        from xmen import Experiment
        from xmen.monitor import Monitor

        X = Experiment(..., ...)

        a, b, c, d, e, f = 0, 0, 0, 0, 0, 0


        def identity(x):
            return x


        def mult(x):
            return 10 * x


        m = Monitor(
            log=('a|b@2s', 'b@1e', 'c@1er', "d|e@1eo", "e@1se"),
            log_fn=(mult, identity, identity, identity, identity),
            log_format='.3f',
            msg='a->X@1s',
            time=('@2s', '@1e'),
            probe='X@10s',
            limit='@20s')
        for _ in m(range(2)):
            for _ in m(range(2)):
                for _ in m(range(3)):
                    for _ in m(range(4)):
                        for _ in m(range(5)):
                            a += 1
                        b += 1
                    c += 1
                d += 1
            e += 1
    """
    def __init__(self, *, hooks=[],
                 log=None, log_fn=None, log_format='',
                 time=(),
                 msg=None, msg_keep='latest', msg_leader=None, msg_expand=False, msg_prep=None,
                 probe=None,
                 limit=None):
        """
        Args:
            hooks (Iterable[Hook]): User defined hooks used to extend the functionality of the Monitor class inheriting
                from ``Hook``.
            log (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to log @ a particular logging frequency to stdout.
            log_fn (Callable,  Iterable[Callable]): Converts the variable into a string for logging.
            log_format (str,  Iterable[str]): A format used to format a string as ``f"{string}:{format}"``
            time (str,  Iterable[str]): A string of the form ``f"@{steps}"`` to log timing statistics at (or list of for different
                triggers).
            msg (str,  Iterable[str]): A modulo string of the form ``"{regex}->{exp_regex}@{steps}s"`` (or list of)
                giving the variables to log as messages with the experiments matching ``exp_regex``.
            msg_keep (str, Iterable[str]): One of ['latest', 'max', 'min'] giving the protocol to use on message collision
            msg_leader (str, Iterable[str]): A regex to a single variable. If supplied then this variable will be treated as the leader
                and all other variables will be logged only if the keep condition is met for the leader
            msg_expand (bool, Iterable[str]): If True then dictionary variables with K keys will be expanded to give K variables
            msg_prep (bool, Iterable[str]): If True then each variable in the dictionary will be prepended by the dictionary name.
            probe (str, Iterable[str]): A string of the form ``f"{regex}@{steps}"`` to log resource use to each
                experiment that matches regex (or list of for different triggers).
            limit (str,  Iterable[str]): A modulo string of the form ``f"@{modulo}{triger}" used to limit the number of iterations of
                an experiment at a particular trigger level. This is useful if an experiment is restarted for
                example.

        Note:
            All variables ``ckpt_keep``, ``msg_leader``,
            ``msg_expand`` and ``msg_prep`` can be supplied as a single or as a list of entries, one for each set of
            variables matching each modulo string in each case."""
        self.hooks = []
        self.triggers = {k: 0 for k in TRIGGERS}
        self.log_regex = None
        self.timers = {}
        self.message = None
        self._limit_reached = False

        self.limit = read_modulo_string(limit)[1:] if limit is not None else None

        if log is not None:
            if not isinstance(log, (list, tuple)):
                log = [log]
            if not isinstance(log_fn, (list, tuple)):
                log_fn = tuple([log_fn] * len(log))
            if not isinstance(log_format, (list, tuple)):
                log_format = tuple([log_format] * len(log))
            warn = f"Either one fn, pref and prep or one for each '{log}' must be set"
            assert all(len(f) == len(log) for f in (log_fn, log_format)), warn
            for l, f, fmt in zip(log, log_fn, log_format):
                self.hooks.append(Logger(l, fmt, f))

        if time is not None:
            if not isinstance(time, (list, tuple)):
                time = [time]
            for t in time:
                self.hooks.append(Timer(t))
        if msg is not None:
            if not isinstance(msg, (list, tuple)):
                msg = [msg]
            if not isinstance(msg_keep, (list, tuple)):
                msg_keep = tuple([msg_keep] * len(msg))
            if not isinstance(msg_leader, (list, tuple)):
                msg_leader = tuple([msg_leader] * len(msg))
            if not isinstance(msg_prep, (list, tuple)):
                msg_prep = tuple([msg_prep] * len(msg))
            if not isinstance(msg_expand, (list, tuple)):
                msg_expand = tuple([msg_expand] * len(msg))
            warn = f"Either one fn, pref and prep or one for each '{msg}' must be set"
            assert all(len(f) == len(msg) for f in (msg_keep, msg_leader, msg_expand, msg_prep)), warn
            for m, k, l, ex, pr in zip(msg, msg_keep, msg_leader, msg_expand, msg_prep):
                self.hooks.append(XmenMessenger(m, k, l, ex, pr))

        if probe is not None:
            if not isinstance(probe, (list, tuple)):
                probe = [probe]
            for p in probe:
                self.hooks.append(Probe(p))

        # Add user hooks
        self.n_user_hooks = len(hooks)
        self.hooks.extend(hooks)

        # -- Use Logger and Check pointer for manual logging
        # These special hooks are always available
        self._logger = Logger(r'.*')

    @property
    def supereon(self):
        return self.triggers['supereon']

    @property
    def eon(self):
        return self.triggers['eon']

    @property
    def era(self):
        return self.triggers['era']

    @property
    def epoch(self):
        return self.triggers['epoch']

    @property
    def step(self):
        return self.triggers['step']

    def stop(self, msg):
        self._limit_reached = True
        self.log(msg)

    def __call__(self, iter, length=None, back=2):
        """Manage incrementing triggers, logging and collecting timing statistics around current the passed
        iterator.

        Args:
            iter: An iterator
            back: The number of calls back up to the frame from which called once inside ``inc()``.
            length: The length of the iterator (useful i length is known but iterator does not have attribute __len__)
        """
        if self._limit_reached:
            self.log(f'---- Stop Criterion @ ' + ', '.join(
                [f"{k}={v}" for k, v in self.triggers.items() if v != 0.]) + ' ----')
            return

        length = length if length is not None else len(iter) if hasattr(iter, '__len__') else None
        load, step = 'load', 'step'

        # Promote old timers
        if TRIGGERS[0] in self.timers:  # Setup (all triggers start at 0
            for i, (next_trigger, trigger) in enumerate(
                    zip(reversed(TRIGGERS[1:len(self.timers) + 1]),
                        reversed(TRIGGERS[:len(self.timers)]))):
                self.timers.update({next_trigger: copy.deepcopy(self.timers[trigger])})

        # Get current trigger level
        if len(self.timers) == 0 or TRIGGERS[0] in self.timers:
            current = TRIGGERS[0]
        else:
            current = TRIGGERS[[i for i, t in enumerate(TRIGGERS) if t in self.timers][0] - 1]

        # Add timing and logging for current trigger level
        self.timers[current] = {load: StopWatch(load, length), step: StopWatch(step, length)}

        def trigger():
            return [t for t in TRIGGERS if t in self.timers][0]

        self.timers[trigger()][load].start()
        for x in iter:
            if self.limit is not None and trigger() == self.limit[1] and self.triggers[trigger()] == self.limit[0]:
                self.stop(f'---- Limit reached for {self.limit[1]} {self.limit[0]} ----')
                # self._limit_reached = True
                # print(f'Limit reached for {self.limit[1]} {self.limit[0]}')
                break
            elif self._limit_reached:
                break
            else:
                self.timers[trigger()][load].stop()
                self.timers[trigger()][step].start()
                yield x
                self.timers[trigger()][step].stop()
                self.inc(trigger(), back)
                self.timers[trigger()][load].start()

        # Remove timers and hooks for trigger
        trigger = trigger()
        self.timers.pop(trigger)

    def inc(self, trigger, back=1):
        """Manually increment trigger. Will also run modulo hooks defined by the user."""
        if trigger not in TRIGGERS:
            raise NotImplementedError(f'{trigger} is not in {TRIGGERS}')
        self.triggers[trigger] += 1
        f = inspect.currentframe()
        for _ in range(back):
            f = f.f_back
        possible_vars = {k: v for k, v in f.f_locals.items()}

        for hook in self.hooks:
            if hook.modulo is not None and self.modulo(trigger, hook.modulo) and hook.trigger == trigger:
                matches = {}
                if hook.regex is not None:
                    matches = {
                        k: v for k, v in possible_vars.items() if re.match(hook.regex, k) is not None}
                hook(matches, self)

    def to_hooks(self, **kwargs):
        """Pass all keyword arguments to manual hooks (hooks where modulo is None)"""
        for hook in self.hooks:
            if hook.modulo is None:
                matches = {
                    k: v for k, v in kwargs.items() if re.match(hook.regex, k) is not None}
                hook(matches, self)

    def log(self, x, format='', process_func=str):
        """Log string with time, epoch and step.

        Args:
            x: Value to be logged
            format: The format used to convert x to a string as '{x:{format}}'
            process_func: A callable able to convert x to a valid format for printing as {x:{format}}'
        """
        self._logger({}, self, x, format=format, process_func=process_func)

    def modulo(self, trigger, modulo, exclude_1st=False):
        """Check if trigger % modulo == 0. If exclude_1st is True then modulo will return False for triggers at 0"""
        is_trigger = self.triggers[trigger] % modulo == 0
        if exclude_1st:
            is_trigger = is_trigger and self.triggers[trigger] > 0
        return is_trigger

    def summary(self, verbose=0):
        """Summarise the current state of the experiment monitor"""
        # Get statistics
        timings = {}
        summaries = {'last': datetime.datetime.now().strftime("%I:%M%p %d/%m/%y")}

        if len(self.timers) > 0:
            to_date = lambda x: str(datetime.timedelta(seconds=x))

            k, N, length, wall = zip(*[(
                k,
                self.timers[k]['step'].n,
                self.timers[k]['step'].length,
                self.timers[k]['step'].wall_time())
                for k in reversed(TRIGGERS) if k in self.timers])

            so_far, total = 0., [1.]
            offset = [l for l in length[1:]] + [1]
            wall = wall[0]
            if length[0] is not None:
                for k, (n, lnow, lnext) in enumerate(zip(N, length, offset)):
                    so_far += n
                    total += [total[k] * lnow]
                    if lnext is None:
                        break
                    else:
                        so_far *= lnext
                if so_far == 0.:
                    # There is a case once an inner iterator terminates for the
                    # first time meaning that the higher order iterator has yet to be triggered,
                    # left == 0. and a ZeroDivisionError will occur. It could be assumed that the trigger is just
                    # about to be triggered and left promoted to 1.0. In the
                    # case of iterators with a lot of work after the inner iterator has terminated then
                    # this timing estimate could be inaccurate. Instead when this case occurs (which is a minority)
                    # I ignore the timing.
                    lnext = None
                if lnext is not None:
                    left = (wall / so_far) * (total[-1] - so_far)
                else:
                    left = None
            else:
                left = None

            length = list(length)
            length[:len(total)] = total[1:]

            for i, (trigger, timer) in enumerate([(k, self.timers[k]) for k in reversed(TRIGGERS) if k in self.timers]):
                step_timer = timer['step']
                val = f'{int(self.triggers[trigger])}'
                if step_timer.length is not None:
                    val += f'/{int(length[i])}'
                    # val += f'%{step_timer.length}'
                summaries.update({BRIEF[trigger]: val})

            if verbose:
                # Get trigger with highest precedance
                trigger, timer = [(k, self.timers[k]) for k in TRIGGERS if k in self.timers][0]
                step_timer, load_timer = timer['step'], timer.get('load', None)
                if load_timer is not None:
                    if step_timer.delta != 0.:
                        if step_timer.length is not None:
                            timings.update(
                                {'next': to_date(
                                    (step_timer.average + load_timer.average) * (step_timer.length - step_timer.n))})
                        timings.update(
                            {'step': to_date(step_timer.delta),
                             'm_step': to_date(step_timer.average),
                             'load': to_date(load_timer.delta),
                             'm_load': to_date(load_timer.average)})
                else:
                    if step_timer.length is not None:
                        timings.update(
                            {'next': to_date(
                                step_timer.average * (step_timer.length - step_timer.n))})
                    timings.update(
                        {'step': to_date(step_timer.delta),
                         'm_step': to_date(step_timer.average)})
            summaries.update({'wall': to_date(wall)})
            if left is not None:
                summaries.update({'end': to_date(left)})
            summaries.update(timings)
        return summaries

    def __repr__(self):
        summary = self.summary(1)
        string = f'triggers: {self.triggers}'
        if len(summary) > 0:
            string += f', current state: {summary}'
        return string


class TorchMonitor(Monitor):
    """Automate tracking, logging and saving of experiments with additional
    hooks for logging to tensorboard and checkpointing of experiments.

    Manual logging and checkpoint are also supported as
    ``monitor.log(...)`` and ``monitor.checkpoint(...)``"""

    def __init__(self, directory=None, *,
                 # User defined hooks (either) modulo or not modulo
                 hooks=[],
                 # Default saving for parameters
                 ckpt=None, ckpt_keep=None,
                 log=None, log_fn=None, log_format='',
                 img=None, img_fn=None, img_pref='', img_prep=True, img_options={},
                 sca=None, sca_fn=None, sca_pref='', sca_prep=True,
                 hist=None, hist_fn=None, hist_pref='', hist_prep=True,
                 fig=None, fig_fn=None, fig_pref='', fig_prep=True,
                 txt=None, txt_fn=None, txt_pref='', txt_prep=True,
                 vid=None, vid_fn=None, vid_pref='', vid_prep=True,
                 time=(),
                 msg=None, msg_keep='latest', msg_leader=None, msg_expand=False, msg_prep=None,
                 probe=None,
                 limit=None):
        """
        Args:
            directory (str): The directory used to log checkpoints in
            hooks (Iterable[Hook]): User defined hooks used to extend the functionality of the Monitor class
            ckpt (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to checkpoint @ a particular logging frequency to stdout. The regex much match
                objects inheriting form torch.Module.
            ckpt_keep (int, Iterable[int]): The number of checkpoints to keep. The most recent checkpoints will be kept. If None
                then all checkpoints will be kept.
            log (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to log @ a particular logging frequency to stdout.
            log_fn (Callable, Iterable[Callable]): Converts the variable into a string for logging.
            log_format (str, Iterable[str]): A format used to format a string as ``f"{string}:{format}"``
            time (str, Iterable[str]): A string of the form ``f"@{steps}"`` to log timing statistics at (or list of for different
                triggers).
            img (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard images @ a particular logging frequency.
            img_fn (Callable, Iterable[Callable]): Converts the variable into an image of shape [B, C, H, W] or [C, H, W] for tensorboard.
                See TensorBoardLogger for more details in terms of automatic-processing. If ``img`` is a list then
                ``img_fn`` can also be passed as list with a callable for each entry in ``img`` or can be passed as
                a single callable used for all entries.
            img_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            img_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            img (str, Iterable[str]): A modulo string of the form ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard scalars @ a particular logging frequency.
            sca_fn (Callable, Iterable[Callable]): Converts the variable into a scalar for tensorboard.
                See TensorBoardLogger for more details in terms of automatic-processing. If ``sca`` is a list then
                ``sca_fn`` can also be passed as list with a callable for each entry in ``sca`` or can be passed as
                a single callable used for all entries.
            sca_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            sca_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            hist (str, Iterable[str]): A modulo string of the form ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard histograms @ a particular logging frequency.
                If no logging frequency is supplied then any variable logged in the experiment which matches
                ``regex`` will be logged to tensorboard each time it is passed to the logger.
                This is useful for logging variables at the end of an epoch for example.
            hist_fn (Callable, Iterable[Callable]): Preprocess variable before logging to tensorboard.
                If ``hist`` is a list then ``hist_fn`` can also be passed as list with a callable for each entry in
                ``hist`` or can be passed as a single callable used for all entries.
            hist_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            hist_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            fig (str, Iterable[str]): A modulo string of the form or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard figures @ a particular logging frequency.
            fig_fn (Callable, Iterable[Callable]): Preprocess variable before logging to tensorboard into a plt.figure().
                If ``fig`` is a list then ``fig_fn`` can also be passed as list with a callable for each entry in
                ``fig`` or can be passed as a single callable used for all entries.
            fig_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            fig_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            txt (str): A modulo string of the form ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard text @ a particular logging frequency.
            txt_fn (Callable, Iterable[Callable]): Preprocess variable before logging to tensorboard.
                If ``txt`` is a list then ``txt_fn`` can also be passed as list with a callable for each entry in
                ``txt`` or can be passed as a single callable used for all entries.
            txt_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            txt_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            vid (str, Iterable[str]): A modulo string of the form ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard videos @ a particular logging frequency.
            vid_fn (Callable, Iterable[Callable]): Preprocess variable to a tensor of shape [B, T, C, H, W] before logging to tensorboard.
                If ``vid`` is a list then ``vid_fn`` can also be passed as list with a callable for each entry in
                ``vid`` or can be passed as a single callable used for all entries.
            vid_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            vid_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            msg (str, Iterable[str]): A modulo string of the form ``"{regex}->{exp_regex}@{steps}s"`` (or list of)
                giving the variables to log as messages with the experiments matching ``exp_regex``.
            msg_keep (str, Iterable[str]): One of ['latest', 'max', 'min'] giving the protocol to use on message collision
            msg_leader (str, Iterable[str]): A regex to a single variable. If supplied then this variable will be treated as the leader
                and all other variables will be logged only if the keep condition is met for the leader
            msg_expand (bool, Iterable[bool]): If True then dictionary variables with K keys will be expanded to give K variables
            msg_prep (bool, Iterable[bool]): If True then each variable in the dictionary will be prepended by the dictionary name.
            probe (str, Iterable[str]): A string of the form ``f"{regex}@{steps}"`` to log resource use to each
                experiment that matches regex (or list of for different triggers).
            limit (str): A modulo string of the form ``f"@{modulo}{triger}" used to limit the number of iterations of
                an experiment at a particular trigger level. This is useful if an experiment is restarted for
                example.

        Note:
            All variables  `..._fn`, `..._pref` and `..._prep` as well as ``ckpt_keep`` and ``msg_leader``,
            ``msg_expand`` and ``msg_prep`` can be supplied as a single or as a list of entries, one for each set of
            variables matching each modulo string in each case.


        Example 1::

            nn, opt, dataset = ..., ...

            m = Monitor(
                directory,
                checkpoint=('model@1e', 'opt@100s'),   # Checkpoint the model once per epoch and opt every 100 steps
                log='^loss$@100s',   # Log the loss to stdout every 100 steps
                img='^x$@1000s', sca=('^loss$@100s', 'eval_.*@1e'), time=('@100s')     # Log to tensorboard
                time=('@100s', ),   # Generate timing statistics every 100 steps
                hooks=[  # Custom hooks are also supported
                    MyVeryOwnHook(...)])

            # The only modification needed to the training loop are the em calls.
            # Nested loops corresponds to different triggers from inside out
            # we have ["step" or "s", "epoch" or "e", "era" or "er", "eon" or "eo", "supereon" or "se"]
            for epoch in m(range(10)):
                for x, y in m(datset):
                    _y_ = model(x)
                    opt.zero_grad()
                    loss = loss_fn(y, _y_)
                    loss.backward()
                    loss.step()
                    em.log('Manual Logging is also supported')
                eval_1, eval_2 = eval(model, ds)

            # Steps and epoch have been incremented
            assert em.step == len(ds) * 10
            assert em.epoch == 10

            # Lets reload the model at the 5th epoch
            em.load(step=5*len(ds), model)
            # The step and epoch will be updated
            print(em.step, em.epoch)

        Example 2::

            from xmen.monitor import Monitor
            import numpy as np
            import torch
            import os
            import matplotlib.pyplot as plt
            from torchvision.datasets.mnist import MNIST
            from torch.utils.data import DataLoader
            import torchvision.transforms as T

            plt.style.use('ggplot')
            ds = DataLoader(MNIST(os.getenv("HOME") + '/data/mnist', download=True,
                  transform=T.Compose(
                      [T.Resize([64, 64]), T.CenterCrop([64, 64]),
                       T.ToTensor(), T.Normalize([0.5], [0.5])])), 8)

            m = Monitor(
                directory='/tmp/tb_5',
                sca=['^z$|^X$@10s', '^a|^x$@5s'],
                img=['^mnist@10s', '^mnist@5s'], img_fn=[lambda x: x[:2], lambda x: x[:5]], img_pref=['2', '5'],
                hist='Z@1s',
                fig='fig@10s',
                txt='i@5s', txt_fn=lambda x: f'Hello at step {x}',
                vid='^mnist@10s', vid_fn=lambda x: (x.unsqueeze(0) - x.min()) / (x.max() - x.min())
            )

            # variables
            x = 0.
            a = [1, 2]
            z = {'x': 5, 'y': 10}
            for i, (mnist, _) in m(zip(range(31), ds)):
                # plot a figure
                fig = plt.figure(figsize=[10, 5])
                plt.plot(np.linspace(0, 1000), np.cos(np.linspace(0, 1000) * i))
                # random tensor
                Z = torch.randn([10, 3, 64, 64]) * i / 100
                # scalars
                x = (i - 15) ** 2
                z['i'] = i
                z['x'] += 1
                z['y'] = z['x'] ** 2

        """
        super().__init__(hooks=hooks, log=log, log_fn=log_fn, log_format=log_format, time=time,
                         msg=msg, msg_keep=msg_keep, msg_leader=msg_leader, msg_expand=msg_expand, msg_prep=msg_prep,
                         probe=probe,
                         limit=limit)

        self.directory = directory

        # -- Add modulo hooks to hooks
        if ckpt is not None:
            if not isinstance(ckpt, (list, tuple)):
                ckpt = [ckpt]
            if not isinstance(ckpt_keep, (list, tuple)):
                ckpt_keep = tuple([ckpt_keep] * len(ckpt))
            warn = f"Either one fn, pref and prep or one for each '{ckpt_keep}' must be set"
            assert all(len(f) == len(ckpt) for f in (ckpt_keep, )), warn
            for c, k in zip(ckpt, ckpt_keep):
                self.hooks.append(Checkpointer(c, to_keep=k))

        # tensorboard summaries
        kinds = {'sca': 'scalar', 'img': 'image', 'hist': 'histogram',
                 'fig': 'figure', 'txt': 'text', 'vid': 'video'}
        for k in kinds:
            kind = locals()[k]
            if kind is not None:
                options = {}
                if k == 'img':
                    options = locals()[k + '_options']
                fn = locals()[k + '_fn']
                pref = locals()[k + '_pref']
                prep = locals()[k + '_prep']
                if not isinstance(kind, (list, tuple)):
                    kind = [kind]
                if not isinstance(fn, (list, tuple)):
                    fn = tuple([fn] * len(kind))
                if not isinstance(pref, (list, tuple)):
                    pref = tuple([pref] * len(kind))
                if not isinstance(prep, (list, tuple)):
                    prep = tuple([prep] * len(kind))
                warn = f"Either one fn, pref and prep or one for each '{kind}' must be set"
                assert all(len(f) == len(kind) for f in (fn, pref, prep)), warn
                for c, f, pr, pre in zip(kind, fn, pref, prep):
                    self.hooks.append(TensorboardLogger(kinds[k], c, fn=f, prefix=pr, prepend=pre, **options))

        self._checkpointer = Checkpointer(r'.*', to_keep=ckpt_keep)

    def checkpoint(self, **kwargs):
        """Checkpoint the torch.nn objects with step and epoch passed as ``name==variable_to_save``"""
        self._checkpointer(kwargs, self)

    def load(self, directory=None, step=None, attempt=True, update_triggers=True, **modules):
        """Load the torch torch.nn objects passed as name=variable_to_load,
        from the directory and reset the state of the em (if update_triggers == True). If attempt == False then an
        Exception will be raised if either the directory does not contain checkpoints corresponding to modules.
        """
        if directory is None:
            if self.directory is None:
                self.log(f'WARNING: Cannot load as monitor does not have a directory')
                return {}
            else:
                directory = os.path.join(self.directory, 'checkpoints')

        modules, triggers = load(directory, step=step, attempt=attempt, **modules)
        if update_triggers:
            self.triggers.update(triggers)
        return modules
