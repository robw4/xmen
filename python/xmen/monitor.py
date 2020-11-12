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


class Hook(object):
    def __init__(self, regex=None, modulo=None, trigger='step'):
        if '@' in regex:
           regex, modulo, trigger = read_modulo_string(regex)
        self.regex = regex
        self.modulo = modulo
        self.trigger = trigger
        if trigger not in TRIGGERS:
            raise NotImplementedError(f'Trigger {trigger} is not in {TRIGGERS}')

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
    def __init__(self, regex='.*', modulo=None, trigger='step', to_keep=None, expand=True):
        super(Checkpointer, self).__init__(regex, modulo, trigger)
        self.to_keep = to_keep
        self._checkpoint_buffer = {}
        self.expand = expand

    def __call__(self, model_dict, monitor):
        import torch
        from xmen.utils import get_version
        saved = []

        if self.expand:
            pops = []
            updates = {}
            for k, v in model_dict.items():
                if isinstance(v, dict):
                    pops.append(k)
                    updates.update({k + '_' + kk: vv for kk, vv in v.items()})
            for p in pops:
                model_dict.pop(p)
                model_dict.update(updates)

        for k, v in model_dict.items():
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
    def __init__(self, exp, regex, modulo=None, trigger=None, keep='latest', leader=None, expand=False, prepend=None):
        assert keep in ['max', 'min', 'latest']
        self.experiments = exp
        self.keep = keep
        self.expand = expand
        self.leader = leader
        self.prepend = prepend
        if regex not in ['', '^$', None]:
            exp += '|' + regex
        super(XmenMessenger, self).__init__(regex=exp, modulo=modulo, trigger=trigger)

    def __call__(self, var_dict, monitor):
        from xmen.experiment import Experiment
        # Get and remove experiments from var_dict
        results = zip(*[(k, var_dict[k]) for k in var_dict if re.match(self.experiments, k) is not None])
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
    def __call__(self, var_dict, monitor):
        s = monitor.summary(verbose=1)
        keys = [k for k in s if k not in BRIEF.values() and k != 'last']
        monitor.log(' '.join(f'{k}={s[k]}' for k in keys))


class Logger(Hook):
    # Logging options
    def __init__(self, regex=None, modulo=None, format='', trigger='step', timings=False, process_func=None):
        super(Logger, self).__init__(regex=regex, modulo=modulo, trigger=trigger)
        self.format = format
        self.timings = timings
        self.process_func = process_func

    def __call__(self, var_dict, monitor, *args, format=None, process_func=None):
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
        logging.info(string)


class TensorboardLogger(Hook):
    def __init__(self, type, regex, modulo=None, trigger='step', process_func=None, prefix='',
                 image_n=None, scalar_reduce=True, prepend=True, **options):
        super(TensorboardLogger, self).__init__(regex, modulo, trigger)
        self.process_func = process_func
        self.type = type
        self.prefix = prefix
        self.image_n = image_n
        self.scalar_reduce = scalar_reduce
        self.n_images = 1
        self.prepend = prepend
        if type not in ['image', 'scalar', 'histogram', 'figure', 'video', 'text', 'pr_curve', 'mesh']:
            raise NotImplementedError("Only image and scalar summaries are currently supported.")
        self.options = options

    def __call__(self, var_dict, monitor):
        if len(var_dict) > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning, append=True)
                import torch.utils.tensorboard as tb

            if not os.path.join(monitor.directory, 'summaries'):
                os.makedirs(os.path.join(monitor.directory, 'summaries'))
            tb_writer = tb.SummaryWriter(os.path.join(monitor.directory, 'summaries'))
            monitor.log(f'saved tb {self.type} summaries for {list(var_dict.keys())} at {monitor.directory}')
            # print(monitor.directory, 'summaries')
            for k, v in var_dict.items():
                v = v.detach().clone()
                k = self.prefix + k
                if v is not None:
                    if self.process_func is not None:
                        v = self.process_func(v)
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
        import torch
        import numpy as np
        from torchvision.utils import make_grid
        if self.type == 'image':
            options = {'nrow', 'padding', 'normalize', 'range', 'scale_each', 'pad_value'}
            make_grid_params = {'normalize': True, 'scale_each': True}
            make_grid_params.update(
                {k: v for k, v in self.options.items() if k in options})
            if len(v.shape) == 4:
                if self.image_n is not None:
                    v = v[:self.image_n]
                v = make_grid(v.float(), **make_grid_params)
            elif len(v.shape) == 3:
                # Make images a grid if batched and normalise each
                v = make_grid([v], **make_grid_params)
        if self.type == 'scalar' and self.scalar_reduce:
            if isinstance(v, (torch.Tensor, np.ndarray)) and len(v.shape) > 0:
                v = v.mean()
        return v

    def get_summary_writer(self, v, tb_writer):
        return


class StopWatch(object):
    def __init__(self, name, length=None, time_format='.4f', date_format='%j %H:%M:%S'):
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
        self.reference = time.time()
        return self

    def stop(self):
        now_time = time.time()
        self.delta = now_time - self.reference
        self.average += (self.delta - self.average) / (self.n + 1)
        self.n += 1.
        return self

    def projected(self, n_more=None):
        if n_more is None:
            n_more = (self.length - self.n)
        return self.average * n_more

    def wall_time(self):
        return time.time() - self.start_time

    def __repr__(self):
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
    def __init__(self, directory=None, *,
                 # User defined hooks (either) modulo or not modulo
                 hooks=[],
                 # Default saving for parameters
                 checkpoint=None, checkpoints_to_keep=None,
                 log=None,
                 img=None, img_fn=None, img_n=None,
                 sca=None, sca_fn=None,
                 time=('@100s', '@1e', '@1er', '@1eo', '@1se'),
                 message=tuple(f'self@100{t}' for t in TRIGGERS),
                 limit=None):
        """Automate tracking, logging and saving of experiments.

        Logging / checkpointing is configured through arguments of the form ``f'{regex}@{modulo}{trigger}'``.
        Any variables in the local scope of the monitor which match the
        specified regular expression will be logged / checkpointed every modulo steps of the trigger. Triggers are
        incremented either manually (using the ``inc`` method) or automatically using nested iterators
        (see example). Supported triggers include ["step", "epoch", "era", "eon", "supereon"] or their
        abbreviations ["s", "e", "er", "eo", "se"].

        Alongside the built in logging / checkpointing options, more sophisticated cases can be configured
        through user defined hooks. If a hook is passed with modulo == None it can instead be triggered manually
        as ``monitor.to_hooks()``. Manual logging and checkpoint are also supported as
        ``monitor.log(...)`` and ``monitor.checkpoint(...)``

        Args:
            directory: A path to store checkpoints / logs in (required for checkpointing and tensorboard logging
                otherwise optional)
            checkpoint: checkpoint specified torch.Module objects
            log: log specified objects on triggers to stdout
            img: log as tensorboard images
            img_fn: Apply this function to each variable before logging as images
            sca: log as tensorboard scalars
            sca_fn: Apply this function to each variable before logging as scalars
            time: log timing statistics at these frequencies
            hooks: A list of user specified hooks conforming the the ``Hook`` api.
            message: summarise the current state of the experiment at these frequencies using the xmen.Experiment api.

        Example::

            nn, opt, dataset = ..., ...

            em = Monitor(
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
            for epoch in em(range(10)):
                for x, y in em(datset):
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

        """
        # Hooks work in both manual and modulo mode
        self.hooks = []
        self.directory = directory
        self.triggers = {k: 0 for k in TRIGGERS}
        self.log_regex = None
        self.timers = {}
        self.message = None
        self._limit_reached = False

        self.limit = read_modulo_string(limit)[1:] if limit is not None else None

        # -- Add modulo hooks to hooks
        if checkpoint is not None:
            if not isinstance(checkpoint, (list, tuple)):
                checkpoint = [checkpoint]
            for c in checkpoint:
                regex, modulo, trigger = read_modulo_string(c)
                self.hooks.append(Checkpointer(modulo=modulo, regex=regex, trigger=trigger,
                                               to_keep=checkpoints_to_keep))

        if log is not None:
            if not isinstance(log, (list, tuple)):
                log = [log]
            for l in log:
                regex, modulo, trigger = read_modulo_string(l)
                self.hooks.append(Logger(modulo=modulo, regex=regex, trigger=trigger))

        if isinstance(sca, bool) and sca:
            # This allows scalars to follow log
            sca = log
        if sca is not None:
            if not isinstance(sca, (list, tuple)):
                sca = [sca]
            for s in sca:
                regex, modulo, trigger = read_modulo_string(s)
                self.hooks.append(
                    TensorboardLogger(regex=regex, modulo=modulo, trigger=trigger,
                                      type='scalar', process_func=sca_fn))

        if img is not None:
            if not isinstance(img, (list, tuple)):
                img = [img]
            for i in img:
                regex, modulo, trigger = read_modulo_string(i)
                self.hooks.append(
                    TensorboardLogger(regex=regex, modulo=modulo, trigger=trigger,
                                      type='image', process_func=img_fn, image_n=img_n))

        if time is not None:
            if not isinstance(time, (list, tuple)):
                time = [time]
            for t in time:
                _, modulo, trigger = read_modulo_string(t)
                self.hooks.append(Timer(regex=r'^$', modulo=modulo, trigger=trigger))
        if message is not None:
            if not isinstance(message, (list, tuple)):
                message = [message]
            for m in message:
                if '->' in m:
                    # regex->exps@modulo
                    regex, m = m.split('->')
                else:
                    regex = ''
                exp, modulo, trigger = read_modulo_string(m)
                self.hooks.append(XmenMessenger(exp=exp, regex=regex, modulo=modulo, trigger=trigger))
        # Add user hooks
        self.n_user_hooks = len(hooks)
        self.hooks.extend(hooks)

        # -- Use Logger and Check pointer for manual logging
        # These special hooks are always available
        self._logger = Logger(regex=r'.*', timings=False)
        self._checkpointer = Checkpointer(regex='r.*', to_keep=checkpoints_to_keep)

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

    def checkpoint(self, **kwargs):
        """Checkpoint the torch.nn objects with step and epoch passed as ``name==variable_to_save``"""
        self._checkpointer(kwargs, self)

    def load(self, directory=None, step=None, attempt=True, update_triggers=True, **modules):
        """Load the torch torch.nn objects passed as name=variable_to_load,
        from the directory and reset the state of the em (if update_triggers == True). If attempt == False then an
        Exception will be raised if either the directory does not contain checkpoints corresponding to modules.
        """
        if directory is None:
            directory = os.path.join(self.directory, 'checkpoints')
        modules, triggers = load(directory, step=step, attempt=attempt, **modules)
        if update_triggers:
            self.triggers.update(triggers)
        return modules

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
                step_timer, load_timer = timer['step'], timer['load']
                val = f'{int(self.triggers[trigger])}'
                if step_timer.length is not None:
                    val += f'/{int(length[i])}'
                    # val += f'%{step_timer.length}'
                summaries.update({BRIEF[trigger]: val})

            if verbose:
                # Get trigger with highest precedance
                trigger, timer = [(k, self.timers[k]) for k in TRIGGERS if k in self.timers][0]
                step_timer, load_timer = timer['step'], timer['load']
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




