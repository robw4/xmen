"""Helper classes for interfacing with pytorch_lighnting. Will also need to have pytorch_lightning installed
to work with this implementation."""

import pytorch_lightning as pl

from pytorch_lightning.utilities import rank_zero_only

from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
import xmen
from xmen.monitor import TorchMonitor
from xmen.monitor import StopWatch
import re

from typing import Iterable, Union, Optional, List, Dict
from pytorch_lightning.callbacks import Callback
# from pytorch_lightning.accelerators import Accelerator


class Trainer(pl.Trainer):
    """A light weight wrapper around the pl.Trainer with some minor modifications:

        - If using with a logger of type ``XmenLogger`` then:
            - ``log_ever_n_steps`` must be set to 1
            - it cannot be used in combination with other loggers
            - the ``metrics_to_scalars`` method will be overloaded to simply return the inputs in ``metrics`` with
                the length of the data loader and the number of epochs added to ``metrics``

    Else, everything else should be the same apart from the changed defaults of ``log_every_n_steps`` and
    ``refresh_progress_bar``.

    Note:
        This is a hack in order to work around the current Lightning Logger implementation. When self.log(...)
        is called from within a lightning module the metrics and values are passed to Trainer which in turn
        interfaces with the logger through the its method ``logger_connector`` of type
        ``LoggerConnector`` with processing defined in ``TrainerLoggingMixin.metrics_to_scalars()`` which is
        overloaded here to simply return the original metrics without converting them to scalars.

        As far as I can tell the only place ``metrics_to_scalars`` is called is in ``LoggerConnector.log_metrics()``
        before passing the metrics to the user defined logger in the line::

            self.trainer.logger.agg_and_log_metrics(scalar_metrics, step=step)


        However, I am not sure what side effects this will have. In particular from what I can scalar_metrics
        is also preserved in state as::

            self.logged_metrics.update(scalar_metrics)
            self.trainer.dev_debugger.track_logged_metrics_history(scalar_metrics)

        In the first case, this seems to simply update a dictionary of logged variables which is probably fine unless
        the variables in ``scalar_metrics`` are very large. In the second case ``scalar_metrics`` are appended
        which is none ideal. Luckily this only occurs when running in debug mode.
    """
    def __init__(
            self,
            default_root_dir=None,
            logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
            checkpoint_callback: bool = True,
            callbacks: Optional[List[Callback]] = None,
            gradient_clip_val: float = 0,
            process_position: int = 0,
            num_nodes: int = 1,
            num_processes: int = 1,
            gpus: Optional[Union[List[int], str, int]] = None,
            auto_select_gpus: bool = False,
            tpu_cores: Optional[Union[List[int], str, int]] = None,
            log_gpu_memory: Optional[str] = None,
            progress_bar_refresh_rate: int = 0,
            overfit_batches: Union[int, float] = 0.0,
            track_grad_norm: Union[int, float, str] = -1,
            check_val_every_n_epoch: int = 1,
            fast_dev_run: bool = False,
            accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
            max_epochs: int = 1000,
            min_epochs: int = 1,
            max_steps: Optional[int] = None,
            min_steps: Optional[int] = None,
            limit_train_batches: Union[int, float] = 1.0,
            limit_val_batches: Union[int, float] = 1.0,
            limit_test_batches: Union[int, float] = 1.0,
            val_check_interval: Union[int, float] = 1.0,
            flush_logs_every_n_steps: int = 100,
            log_every_n_steps: int = 1,
            accelerator = None,
            sync_batchnorm: bool = False,
            precision: int = 32,
            weights_summary: Optional[str] = 'top',
            weights_save_path: Optional[str] = None,
            num_sanity_val_steps: int = 2,
            truncated_bptt_steps: Optional[int] = None,
            resume_from_checkpoint: Optional[str] = None,
            profiler=None,
            benchmark: bool = False,
            deterministic: bool = False,
            reload_dataloaders_every_epoch: bool = False,
            auto_lr_find: Union[bool, str] = False,
            replace_sampler_ddp: bool = True,
            terminate_on_nan: bool = False,
            auto_scale_batch_size: Union[str, bool] = False,
            prepare_data_per_node: bool = True,
            plugins: Optional[list] = None,
            amp_backend: str = 'native',
            amp_level: str = 'O2',
            distributed_backend: Optional[str] = None,
            automatic_optimization: bool = True):
        """The same as pytorch lightning Trainer class with defaults changed for ``progress_bar_refresh_rate``
        and ``log_every_n_steps``"""
        import inspect
        keys = set(inspect.signature(Trainer).parameters.keys())
        kwargs = {k: v for k, v in locals().items() if k in keys}
        super(Trainer, self).__init__(**kwargs)

        if isinstance(logger, Iterable) and any(isinstance(l, TensorBoardLogger) for l in logger):
            raise ValueError('XmenLogger is not currently supported with other Loggers but can be'
                             ' extended using the hooks functionality in xmen.')

        self._dont_reduce = False
        if isinstance(logger, TensorBoardLogger):
            self._dont_reduce = True

    def metrics_to_scalars(self, metrics):
        """Overloaded to let xmen logger log tensor variables."""
        if self._dont_reduce:
            metrics.update({
                'n_steps': len(self.train_dataloader) if self.train_dataloader is not None else None,
                'n_epochs': self.max_epochs,
                'epoch': self.current_epoch
            })
            return metrics
        else:
            return super().metrics_to_scalars(metrics)


class TensorBoardLogger(TensorBoardLogger):
    """An Xmen Tensorboard Logger implentation for interfacing with the pytorch_lighting. Unlike the ``Monitor`` class
    TensorBoardLogger only supports step triggers. Other triggers will remain unused in the Monitor class.

    Example::

        from pytorch_lightning import LightningModule
        import torch
        import torch.nn.functional as F
        from typing import List, Any

        from xmen.lightning import TensorBoardLogger, Trainer


        class LitMNIST(LightningModule):
            def __init__(self):
                super().__init__()
                # mnist images are (1, 28, 28) (channels, width, height)
                self.layer_1 = torch.nn.Linear(28 * 28, 128)
                self.layer_2 = torch.nn.Linear(128, 256)
                self.layer_3 = torch.nn.Linear(256, 10)

            def forward(self, x_in):
                batch_size, channels, width, height = x_in.size()
                # (b, 1, 28, 28) -> (b, 1*28*28)
                x = x_in.view(batch_size, -1)
                x = self.layer_1(x)
                x = F.relu(x)
                x = self.layer_2(x)
                x = F.relu(x)
                x = self.layer_3(x)
                x = F.log_softmax(x, dim=1)
                return x

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=1e-3)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = F.nll_loss(logits, y)
                self.log_dict(
                    {'loss': loss, 'x': x})
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = F.nll_loss(logits, y)
                return loss

            def validation_epoch_end(
                self, outputs: List[Any]
            ) -> None:
                self.log('loss_val', torch.stack(outputs).mean())

            def test_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = F.nll_loss(logits, y)
                return loss

            def test_epoch_end(
                self, outputs: List[Any]
            ) -> None:
                self.log('loss_val', torch.stack(outputs).mean())


        def lit_experiment(
                root,
                batch_size=64,    # The batch size of the experiment
                epochs=5,  # Number of epochs to train for
        )
            import xmen
            import os

            from torch.utils.data import DataLoader
            from torchvision.datasets import MNIST
            from torchvision import transforms

            # prepare transforms standard to MNIST
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])

            # data
            mnist_train = MNIST('/tmp/xmen', train=True, download=True, transform=transform)
            mnist_train = DataLoader(mnist_train, batch_size=batch_size)
            mnist_val = MNIST(os.getcwd(), train=False, download=True, transform=transform)
            mnist_val = DataLoader(mnist_val, batch_size=batch_size)

            model = LitMNIST()
            trainer = Trainer(
                default_root_dir=root.directory,
                max_epochs=epochs,
                logger=TensorBoardLogger(
                    root=root,
                    log=['loss@100s', 'loss_val'],
                    sca=['loss@100s', 'loss_val'],   # log tensorboard scalars
                    img='x@100s',                    # log tensorboard images
                    time='@500s',
                    msg='loss@50s'
                )
            )
            trainer.fit(model, mnist_train, mnist_val)
            trainer.test(model, mnist_val)
    """
    def __init__(
        self,
        root,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        # User defined hooks (either) modulo or not modulo
        hooks=[],
        # Default saving for parameters
        log=None, log_fn=None, log_format='',
        time=(),
        img=None, img_fn=None, img_pref='', img_prep=True, img_options={},
        sca=None, sca_fn=None, sca_pref='', sca_prep=True,
        hist=None, hist_fn=None, hist_pref='', hist_prep=True,
        fig=None, fig_fn=None, fig_pref='', fig_prep=True,
        txt=None, txt_fn=None, txt_pref='', txt_prep=True,
        vid=None, vid_fn=None, vid_pref='', vid_prep=True,
        msg=None, msg_keep='latest', msg_leader=None, msg_expand=False, msg_prep=None,
    ):
        """
        Args:
            root (str, Experiment): The root experiment attached to the logging instance or a path to log in
            name (str): The name of the experiment inherited from ``TensorBoardLogger``
            version (str):  The version of the experiment inherited from ``TensorBoardLogger``
            log_graph (bool): Whether to log the computational graph to tensorboard
            hooks (Hook): User defined hooks used to extend the functionality of the Monitor class
            log (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                givin the variables to log @ a particular logging frequency. If no logging frequency is supplied
                then any variable logged in the experiment which matches ``regex`` will be logged each time it is
                passed to the logger. This is useful for logging variables at the end of an epoch for example.
            log_fn (Callable, Iterable[Callable]): Converts the variable into a string for logging.
            log_format (str, Iterable[str]): A format used to format a string as ``f"{string}:{format}"``
            time (str, Iterable[str]): A string of the form ``f"@{steps}"`` to log timing statistics (should be supplied
                as a single string (unlike in the Monitor class where strings at each trigger level are supported).
            img (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard images @ a particular logging frequency.
                If no logging frequency is supplied then any variable logged in the experiment which matches
                ``regex`` will be logged to tensorboard each time it is passed to the logger.
                This is useful for logging variables at the end of an epoch for example.
            img_fn (Callable, Iterable[Callable]): Converts the variable into an image of shape [B, C, H, W] or [C, H, W] for tensorboard.
                See TensorBoardLogger for more details in terms of automatic-processing. If ``img`` is a list then
                ``img_fn`` can also be passed as list with a callable for each entry in ``img`` or can be passed as
                a single callable used for all entries.
            img_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            img_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            img (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard scalars @ a particular logging frequency.
                If no logging frequency is supplied then any variable logged in the experiment which matches
                ``regex`` will be logged to tensorboard each time it is passed to the logger.
                This is useful for logging variables at the end of an epoch for example.
            sca_fn (Callable, Iterable[Callable]): Converts the variable into a scalar for tensorboard.
                See TensorBoardLogger for more details in terms of automatic-processing. If ``sca`` is a list then
                ``sca_fn`` can also be passed as list with a callable for each entry in ``sca`` or can be passed as
                a single callable used for all entries.
            sca_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            sca_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            hist (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard histograms @ a particular logging frequency.
                If no logging frequency is supplied then any variable logged in the experiment which matches
                ``regex`` will be logged to tensorboard each time it is passed to the logger.
                This is useful for logging variables at the end of an epoch for example.
            hist_fn (Callable, Iterable[Callable]): Preprocess variable before logging to tensorboard.
                If ``hist`` is a list then ``hist_fn`` can also be passed as list with a callable for each entry in
                ``hist`` or can be passed as a single callable used for all entries.
            hist_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            hist_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            fig (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard figures @ a particular logging frequency.
                If no logging frequency is supplied then any variable logged in the experiment which matches
                ``regex`` will be logged to tensorboard each time it is passed to the logger.
                This is useful for logging variables at the end of an epoch for example.
            fig_fn (Callable, Iterable[Callable]): Preprocess variable before logging to tensorboard into a plt.figure().
                If ``fig`` is a list then ``fig_fn`` can also be passed as list with a callable for each entry in
                ``fig`` or can be passed as a single callable used for all entries.
            fig_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            fig_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            txt (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to add as tensorboard text @ a particular logging frequency.
                If no logging frequency is supplied then any variable logged in the experiment which matches
                ``regex`` will be logged to tensorboard each time it is passed to the logger.
                This is useful for logging variables at the end of an epoch for example.
            txt_fn (Callable, Iterable[Callable]): Preprocess variable before logging to tensorboard.
                If ``txt`` is a list then ``txt_fn`` can also be passed as list with a callable for each entry in
                ``txt`` or can be passed as a single callable used for all entries.
            txt_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            txt_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            vid_fn (Callable, Iterable[Callable]): Preprocess variable to a tensor of shape [B, T, C, H, W] before logging to tensorboard.
                If ``vid`` is a list then ``vid_fn`` can also be passed as list with a callable for each entry in
                ``vid`` or can be passed as a single callable used for all entries.
            vid_pref (str, Iterable[str]): Prefix all summaries in tensorboard with this string
            vid_prep (bool, Iterable[bool]): If True then dictionary variables will be prepended by the name of the dictionary
            msg (str, Iterable[str]): A modulo string of the form ``"f{regex}"`` or ``"{regex}@{steps}s"`` (or list of)
                giving the variables to log as messages with the experiment.  If no logging frequency is supplied then
                any variable logged in the experiment which matches ``regex`` will be
                logged with the Experiment each time it is passed to the logger. To use this feature an Experiment
                must be passed for the ``root`` parameter.
            msg_keep (str, Iterable[str]): One of ['latest', 'max', 'min'] giving the protocol to use on message collision
            msg_leader (str, Iterable[str]): A regex to a single variable. If supplied then this variable will be treated as the leader
                and all other variables will be logged only if the keep condition is met for the leader
            msg_expand (bool, Iterable[bool]): If True then dictionary variables with K keys will be expanded to give K variables
            msg_prep (bool, Iterable[bool]): If True then each variable in the dictionary will be prepended by the dictionary name.
        """
        super().__init__(root.directory, name, version, log_graph)

        if msg is not None:
            if not isinstance(msg, (list, tuple)):
                msg = [msg]
            for i, m in enumerate(msg):
                m = m.split('@')
                msg[i] = m[0] + '->^root$' + '@' + m[1]

        self._monitor = TorchMonitor(
            directory=self.log_dir,
            # User defined hooks (either) modulo or not modulo
            hooks=hooks,
            # Default saving for parameters
            log=log, log_fn=log_fn, log_format=log_format,
            img=img, img_fn=img_fn, img_pref=img_pref, img_prep=img_prep, img_options=img_options,
            sca=sca, sca_fn=sca_fn, sca_pref=sca_pref, sca_prep=sca_prep,
            hist=hist, hist_fn=hist_fn, hist_pref=hist_pref, hist_prep=hist_prep,
            fig=fig, fig_fn=fig_fn, fig_pref=fig_pref, fig_prep=fig_prep,
            txt=txt, txt_fn=txt_fn, txt_pref=txt_pref, txt_prep=txt_prep,
            vid=vid, vid_fn=vid_fn, vid_pref=vid_pref, vid_prep=vid_prep,
            msg=msg, msg_keep=msg_keep, msg_leader=msg_leader, msg_expand=msg_expand, msg_prep=msg_prep,
            time=time
        )
        self._root = root
        self._steps, self._epochs = None, None

    def _init_timers(self, metrics):
        # Add timers if not already there
        self._steps, self._epochs = metrics['n_steps'], metrics['n_epochs']

        if 'step' not in self._monitor.timers:
            self._monitor.timers['step'] = {"step": StopWatch('step', self._steps)}
            self._monitor.timers['step']['step'].start()
        if 'epoch' not in self._monitor.timers:
            self._monitor.timers['epoch'] = {"step": StopWatch('step', self._epochs)}
            self._monitor.timers['epoch']['step'].start()

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # extract epoch from metrics

        metrics = {**metrics, **{'root': self._root}}

        def matches_to_hook(hook, metrics):
            # if hook.modulo is not None and self._monitor.modulo(t, hook.modulo) and hook.trigger == t:
            matches = {}
            if hook.regex is not None:
                matches = {
                    k: v for k, v in metrics.items() if re.match(hook.regex, k) is not None}
            if matches or isinstance(hook, xmen.monitor.Timer):
                # the timer hook does not take any matches simply
                # using monitor instead
                hook(matches, self._monitor)

        epoch = metrics['epoch']

        if not self._monitor.timers:
            self._init_timers(metrics)

        # increment triggers
        triggers = []
        delta_step = step - self._monitor.triggers['step']
        assert delta_step <= 1, 'Set log every to 1 to use xmen logger'

        if delta_step == 1:
            self._monitor.timers['step']['step'].stop()
            self._monitor.triggers['step'] += 1
            triggers += ['step']

        delta_epoch = epoch - self._monitor.triggers['epoch']
        assert delta_epoch <= 1, 'Set log_every_n_steps to 1 to use xmen logger'
        if delta_epoch == 1:
            self._monitor.timers.pop('step')
            self._monitor.timers['epoch']['step'].stop()
            self._monitor.triggers['epoch'] += 1
            triggers += ['epoch']

        # modulo hooks
        for t in triggers:
            for hook in self._monitor.hooks:
                if not (hook.trigger is not None and self._monitor.modulo(t, hook.modulo) and hook.trigger == t):
                    continue
                matches_to_hook(hook, metrics)

        # none modulo hooks
        for hook in self._monitor.hooks:
            if hook.trigger is None:
                matches_to_hook(hook, metrics)

        if delta_epoch == 1:
            # reset step timer
            self._monitor.timers['step'] = {"step": StopWatch('step', self._steps)}
            self._monitor.timers['step']['step'].start()
            self._monitor.timers['epoch']['step'].start()
        if delta_step == 1:
            self._monitor.timers['step']['step'].start()
            self._monitor.timers['epoch']['step'].start()

