****
API
****

functional
############

.. autoclass:: xmen.experiment.Root()
   :members: directory, message, messages

.. automodule:: xmen.functional
  :members:


experiment
###########

.. autoclass:: xmen.experiment.Experiment
   :members:

   .. automethod:: __init__


monitor
########

Monitor
--------
.. autoclass:: xmen.monitor.Monitor
  :members:

  .. automethod:: __init__


TorchMonitor
-------------
.. autoclass:: xmen.monitor.TorchMonitor
  :members:

  .. automethod:: __init__


Hooks
------
.. autoclass:: xmen.monitor.Hook
  :members:


.. autoclass:: xmen.monitor.XmenMessenger
  :members:

  .. automethod:: __init__


.. autoclass:: xmen.monitor.Timer
  :members:

  .. automethod:: __init__


.. autoclass:: xmen.monitor.Logger
  :members:

  .. automethod:: __init__


.. autoclass:: xmen.monitor.TensorboardLogger
  :members:

  .. automethod:: __init__


.. autoclass:: xmen.monitor.Checkpointer
  :members:

  .. automethod:: __init__


Helpers
--------
.. autofunction:: xmen.monitor.load


lightning
############

.. autoclass:: xmen.lightning.TensorBoardLogger
  :members:

  .. automethod:: __init__

.. autoclass:: xmen.lightning.Trainer
  :members:


utils
######
.. automodule:: xmen.utils
	:members:


manager
#########
.. autoclass:: xmen.manager.ExperimentManager
   :members:

    .. automethod:: __init__

config
#######
.. autoclass:: xmen.config.GlobalExperimentManager
  :members:

  .. automethod:: __init__