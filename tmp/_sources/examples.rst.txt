*********
Examples
*********

Introduction
############
All examples in this section are defined in ``xmen.examples`` and can be run from the commandline using the ``xmen`` command line interface; just add them using ``xmen --add`` (if they are not already).

Hello World
------------
.. literalinclude:: examples/hello_world.py

A little more detail
----------------------
.. literalinclude:: examples/inheritance.py


The Monitor Class
###################
The ``Monitor`` class is designed to facilitate easy logging of experiements. All the examples in this section can be found in ``xmen.examples.monitor`` and can in python as::
	
	python -m xmen.examples.monitor.logger
	python -m xmen.examples.monitor.messenger.basic
	python -m xmen.examples.monitor.messenger.leader
	python -m xmen.examples.monitor.full


Logging
------------------
.. literalinclude:: examples/monitor/logger.py
.. literalinclude:: examples/monitor/logger.txt


Messaging
--------------------

Messgaging Multiple Experiments
................................

.. literalinclude:: examples/monitor/messenger/basic.py
.. literalinclude:: examples/monitor/messenger/basic.txt

Using a leader
................
.. literalinclude:: examples/monitor/messenger/leader.py
.. literalinclude:: examples/monitor/messenger/leader.txt


A Full example
----------------
.. literalinclude:: examples/monitor/full.py
.. literalinclude:: examples/monitor/full.txt


The TorchMonitor Class
########################
The ``TorchMonitor`` class adds to the functionality of ``Monitor`` also allowing torch modules to be automatically saved and variables to be logged to tensorboard.


python -m xmen.examples.monitor.torch_monitor
	python -m xmen.examples.monitor.checkpoint

Automatic Checkpointing
------------------------
.. literalinclude:: examples/monitor/checkpoint.py
.. literalinclude:: examples/monitor/checkpoint.txt


Tensorboard Logging
--------------------
.. literalinclude:: examples/monitor/torch_monitor.py
.. literalinclude:: examples/monitor/torch_monitor.sh

.. image:: /tb_scalar.png
    :alt: Test

.. image:: /tb_img.png
    :alt: Test

Pytorch experiments with Xmen
##############################
All examples in this section are defined in ``xmen.examples.torch`` and can be run from the commandline using the ``xmen`` command line interface; just add them using ``xmen --add`` (if they are not already).  Pytorch will need to be installed in order to run these examples.

DCGAN using the functional api
------------------------------------
.. literalinclude:: examples/torch/functional.py


DCGAN using the class api
------------------------------------
.. literalinclude:: examples/torch/object.py


Generative modelling with inheritance
--------------------------------------
.. literalinclude:: examples/torch/inheritance.py


Xmen Meets Pytorch Lightning
##############################
.. literalinclude:: examples/torch/lightning.py

