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


The Monitor Helper
###################
The ``MonitorHelper`` class is designed to facilitate easy logging and checkpointing of experiements. All the examples in this section can be found in ``xmen.examples.monitor`` and can in python as::
	
	python -m xmen.examples.monitor.logger
	python -m xmen.examples.monitor.checkpoint
	python -m xmen.examples.monitor.messenger.basic
	python -m xmen.examples.monitor.messenger.leader
	python -m xmen.examples.monitor.tb_monitor


Automatic Logging
------------------
.. literalinclude:: examples/monitor/logger.py
.. literalinclude:: examples/monitor/logger.txt

Automatic Checkpointing
------------------------
.. literalinclude:: examples/monitor/checkpoint.py
.. literalinclude:: examples/monitor/checkpoint.txt


Automatic Messaging
--------------------

Messgaging Multiple Experiments
................................

.. literalinclude:: examples/monitor/messenger/basic.py
.. literalinclude:: examples/monitor/messenger/basic.txt

With a leader
..............
.. literalinclude:: examples/monitor/messenger/leader.py
.. literalinclude:: examples/monitor/messenger/leader.txt


Tensorboard Logging
--------------------
.. literalinclude:: examples/monitor/tb_monitor.py
.. literalinclude:: examples/monitor/tb_monitor.sh

.. image:: /tb_scalar.png
    :alt: Test

.. image:: /tb_img.png
    :alt: Test

Xmen meets Pytorch
###################
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
