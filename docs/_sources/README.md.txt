# Xmen
```
||||||||||||||||||||||||| WELCOME TO ||||||||||||||||||||||||||
||                                                           ||
||    \\\  ///  |||\\        //|||  |||||||||  |||\\   |||   ||
||     \\\///   |||\\\      ///|||  |||        |||\\\  |||   ||
||      ||||    ||| \\\    /// |||  ||||||     ||| \\\ |||   ||
||     ///\\\   |||  \\\  ///  |||  |||        |||  \\\|||   ||
||    ///  \\\  |||   \\\///   |||  |||||||||  |||   \\|||   ||
||                                                           ||
|||||||||||| FAST - REPRODUCIBLE - EXPERIMENTATION ||||||||||||
```
## What?
Define An Experiment
  ```python
  # xmen/examples/hello_world.py
  from xmen import Experiment
  class HelloWorld(Experiment):
      """Experiments are defined classes"""
      # Parameters
      a: str = 'Hello'  # @p The first argument
      b: str = 'World' # @p The second argument

      def run(self):
          print(f'{a} {b}!')
  ```

Regsiter with the experiment manager
  ```bash
  >> xmen py --add xmen.examples.hello_world HelloWorld
  ```

__Xmen will take it from here...__

- Help Generation
  ```bash
  >> xmen py --help
  ...
  My very first experiment

  Parameters:
      HelloWorld
       a (str): The first argument (default='Hello')
       b (str): The second argument (default='World')
  ```
- Automatic run script generation
  ```bash
  >> xmen py -x /tmp/hello_world   # run an experiment
  Hello World!
  ```
- Built in argument parser
  ```bash
  >> xmen py -u "{a: Bye Bye, b: Planet}" -x /tmp/bye_bye_planet   # set parameters
  Bye Bye Planet!
  ```
- Automatic versioning and logging
  ```bash
  >> cat /tmp/bye_bye_planet/params.yml
  _created: 09-04-20-18:27:01  #  The date the experiment was created (default=now_time)
  _messages: {} #  Messages left by the experiment (default={})
  _meta: #  The global configuration for the experiment manager (default=None)
    mac: '0x6c96cfdb71b9'
    host: robw-macbook-2.local
    user: robweston
    home: /Users/robweston
  _name: bye_bye_planet #  The name of the experiment (under root) (default=None)
  _purpose: '' #  A description of the experiment purpose (default=None)
  _root: /tmp #  The root directory of the experiment (default=None)
  _status: finished #  One of ['default' | 'created' | 'running' | 'error' | 'finished'] (default='default')
  _version: #  Experiment version information. See `get_version` (default=None)
    module: /Users/robweston/xmen/python/xmen/examples/hello_world.py
    class: HelloWorld
    git:
      local: /Users/robweston/xmen
      remote: https://github.com/robw4/xmen.git
      commit: 2178773b46a982f67140fd5cef42ae56014b6716
      branch: master
  a: Bye Bye #  The first argument (default='Hello')
  b: Planet #  The second argument (default='World')
  ```
- Rapid config generation and experimentation
  ```bash
  >> mkdir /tmp/exps && cd /tmp/exps
  >> xmen init -n HelloWorld -r .
  >> xmen register -u "{a: Hello | Bye, b: World | Planet}"
  >> xmen list -ds -p ".*"
     root               name      status              created      a       b
  0  exps   a=Hello__b=World  registered  2020-09-04-18-36-12  Hello   World
  1  exps  a=Hello__b=Planet  registered  2020-09-04-18-36-11  Hello  Planet
  2  exps     a=Bye__b=World  registered  2020-09-04-18-36-11    Bye   World
  3  exps    a=Bye__b=Planet  registered  2020-09-04-18-36-10    Bye  Planet
  >> xmen run "*" bash
  Bye Planet!
  Bye World!
  Hello Planet!
  Hello World!
  >> xmen list -ds -p ".*"
     root               name    status              created      a       b
  0  exps   a=Hello__b=World  finished  2020-09-04-18-44-45  Hello   World
  1  exps  a=Hello__b=Planet  finished  2020-09-04-18-44-44  Hello  Planet
  2  exps     a=Bye__b=World  finished  2020-09-04-18-44-43    Bye   World
  3  exps    a=Bye__b=Planet  finished  2020-09-04-18-44-43    Bye  Planet
  ```
- slurm job Scheduler compatibility
```bash
xmen run "*" sbatch
```

__and much, much much more!__

## Installation
To install xmen run:
```bash
git clone https://github.com/robw4/xmen.git ~/xmen
pip install ~/xmen/python
```
To check that everything has run currectly run:

```bash
xmen --help
```
which should display:

```
usage: xmen [-h]
            {py,config,init,register,run,note,reset,unlink,clean,rm,relink,list}
            ...

||||||||||||||||||||||||| WELCOME TO ||||||||||||||||||||||||||
||                                                           ||
||    \\\  ///  |||\\        //|||  |||||||||  |||\\   |||   ||
||     \\\///   |||\\\      ///|||  |||        |||\\\  |||   ||
||      ||||    ||| \\\    /// |||  ||||||     ||| \\\ |||   ||
||     ///\\\   |||  \\\  ///  |||  |||        |||  \\\|||   ||
||    ///  \\\  |||   \\\///   |||  |||||||||  |||   \\|||   ||
||                                                           ||
|||||||||||| FAST - REPRODUCIBLE - EXPERIMENTATION ||||||||||||

positional arguments:
  {py,config,init,register,run,note,reset,unlink,clean,rm,relink,list}
    py                  Python experiment interface
    config              View / edit the global configuration
    init                Initialise an experiment set
    register            Register a set of experiments
    run                 Run experiments matching glob in experiment set that have not yetbeen run.
    note                add notes to an experiment
    reset               Reset an experiment to registered status
    unlink              Unlink experiments from experiment set
    clean               Remove unlinked experiments (DESTRUCTIVE)
    rm                  Remove an experiment set (DESTRUCTIVE)
    relink              Relink experiments to global configuration or to a set root
    list                list experiments to screen

optional arguments:
  -h, --help            show this help message and exit
```

## Tutorials, Examples, Documentation
Full documentation and examples can be found [here](https://robw4.github.io/xmen/).

Several examples are included as part of the repo and can be run as,

```bash
python -m xmen.examples.hello_world --help
```

## Dependencies
- Core Xmen dependencies:
  - `python>=3.6`
  - ruamel.yaml
  - git-python
  - pandas
- Documentation:
  - sphinx
  - recommonmark
  - nbsphinx

## Author, Issues, Contributions
Xmen was created and is managed by Rob Weston (robw@robots.ox.ac.uk)

Author: Rob Weston
Email: robw@robots.ox.ac.uk

Any issues, file an issue or contact me! If you would like to contribute make a pull requests :)
