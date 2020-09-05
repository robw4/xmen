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
## xmen
1. Define experiments in python
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
2. Add to the experiment manager
    ```bash
    >> xmen py --add xmen.examples.hello_world HelloWorld
    ```
__Xmen will take it from here...__
- _Help generation_ `>>> xmen py HelloWorld --help`
  ```bash
  ...
  My very first experiment

  Parameters:
      HelloWorld
       a (str): The first argument (default='Hello')
       b (str): The second argument (default='World')
  ```
- _Command line interface_ `>>> xmen py HelloWorld -x /tmp/hello_world`
  ```bash
  Hello World!
  ```
- _Argument parser_ `>>> xmen py HelloWorld -u "{a: Bye Bye, b: Planet}" -x /tmp/bye_bye_planet`
  ```bash
  Bye Bye Planet!
  ```
- _Built in versioning and records_ `>>> xmen list /tmp/bye_bye_planet`
  ```bash
  _root: /tmp/hello_world_experiments
  _name: a=Bye__b=Planet
  _status: registered
  _created: 2020-09-05-12-22-20
  _messages:
  _version:
    module: /Users/robweston/xmen/python/xmen/examples/hello_world.py
    class: HelloWorld
    git:
      local: /Users/robweston/xmen
      remote: https://github.com/robw4/xmen.git
      commit: 93e5309e0f333419e1cebf49903fdb08df1a5af6
      branch: master
  _meta:
    mac: 0x6c96cfdb71b9
    host: robw-macbook-2.local
    user: robweston
    home: /Users/robweston
  a: Bye
  b: Planet
  ```
- _Rapid experimentation_
    ```bash
    # Initialise Experiment Set
    >>> xmen init -n HelloWorld -r .
    # Register Experiments
    >>> xmen register -u "{a: Hello | Bye, b: World | Planet}"
    # Visulaise
    >>> xmen list -ds -p ".*"
       root               name      status              created      a       b
    0  exps   a=Hello__b=World  registered  2020-09-04-18-36-12  Hello   World
    1  exps  a=Hello__b=Planet  registered  2020-09-04-18-36-11  Hello  Planet
    2  exps     a=Bye__b=World  registered  2020-09-04-18-36-11    Bye   World
    3  exps    a=Bye__b=Planet  registered  2020-09-04-18-36-10    Bye  Planet
    # Run
    >>> xmen run "*" bash
    >>> xmen run "*" screen -dm bash
    >>> xmen run "*" docker ...
    >>> xmen run "*" sbatch
    ```
__... and much, much more!__

## Installation
```bash
>>> git clone https://github.com/robw4/xmen.git ~/xmen
>>> pip install ~/xmen/python
>>> xmen --help
```
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
>>> python -m xmen.examples.hello_world --help
```

## Dependencies
- Core Xmen dependencies:
  - `python>=3.6`
  - `ruamel.yaml`
  - `git-python`
  - `pandas`
- Optional Dependencies
  - ``pytorch``
  - ``tensorboard``
- Documentation:
  - `sphinx`
  - `recommonmark`
  - `nbsphinx`

## Author, Issues, Contributions
- Xmen was created and is managed by Rob Weston (robw@robots.ox.ac.uk) 
- Any issues, file an issue or contact me! 
- If you would like to contribute make a pull requests :)
