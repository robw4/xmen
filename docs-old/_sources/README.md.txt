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
    from xmen import Experiment, Root

    # as classes... 
    class HelloWorld(Experiment):
        """A class experiment"""
        a: str = 'Hello' # @p The first argument
        b: str = 'World' # @p The second argument

        def run(self):
            print(f'{a} {b}!')
      
    # as functions...
    def hello_world(
      root: Root, 
      a: str = 'Hello',  # the first argument
      b: str = 'World'   # the second argument
      ):
      """A functional experiment"""
      print(f'{a}  {b}')
    ```
2. Run from the command line
    ```bash
    # add an experiment
    >>> xmen --add xmen.examples.hello_world HelloWorld
    # get documentation
    >>> xmen HelloWorld  
      A class experiment

      Parameters:
          a: str=Hello ~ the first argument
          b: str=World ~ the second argument
    # execute
    >>> xmen HelloWorld -x /tmp/hello_world  
    Hello World!
    # change the parameters
    >>> men HelloWorld -u "{a: Bye, b: Planet}" -x /tmp/bye_bye_planet
    Bye Bye Planet!
    # built in records
    >>> xmen list /tmp/bye_bye_planet   
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
3. Mass experimentation with ``xgent``
    ```bash
    # Initialise Experiment Set
    >>> xgent init -n HelloWorld -r .
    # Register Experiments
    >>> xgent register -u "{a: Hello | Bye, b: World | Planet}"
    # Visulaise
    >>> xgent list -ds -p ".*"
       root               name      status              created      a       b
    0  exps   a=Hello__b=World  registered  2020-09-04-18-36-12  Hello   World
    1  exps  a=Hello__b=Planet  registered  2020-09-04-18-36-11  Hello  Planet
    2  exps     a=Bye__b=World  registered  2020-09-04-18-36-11    Bye   World
    3  exps    a=Bye__b=Planet  registered  2020-09-04-18-36-10    Bye  Planet
    # Run
    >>> xgent run "*" bash
    >>> xgent run "*" screen -dm bash
    >>> xgent run "*" docker ...
    >>> xgent run "*" sbatch
    ```

## Installation
To install xmen use pip
```bash
pip install git+https://github.com/robw4/xmen.git
```
Alternatively, clone the repo and then run pip if you also want access to the example scripts.
```bash
>>> git clone https://github.com/robw4/xmen.git ~/xmen
>>> pip install ~/xmen/python
>>> xmen
```
```
usage: xman [-h] [--list] [--add MODULE NAME MODULE NAME] [--remove REMOVE] [name [name ...]] ...

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
  name                  The name of the experiment to run
  flags                 Python flags (pass --help for more info)

optional arguments:
  -h, --help            show this help message and exit
  --list, -l            List available python experiments
  --add MODULE NAME MODULE NAME
                        Add a python Experiment class or run script (it must already be on PYTHONPATH)
  --remove REMOVE, -r REMOVE
                        Remove a python experiment (passed by Name)
```

## Tutorials, Examples, Documentation
- Full documentation (including the python API) can be found [here](https://robw4.github.io/xmen/)
- Several example experiment definitions can be found in ``xmen.examples``:
    ```
    ├── examples
    │   ├── __init__.py
    │   ├── hello_world.py      # basic xmen experiments defined using the functional and class apis
    │   ├── inheritance.py      # basic inheritance example
    │   └── torch               # several examples of more complicated experiments in pytorch
    │       ├── __init__.py
    │       ├── functional.py   # a functional dcgan implementation
    │       ├── inheritance.py  # a object-orientated experiments with inheritance
    │       └── object.py       # a object-orientated object-orientated
    ```
  The following experiments are defined
  ```python
  from xmen.examples import hello_world, HelloWorld  # simple examples
  from xmen.examples import AnotherExperiment, MultiParentsExperiment  # with inheritance
  from xmen.examples import dcgan # functional dcgan in pytorch
  from xmen.examples import Dcgan # object-orientated dcgan in pytorch
  from xmen.examples import InheritedMnistGAN, InheritedMnistVae  # inheritance examples
  ```
  All these experimens can be run from using the xmen commandline tool, eg.:
  ```bash
  >> xmen --add xmen.examples Dcgan
  >> xmen --list
  ...
  Dcgan: /home/robw/.xmen/experiments/xmen.examples.Dcgan
  ...
  >> xmen Dcgan
  ```
- Jupyter notebook tutorials can be found at
  - ``xmen/examples/tutorial.ipynb`` contains a quickstart guide to using xmen
  - ``xmen/examples/class-api.ipynb`` contains further details about the python class api
- For command line help run
  ```
  >> xmen --help
  usage: xmen [-h] [--list] [--add MODULE NAME MODULE NAME] [--remove REMOVE] [name [name ...]] ...

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
    name                  The name of the experiment to run
    flags                 Python flags (pass --help for more info)

  optional arguments:
    -h, --help            show this help message and exit
    --list, -l            List available python experiments
    --add MODULE NAME MODULE NAME
                          Add a python Experiment class or run script (it must already be on PYTHONPATH)
    --remove REMOVE, -r REMOVE
                          Remove a python experiment (passed by Name)
  ```
  ```
  >> xgent --help
  usage: xgent [-h] {config,init,register,run,note,reset,unlink,clean,rm,relink,list} ...

  ||||||||||||||||||||||||||| WELCOME TO ||||||||||||||||||||||||||||
  ||                                                              ||
  ||          &@&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&@&%          ||
  ||         *@&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&         ||
  ||          &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&          ||
  ||           &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&          ||
  ||           &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#          ||
  ||           &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&.          ||
  ||           &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.          ||
  ||           &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*          ||
  ||           @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@          ||
  ||   #&@@@@@&%&&&&&&&@@@@@@@@@@@@@@@@@@@@@@@@@&&&&&&&&@@@@@@&#  ||
  ||  /#%%%%%%%%%&&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&&%%&%%%%%%#  ||
  ||   &%&&&&&&&&&&&@@@@@@@@@@@@@@@@@@@@@@@@@@@&&@&&&&&&&&&&&&&   ||
  ||     (@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&.    ||
  ||      ...,*/#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&##(*,...      ||
  ||                                                              ||
  ||  \\\  ///  |||||||||||  |||||||||  |||\\   |||  ||||||||||   ||
  ||   \\\///   |||          |||        |||\\\  |||      |||      ||
  ||    ||||    |||   |||||  ||||||     ||| \\\ |||      |||      ||
  ||   ///\\\   |||     |||  |||        |||  \\\|||      |||      ||
  ||  ///  \\\  |||||||||||  |||||||||  |||   \\|||      |||      ||
  ||                                                              ||
  ||                      %@@,     (@@/                           ||
  ||                     @@@@@@@@@@@@@@@@@@@@@                    ||
  ||        @@        @@@@@@@@@@@@@@@@@@@@@@@@@@/        @#       ||
  ||       @@#     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#     @@       ||
  ||        @@@@@@@@@@@@@@@@@@@@@@@.@@@@@@@@@@@@@@@@@@@@@@.       ||
  ||           ,@@@@@@@@@@@@@@@%       @@@@@@@@@@@@@@@@           ||
  ||                                                              ||
  |||||||||||||| FAST - REPRODUCIBLE - EXPERIMENTATION |||||||||||||

  positional arguments:
    {config,init,register,run,note,reset,unlink,clean,rm,relink,list}
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

## Dependencies
- Core Xmen dependencies:
  - `python>=3.6`
  - `ruamel.yaml`
  - `git-python`
  - `pandas`
- Monior Dependencies:
  - ``pytorch``
  - ``tensorboard``
- Documentation:
  - `sphinx`
  - `recommonmark`
  - `nbsphinx`
  - `sphinx_glpi_theme`

## Author, Issues, Contributions
- Any issues, file an issue or contact me!
- Contributions welcome! Just make a pull request :)
