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
1. Define experiments in experiments
    ```experiments
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
2. Configure and run from the command line
    ```bash
    # add an experiment
    >>> xmen experiments --add xmen.examples.hello_world HelloWorld
    # get documentation
    >>> xmen experiments HelloWorld  
      A class experiment

      Parameters:
          a: str=Hello ~ the first argument
          b: str=World ~ the second argument
    # initialise experient folder
    >>> xmen init -n HelloWorld -r .
    # register experiments
    >>> xmen register -u "{a: Hello | Bye, b: World | Planet}"
    # visulaise
    >>> xgent list -ds -p ".*"
       root               name      status              created      a       b
    0  exps   a=Hello__b=World  registered  2020-09-04-18-36-12  Hello   World
    1  exps  a=Hello__b=Planet  registered  2020-09-04-18-36-11  Hello  Planet
    2  exps     a=Bye__b=World  registered  2020-09-04-18-36-11    Bye   World
    3  exps    a=Bye__b=Planet  registered  2020-09-04-18-36-10    Bye  Planet
    # run
    >>> xmen run "*" bash
    >>> xmen run "*" screen -dm bash
    >>> xmen run "*" docker ...
    >>> xmen run "*" sbatch
    ```

## Installation
To install xmen use pip
```bash
pip install git+https://github.com/robw4/xmen.git
```
Alternatively, clone the repo and then run pip if you also want access to the example scripts.
```bash
>>> git clone https://github.com/robw4/xmen.git ~/xmen
>>> pip install ~/xmen/experiments
>>> xmen
```

```
usage: xmen [-h]
            {experiments,config,init,register,run,note,reset,unlink,clean,rm,relink,list}
            ...

||||||||||||||||||||||||||| WELCOME TO |||||||||||||||||||||||||||
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
||    \\\  ///  |||\\        //|||  |||||||||  |||\\   |||      ||
||     \\\///   |||\\\      ///|||  |||        |||\\\  |||      ||
||      ||||    ||| \\\    /// |||  ||||||     ||| \\\ |||      ||
||     ///\\\   |||  \\\  ///  |||  |||        |||  \\\|||      ||
||    ///  \\\  |||   \\\///   |||  |||||||||  |||   \\|||      ||
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
  {experiments,config,init,register,run,note,reset,unlink,clean,rm,relink,list}
    experiments              Python interface
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
  -h, --help            show this help message and exi
```

## Tutorials, Examples, Documentation
- Full documentation (including the experiments API) can be found [here](https://robw4.github.io/xmen/)
- Several example experiment definitions can be found in ``xmen.examples``:
    ```
    xmen.examples
    ├── hello_world.py
    ├── inheritance.py
    ├── monitor
    │   ├── __init__.py
    │   ├── checkpoint.py
    │   ├── logger.py
    │   ├── messenger
    │   │   ├── __init__.py
    │   │   ├── basic.py
    │   │   ├── expand.py
    │   │   ├── leader.py
    │   │   └── prepend.py
    │   └── tb_monitor.py
    └── torch
        ├── functional.py
        ├── inheritance.py
        ├── models.py
        └── object.py
  ```
  The following experiments are defined
  ```experiments
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
  - ``xmen/examples/class-api.ipynb`` contains further details about the experiments class api
- For command line help run
  ```
  >> xmen --help
  ```

## Dependencies
- Core Xmen dependencies:
  - `experiments>=3.6`
  - `ruamel.yaml`
  - `git-experiments`
  - `pandas`
- Monitor Dependencies:
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
