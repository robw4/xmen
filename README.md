# Xmen
Author: Rob Weston
Email: robw@robots.ox.ac.uk
> Xmen is currently in the beta phase and needs testing. Please give it a go and let me know if there are any issues/ feature requests

The E`x`peri`men`t management suite is designed to facilitate fast, reproducible and platform agnostic experimentation from both within python and the shell. Highlights:

- Minimal overhead
- Lightweight
- Built in version control
- Inter experiment communication
- Reproducible experimentation
- Human interpretable
- Remote compatible
- Language agnostic

## Setup
```bash
git clone ssh://git@mrgbuild.robots.ox.ac.uk:7999/~robw/xmen.git ~/xmen
pip install ~/xmen/python
```


## Command line
__Python Interface__
```bash
xmen py --add xmen.examples.object MnistCGan  # add a new python experiment (must be on path)
xmen py --list  # list current experiments
xmen py MnistCGan --help  # display ex
xmen py MnistCGan [--debug] [-u "{lr: 0.01}"] -x "/tmp/my_exp"  # execute an experiment at "/tmp/my_exp"
xmen py MnistCGan [-u "/path/to/params.yml"] -x "/tmp/my_exp"  # execute an experiment using parameters in a params.uml file
xmen py --remove MnistCGan # remove experiment from the xmen interface (non-destructive)
```
__Config__
```bash
xmen config --enable_prompt  # turn prompting on (recommended)
xmen config --disable_prompt  # turn prompting off
xmen config --clean  # remove experiments from config that no longer exist
xmen config -H /path/to/header.txt  # update header prepended to each run script
xmen config --list  # list the current configuration
```

__Initialise an Experiment Set__
```bash
xmen init -n MnistCGan  # from a python experiment in the current dirctory
xmen init -n MnistCGan -r mnist_gann # at folder relative to cwd
xmen init -s /path/to/script.sh -d /path/to/defaults.yml  # from a script and defaults 
```

__Register__
```bash
xmen register -u "{lr: 1e-2 | 1e-3, epochs: 10}" # experiments from parameter combinations
xmen register -u "{lr: 1e-2}" -x 10  # register 10 experiments with the same configiration
xmen register [-u "{lr: 1e-2}" ] -n test # register an experiment named test
```

__Run__
```bash
xmen run "*" bash # run all experiments iteratively matching glob and with default status
xmen run "lr=0.0001" bash   # run a particular experiment
xmen run "*" sbatch  # run each experiment using the slurm job shecudler
xmen run "*" screen -dm bash   # run each experiment in a seperate screen session
```

__Note__
```bash
xmen note "some message" # add note to experiment set
xmen note "TODO: do something" # add note to experiment set
```

__Manipulating__
```bash
xmen reset {PATTERN}   # reset experiments to default status
xmen unlink {PATTERN}  # unlink experiment(s) from a set 
xmen relink -e {PATTERN}  # relink experiments matching pattern
xmen clean  # remove all experiments no longer in the set
xmen rm {FOLDER}  # remove entire experiment folder

# moving a single experiment set
mv experiment-set experiment-set-2   # move experiment folder
xmen relink -r experiment-set-2   # relink the experiment folder

# moving a folder of multiple experiment sets
mv folder folder-2  # move experiment folder
xmen relink -r folder-2 --recursive # recursively relink all experiment folders
```

__List__
```bash
xmen list  # list experiments in cwd
xmen list -l  # display experiments in notebook form
xmen list -d  # display date registered
xmen list -g  # display version information
xmen list -P  # display purpose message
xmen list -M  # display meta information
xmen list -s  # display experiment status
xmen list "*"  # display experiments matching glob string
xmen list -p "lr.*"  # display all experiments with parrameters matching regex search
xmen list -m   # display messages matching default regex search
xmen list -m "e|s"  # display messages matching specified regex search
xmen list "*" -sd -m "e|s"  # options can be used together
xmen list -v  # verbose list
xmen list [...] --to_csv  # list as csv instead of formatted table
xmen list --csv >> my_file.txt   # list as csv and pipe to text file
xmen list --max_width 1000   # maximum width of each collumn
xmen list --max_rows 100    # maximum number of rows
```


## Create a frozen package
This will allow the code to be distributed as a single complete repo (including the python
interpreter).
```bash
pip install pyinstaller
cd ~/xmen
pyinstaller python/xmen/main.py --hidden-import='pkg_resources.py2_warn' --name xmen

# Note that pkg_resources.py2_warn is not found automatically as a dependency
# To add to your bashrc / zshrc run
echo alias xmen="~/xmen/dist/xmen/xmen" >> ~/.zshrc
```
Xmen can then be distributed by simply copying the ``dist/xmen/xmen`` folder
to others without any environment dependency.

## Xmen is too slow!
If you are finding that xmen is running too slow this is most
likely as a result of slow imports within your own project.
To avoid slow imports adopt these good practices:

1. *Use lazy imports where possible*: Instead of importing 
  everything at the start of your experiment module
  add your imports to the experimens `run` method. For experiments which
  require a lot of other dependencies this can significantly
  speed up the command line tools which typically only call
  an experiments `__init__` and `to_root` methods.
  This will have exactly the same overhead
  as having global imports when it comes to running the
  experiment. The import time is instead distributed
  throughout the execution of the program instead of 
  all at start up avoiding unnessercary wait times.
2. *Use minimal environemnts*: Make sure your python
  environement is as slim as possible containing only 
  the packages that are neccessary to run your code.
3. *Freeze*: Freezing xmen in a stand alone distribution
  can help to speed up the time looking for xmens dependencies
  in a bloated enviroment (see avove).
 

## Examples
For examples see `examples/quick_start.py`!
