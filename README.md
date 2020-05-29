
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
> The recommended setup is to use a conda environment. If you do not have conda install you can install as:
> 
>```bash
> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
> bash Miniconda3-latest-Linux-x86_64.sh
> ```


```bash
# Clone
git clone ssh://git@mrgbuild.robots.ox.ac.uk:7999/~robw/xmen.git ~/xmen

# Expose the command line interface
echo alias xmen="~/xmen/python/xmen/main.py" >> ~/.zshrc
source ~/.zshrc

# setup a conda environment to run xmen in
conda create --name xmen python=3.6
activate xmen
conda install ruamel.yaml gitpython pandas jupyterlab

# Install xmen
pip install ~/xmen/python

# Check everything is installed correctly
xmen --help
python -c 'import xmen'
```

> Global installation should also work but has not been fully tested

## Create a frozen package
This will allow the code to be distributed as a single complete repo (including the python
interpreter).
```bash
conda activate xmen  # Activate the xmen environment to give *only* the packages that xmen depends on
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
