
# Xmen
Author: Rob Weston
Email: robw@robots.ox.ac.uk
> Xmen is currently in the beta phase and needs testing. Please give it a go and let me know if there are any issues/ feature requests

The E`x`peri`men`t repo is designed to facilitate fast, reproducible and platform agnostic experimentation from both within python and the shell. Highlights:

- Minimal overhead
- Lightweight
- Built in version control
- Inter experiment communication
- Reproducible experimentation
- Human interpretable
- Remote compatible
- Language agnostic

## Setup
The recommended setup is to use a conda environment. If you do not have conda install you can install as:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

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

## Examples
For examples see `examples/quick_start.py`!
