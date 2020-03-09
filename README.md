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
```bash
# Clone
git clone ssh://git@mrgbuild.robots.ox.ac.uk:7999/~robw/xmen.git ~/xmen

# Expose the command line interface
echo alias xmen="~/xmen/python/xmen/main.py" >> ~/zshrc
source ~/.zshrc

# setup a conda environment to run xmen in
conda create --name xmen python=3.6
activate xmen
conda install ruamel.yaml gitpython pandas ipython jupyter

# Install xmen
pip install ~/xmen/python

# Check everything is installed correctly
xmen --help
python -c 'import xmen'
```

## Examples
For examples go to `examples/quick_start.py`!
