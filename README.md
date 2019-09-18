# Setup
## Requirements
* `python>=3.5` (tested extensively with python 3.6. Should also be compatible with >3.5 but not tested. Python2 is not supported)
* Packages:
    * ruamel
    * git-python
    * pandas

## Installation
The repository is installed from source using:
```bash
git clone https://github.com/robw4/xmen.git
```

It is recommended you add the following to you `~/.bashrc` or equivalent:
```bash
export PYTHONPATH=$PYTHONPATH:~/xmen/python
alias xmen="~/xmen/python/xmen/experiment_manager.py"
```

The latter exposes the experiment manager command line interface. To check that is has been linked correctly try:

```bash
>>> xmen list
The current work directory is not a valid experiment folder. It is either missing one of"script.sh", "experiment.yml" or "defaults.yml" or the experiment.yml file is not valid
```