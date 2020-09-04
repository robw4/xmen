# xmen
## About
The E`x`peri`men`t repo is designed to facitilitate fast, reproducible and platform agnostic experimentation from both within python and the shell. It currently supports oxs and unix based systems both remote and local. 


## Setup
The following are required:
* `python>=3.5` (tested extensively with python 3.6. Should also be compatible with >3.5 but not tested. Python2 is not supported)
* Packages:
    * ruamel
    * git-python
    * pandas
    * sphinx
    * recommonmark
    * nbsphinx

For osx a conda environement is provided which can be installed from `xmen-env-osx64.txt` as:

```bash 
conda create --name xmen --file xmen-env.txt
conda activate xmen
```

The repository is cloned from:
```bash
git clone https://github.com/robw4/xmen.git ~/xmen
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

## Documentation
The documentation for `xmen` is built using sphinx. In order to build the documentation from source once you have installed the correct dependencies you can generate the documentation by running:

```bash
cd ~/xmen/docs
make html
open ~/xmen/docs/_build/html/index.html
```

Extensive examples covering a wide range of the functionality included in `xmen` are found in `/xmen/examples/example.ipynb` which can be run ineractively. Before running make sure that `xmen` is on the PYTHONPATH (see above)


When initially installed the repository also contains a prebuilt doucmentation folder which can be used to launch the docs as:

```bash
open ~/xmen/docs/_build/html/index.html
```


## Contributing
For additional features/ bugs please file an `https://github.com/robw4/xmen` and feel free to make changes yourself before sending a pull request.