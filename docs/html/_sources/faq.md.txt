# FAQ

## Create a frozen package
This will allow the code to be distributed as a single complete repo (including the python interpreter).

```bash
pip install pyinstaller
cd ~/xmen
pyinstaller python/xmen/main.py --hidden-import='pkg_resources.py2_warn' --name xmen

# Note that pkg_resources.py2_warn is not found automatically as a dependency
# To add to your bashrc / zshrc run
echo alias xmen="~/xmen/dist/xmen/xmen" >> ~/.zshrc
```

Xmen can then be distributed by simply copying the `dist/xmen/xmen` folder
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
