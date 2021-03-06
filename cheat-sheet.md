# Cheat Sheet
## xmen
```bash
# config - global configuration
xgent config --enable_prompt  # turn prompting on (recommended)
xgent config --disable_prompt  # turn prompting off
xgent config --clean  # remove experiments from config that no longer exist
xgent config -H /path/to/header.txt  # update header prepended to each run script
xgent config --list  # list the current configuration

# experiments - experiments interface
xgent experiments -a MODULE NAME # add a experiments experiment to xmen
xgent experiments -l # list available experiments experiments
xgent experiments -r NAME # remove a experiments experiment
xgent experiments HelloWorld # display experiemnt docs
xgent experiments HelloWorld --help # get help with running the experiment
xgent experiments HelloWorld [--debug] [-u "{lr: 0.01}"] -x "/tmp/my_exp"  # execute an experiment at "/tmp/my_exp"
xgent experiments HelloWorld [-u "/path/to/params.yml"] -x "/tmp/my_exp"  # execute an experiment using parameters in a params.uml file

# init - initialise an experiment set
xgent init -n HelloWorld  # from a experiments experiment in the current dirctory
xgent init -n HelloWorld -r mnist_gann # at folder relative to cwd
xgent init -s /path/to/script.sh -d /path/to/defaults.yml  # from a script and defaults

# link - link experiments for running
xgent link -u "{lr: 1e-2 | 1e-3, epochs: 10}" # experiments from parameter combinations
xgent link -u "{lr: 1e-2}" -x 10  # link 10 experiments with the same configiration
xgent link [-u "{lr: 1e-2}" ] -n test # link an experiment named test

# run
xgent run "*" bash # run all experiments iteratively matching glob and with default status
xgent run "lr=0.0001" bash   # run a particular experiment
xgent run "*" sbatch  # run each experiment using the slurm job shecudler
xgent run "*" screen -dm bash   # run each experiment in a seperate screen session

# note
xgent note "some message" # add note to experiment set
xgent note "TODO: do something" # add note to experiment set

# list
xgent list  # list experiments in cwd
xgent list -l  # display experiments in notebook form
xgent list -d  # display date registered
xgent list -g  # display version information
xgent list -P  # display purpose message
xgent list -M  # display meta information
xgent list -s  # display experiment status
xgent list "*"  # display experiments matching glob string
xgent list -p "lr.*"  # display all experiments with parrameters matching regex search
xgent list -m   # display messages matching default regex search
xgent list -m "e|s"  # display messages matching specified regex search
xgent list "*" -sd -m "e|s"  # options can be used together
xgent list -v  # verbose list
xgent list [...] --to_csv  # list as csv instead of formatted table
xgent list --csv >> my_file.txt   # list as csv and pipe to text file
xgent list --max_width 1000   # maximum width of each collumn
xgent list --max_rows 100    # maximum number of rows

# manipulating
xgent reset {PATTERN}   # reset experiments to default status
xgent unlink {PATTERN}  # unlink experiment(s) from a set 
xgent relink -e {PATTERN}  # relink experiments matching pattern
xgent clean  # remove all experiments no longer in the set
xgent rm {FOLDER}  # remove entire experiment folder

# moving a single experiment set
mv experiment-set experiment-set-2   # move experiment folder
xgent relink -r experiment-set-2   # relink the experiment folder

# moving a folder of multiple experiment sets
mv folder folder-2  # move experiment folder
xgent relink -r folder-2 --recursive # recursively relink all experiment folders
```