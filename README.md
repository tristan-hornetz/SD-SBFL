# StatisticalDebugger

_Work in Progress..._

Statistical Debugging framework for [BugsInPy](https://github.com/soarsmu/BugsInPy).  

## Setup

This framework will only run on Linux. Windows is not supported.

Make sure that Git and GNU Make are installed on your system.
The debugger also requires [Pyenv](https://github.com/pyenv/pyenv) to function correctly. 
Installation instructions can be found [here](https://github.com/pyenv/pyenv-installer).

Once Pyenv is installed, you can run ```make```. This will clone the BugsInPy repository 
and set up a Pyenv environment for running the debugger.

_Note: Any actions performed by ```make``` can be reverted with ```make clean```._

## Usage

Before running any of the scripts, activate the Pyenv environment with ```$(cat activate)```.

To start a test-run for a specific bug, run
```shell
python run_test.py -p <project_name> -i <bug_id> [-o <output_file>] 
```
The selected project is then cloned to a temporary folder and a pyenv
environment is set up to replicate the environment in which the bug occurs.
The test-run itself may take anywhere from a few minutes to several hours, depending on the project you choose.
The recorded events are pickled and then dumped to ```/TestWrapper/results.pickle.gz``` (unless another destination was specified with ```-o```). 

Some projects that require a special setup for testing cannot (yet) be directly 
tested like this without manual intervention.

If you want to evaluate the results test-run, you can use

```shell
python evaluate.py [-r <output_file>] 
```

## Acknowledgements

This repository contains code originally from [The Debugging Book](https://github.com/uds-se/debuggingbook) (see the 
_/debuggingbook_ folder).
It was slightly modified so that no third-party packages are required to run it.



