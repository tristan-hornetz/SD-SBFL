# SD-SBFL Hybrid

This repository contains the implementation of a hybrid model of 
Spectrum Based Fault Localization and Statistical Debugging, as proposed in my Bachelor's Thesis.
It was specifically designed for conducting an empirical study
on the bugs listed in [BugsInPy](https://github.com/soarsmu/BugsInPy).

# Setup

This framework will only run on Linux. Windows is not supported.

Make sure that Git, GNU Make, and all necessary software for building Python from source are installed on your system.
The debugger also requires [Pyenv](https://github.com/pyenv/pyenv) to function correctly. 
Installation instructions can be found [here](https://github.com/pyenv/pyenv-installer).

Once Pyenv is installed, you can run ```make```. This will clone the BugsInPy repository 
and set up a Pyenv environment for running the debugger.

_Note: Any actions performed by ```make``` can be reverted with ```make clean```._

# Usage
## Recording Events

Before running any of the scripts, activate the Pyenv environment with ```$(cat activate)```.

To start a test-run for a specific bug, run
```shell
python run_single_test.py -p <project_name> -i <bug_id> [-o <output_file>] 
```
The selected project is then cloned to a temporary folder and a pyenv
environment is set up to replicate the environment in which the bug occurs.
The test-run itself may take anywhere from a few minutes to several hours, depending on the project you choose.
The recorded events are pickled and then dumped to ```/TestWrapper/results.pickle.gz``` (unless another destination was specified with ```-o```). 

Alternatively, you can use
```shell
python test_complete_application.py -p <project_name>
```
to test all bugs for a specific application at once.
In this case, the results are dumped to ```/results/<project_name>/```.

## Evaluation
### Translation step

Before you can evaluate the recorded events,
they require a translation step. This is separate from the Recording
process due to performance reasons.

```shell
python translate.py -rd <results_directory>
```
The output is written to ```/results_translated/```.

### Final Evaluation

To evaluate the recorded events with a basic configuration, use
```shell
python evaluate.py -r ./results_translated
```
For an evaluation involving more complex configurations, you may append ```-a```.
This will start an evaluation with every configuration presented in my thesis.
Output files are written to ```/results_evaluation/```.


If you wish to view the results of a previous evaluation, use
```shell
python view_results.py -r <result_file>
```


# Acknowledgements

This repository contains code originally from [The Debugging Book](https://github.com/uds-se/debuggingbook) (see the 
_/debuggingbook_ folder).
It was slightly modified so that no third-party packages are required to run it.



