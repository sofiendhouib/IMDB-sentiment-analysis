# Clarifications about the code

## Required librarie

* All of the requirements can be installed by creating a new conda environement via loading `environment.yaml`

## Files and folders

* `main.ipynb`: The main file to run the experiments.
* `main_experimental.py`: An experimental version of the main file, divided into several sections, but that can also be run as a python file.
* `data.py`: functions to load and clean the data
* `tuning.py`: functions used for hyperparameter tuning and model selection.
* `postprocessing.py` : functions used for assessing the performance of a model.
* `utils.py`: other useful functions
* `./input/`: Directory contaning both the raw and cleaned data. The latter can be constructed by running the main file.
* `./cv_results/`: Directory for saving the results of the grid search.
