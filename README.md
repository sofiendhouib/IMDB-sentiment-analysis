# Clarifications about the code

## Dataset

The dataset can be downloaded from this [Kaggle link]([IMDB Dataset of 50K Movie Reviews | Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)), which is a structured version of the data available at the [Stanford University link](https://ai.stanford.edu/~amaas/data/sentiment/).

## Required libraries

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

## Web resources

* [Elegant text pre-processing with NLTK in sklearn pipeline | Medium](https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8)
* [Getting started with Text Preprocessing | Kaggle](https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing/input)
