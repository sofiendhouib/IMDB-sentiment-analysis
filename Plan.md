# Plan

# Shallow classifier

## Data preparation

* When opening with LibreOfficeCalc, uncheck "space"

### Data cleaning:

This is a step where no information is destroyed from the train or test, and is done in a totally unsupervised way 

* [ ] Spell check --> reduce the number of detected words. example "cheesy", "cheezy"

- [ ] Remove html. Example `<br />` for line break

- [ ] Store clean data

### Data preprocessing

* [x] Code data parser

* [ ] Code data representation:
  
  * [x] tf-idf
  - [ ]
  * [x] N-grams ? Bag of words? Word 2 vec ?how to choose em
  
  - [ ] Do words written in capital size express sthg ? in that case, should we lower case ?
  
  * [ ] Stemming, lemmatization etc
  
  * See off-the-shelf solutions: sklearn feature extractors, and try them all via a pipeline.
  
  * remove titles of the movies from the reviews
  
  * Stop words ? "not good" and
  
  * [ ] Other features: length of review, expressions between parentheses (for sarcarsm ?)
- [x] Check if data is balanced or not  ---> balanced

- [ ] Store preprocessed data

- [ ] Eliminate variables with low variance

- [ ] Show which words correlate better with output
  
  ### Remarks
* Many sentiment words are independent of the movie topic, like "good", "bad", "surprising", "average" etc. They can appear in many documents. Is it good to use idf with them ?
  
  ## Classifier

* [ ] Build and run a simple model:
  
  * [ ] logistic: l1 and l2 penalties
  
  * [ ] SVM
  - [ ] random forest
  
  - [ ] boosting
  
  - [ ] MLP
  
  - [ ] A voting model from the previous
- [ ] Try auto-sklearn
* [ ] Build and run advanced deep model with Tensorflow-Keras.

## Postprocessing

* [ ] Table of results: accuracy, precision, recall, F1, ROC
- [ ] Visualizations : tensorboard for the deep model

# 

# Other

* Store the environment
