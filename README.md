# Final Project: Stroke prediction
Final 2-week project of the UvA AI minor, which employs different classification algorithms to classify a dataset with a binary output. Specifically, we worked with the stroke prediction dataset (https://www.kaggle.com/fedesoriano/stroke-prediction-dataset), to classify which patients are likely to get a stroke in the future based on several predictor variables, both continuous and categorical. We employed k-NN, a neural network, and a decision tree (+ forests).



## Structure:
This repository contains the following folders:
* /logs: a folder that holds a folder for each of the group members’ daily log files.
* /docs: a folder holding the pdf of our report
  * /docs/images: a folder within /docs that holds a few images in the report. As we hand in a pdf, we don't actively use this folder.
* /code: a folder holding all code and notebooks used in our project: every model that we use for classification has it's own jupyter notebook, as well as the combined models.
There are also several python files, which contain basic functions we use in several of the notebooks.

## Required libraries/programs:

* python (>=3.6)
* numpy (>=1.13.3)
* scipy (>=0.19.1)
* scikit-learn (>=0.23)
* keras 2 (optional)
* tensorflow (optional)
* imbalanced-learn (0.8.0)



## Description / Example
The model classifies a sample from a dataset as one of two binary classes, based on various categorical (one-hot encoded) and continuous features, by combining the prediction of a decision tree and a neural network. Model parameters are specifically set for imbalanced data, so data where one of two output classes is rare (about 5% of cases). 

### Authors:
* Jana Bersee
* Koen Ceton
* Jeroen Dijkmans
* Dominique Weltevreden
