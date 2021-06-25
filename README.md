# Final Project: Stroke prediction
Final 2-week project of the UvA AI minor, which employs different classification algorithms to classify a dataset with a binary output. Specifically, we worked with the stroke prediction dataset (https://www.kaggle.com/fedesoriano/stroke-prediction-dataset), to classify which patients are likely to get a stroke in the future based on several predictor variables, both continuous and categorical. We employed k-NN, a neural network, and a decision tree (+ forests).

# Let's get to it:

## Usage:

This project contains multiple notebooks with different pipelines for predicting stroke. These notebooks should be ran in jupyter or other notebook software. Additional custom code is found under /code. Besides, the notebooks require a number of libraries listed below (most of which are included in the ProgLab environment).

### Required libraries/programs:

#### packages included in ProgLab:
* python (>=3.6)
* numpy (>=1.13.3)
* pandas (>=1.2
* scipy (>=0.19.1)
* tqdm (>=4.55.1)
* IPython (>=7.19.0)
* matplotlib (>=3.3.2)
* seaborn (>=0.11.1)
* scikit-learn (sklearn)(>=0.23)
* keras (>=1.0.8)
* tensorflow (>=2.1.0)

#### packages not included in ProgLab:
* imbalanced-learn (0.8.0)

To install, run the following command in a terminal:

`conda install -c conda-forge imbalanced-learn`

* nbimporter
Package for combining functions from other notebook, used in combined model (code/Combined_model.ipynb)
To install, run the following command in notebook (included in data/Combined_model.ipynb):

`pip install nbimporter` 


### Custom code and its usage
## Structure:
This repository contains the following folders:

* /logs: a folder that holds a folder for each of the group members’ daily log files.
* /docs: a folder holding the pdf of our report
  * /docs/images: a folder within /docs that holds a few images in the report. As we hand in a pdf, we don't actively use this folder.
* /code: a folder holding all code in notebooks. including one separate notebook for every model that we use for classification in our project, as well as the combined models.
There are also several python files, which contain basic functions we use in several of the notebooks.
  * /code/visualization: a folder containing all code for data visualization to explore the raw dataset.

#### model notebooks:
The models classify a sample from a dataset as one of two binary classes, based on various categorical (one-hot encoded) and continuous features, by combining the prediction of a decision tree and a neural network. Model parameters are specifically set for imbalanced data, so data where one of two output classes is rare (about 5% of cases).

* /code
  * k_Nearest_Neighbors.ipynb: pipeline for improved k-NN model.
  * deep_neural_network_baseline.ipynb: file containing baseline model before optimization for deep neural network.
  * deep_neural_networks.ipynb: pipeline for improved deep neural network.
  * Decision_tree.ipynb: pipeline for improved decision tree and random forest.
  * Combined_model.ipynb: pipeline for stacking the different models from the other notebooks (using final optimized model hyperparameters).

Run notebooks in jupyter or other notebook software. Additional information on how to use the different functions in these notebooks can be found in the function descriptions and markdown cells contained in the notebooks. Example code for running and optimizing the different models is also provided there.

#### python files containing custom functions used in notebook.
* /code
  * data_processing.py: file containing functions for data preprocessing.
  * helper_functions.py: file containing functions to retrieve metrics for model performance evaluation.
  * oversampling.py: file containing functions to oversample stroke class data.

These functions are imported within the model notebooks.

#### raw data visualization:
* /code/visualization/
  * data_visual.py: This file visualizes the numeric data of the stroke dataset.
  * data_visual_cat.py: This file visualizes the numeric data of the stroke dataset.

Run in terminal using: `python file_name.py`

## Authors:
* Jana Bersee
* Koen Ceton
* Jeroen Dijkmans
* Dominique Weltevreden
