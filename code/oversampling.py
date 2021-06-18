# Oversampling code

#import imblearn
#from imblearn.over_sampling import RandomOverSampler, SMOTENC
import pandas as pd
from data_processing import prepare_data, split_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from imblearn.over_sampling import RandomOverSampler, SMOTENC


data = prepare_data('healthcare-dataset-stroke-data.csv', one_hot = False, binary = True, normalize = True)

train_data, test_data, train_labels, test_labels, = split_data(data, split_size=(0.999, 0.001))

def smote_loop(data, labels, n_features, start, stop, step):

    """
    Function returns 3 lists:
    list_data: list of split data
    list_labels: list of split labels
    list_ratio: list of the ratios split on

    parameters:
    data: training data set
    labels: corresponding labels
    n_features: numpy array with categorial features
    start: at what value do you want the ratio to start
    stop: at what value do you want the ratio to stop
    step: the step size
    """

    list_data = []
    list_labels = []
    list_ratio = []

    for i in np.arange(start, stop, step):

        # Make smote object: sample stagagy = minority / majority
        smote_nc = SMOTENC(categorical_features = n_features, sampling_strategy = i)

        # create resampled data and labels
        train_data_res, train_labels_res = smote_nc.fit_resample(data, labels)

        # list with sampled data and  labels
        list_data.append(train_data_res)
        list_labels.append(train_labels_res)
        list_ratio.append(i)

        train_data_res['stroke'] = train_labels_res

        sns.histplot( x=train_data_res['age'], hue=train_data_res['stroke'], bins=30, kde=True)
        plt.show()

    return list_data, list_labels, list_ratio

# Define categorial features
n_features = np.array([True, False, True, True, True, True,True, False, False, True])

print(smote_loop(train_data, train_labels, n_features, 0.2, 1.1, 0.2))
