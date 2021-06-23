# Code to create oversampled data
# 22-06-2021
# Jana Bersee, Koen Ceton, Jeroen Dijkmans, Dominique Weltevreden

import pandas as pd
from data_processing import prepare_data, split_data, one_hot_encode
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from imblearn.over_sampling import RandomOverSampler, SMOTENC

def smote_loop(data, labels, start, stop, step):

    """
    Function returns 3 lists:
    list_data: list of split data
    list_labels: list of split labels
    list_ratio: list of the ratios split on

    Parameters:
    data: training data set
    labels: corresponding labels
    start: at what value do you want the ratio to start
    stop: at what value do you want the ratio to stop
    step: the step size
    """

    # Create empty list to return later
    list_data = []
    list_labels = []
    list_ratio = []

    # # Turn of the SetCopyWarning because it keeps showing up but not in another
    # # file with the exact same code
    # pd.set_option('mode.chained_assignment', None)
    #
    # # Replace 1 and 0 for hypertension values with actual names for the
    # # classes for one hot encoding later
    # data['hypertension'].replace(to_replace = (0, 1), value = ('normal',
    #                              'hypertension'), inplace = True)
    # data['heart_disease'].replace(to_replace = (0, 1), value = ('healthy',
    #                                'heart disease'), inplace = True)

    # Create a list of booleans where true means column contains categorical
    # data
    n_boolean = [len(data[column].unique()) < 10 for column in data.columns]

    # Loop over numbers in the given range
    for i in np.arange(start, stop, step):

        # Make smote object: sample strategy = minority / majority
        smote_nc = SMOTENC(categorical_features = n_boolean,
                           sampling_strategy = i)

        # Create resampled data and labels
        train_data_res_t, train_labels_res = smote_nc.fit_resample(data, labels)

        # Encode one hot
        train_data_res = one_hot_encode(train_data_res_t)

        # Append lists with sampled data and labels
        list_data.append(train_data_res)
        list_labels.append(train_labels_res)
        list_ratio.append(i)

    return list_data, list_labels, list_ratio


if __name__ == '__main__':

    # Load the normalized data without one-hot encoding
    data__ = prepare_data('healthcare-dataset-stroke-data.csv', one_hot = False,
                         binary = False, normalize = True)

    # Split the data into training and testing data
    train_data, test_data, train_labels, test_labels = split_data(data__,
                                                                    split_size=
                                                                    (0.999, 0.001)
                                                                    )

    # Create the oversampled data with its labels and ratios
    list_data, list_labels, list_ratio = smote_loop(train_data, train_labels,
                                                    0.2, 1.1, 0.2)
