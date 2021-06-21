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

def one_hot(data, columns):

    """
    Returns pandas dataframe with one hot encoded columns
    data: data used
    columns: list of strings of names of columns to one hot encode
    """

    # create new frame list
    new_frame = [data]

    # loop over every column in column list
    for column_name in columns:

        # Create dummies objects for one-hot encoded columns
        column_dummie = pd.get_dummies(data[column_name])

        # append to list
        new_frame.append(column_dummie)

    # Drop not one-hot endcoded columns
    data = data.drop(columns, axis=1)

    # Create new dataframe with one-hot endcoded columns
    data = pd.concat(new_frame, axis=1)

    # Rename the unknown smoking status column
    data = data.rename(columns={'Unknown':'unknown_smoking_status'})

    return data

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

    print('data', data.shape, 'labels', labels.shape)

    for i in np.arange(start, stop, step):

        # Make smote object: sample stagagy = minority / majority
        smote_nc = SMOTENC(categorical_features = n_features, sampling_strategy = i)

        # create resampled data and labels
        train_data_res_t, train_labels_res = smote_nc.fit_resample(data, labels)

        print('data_2', train_data_res_t.shape)

        # encode one hot
        train_data_res = one_hot(train_data_res_t, ['work_type', 'smoking_status'])

        print('data_2', train_data_res.shape)

        # list with sampled data and  labels
        list_data.append(train_data_res)
        list_labels.append(train_labels_res)
        list_ratio.append(i)

        print(list_data[0].shape, ':data')

    return list_data, list_labels, list_ratio


if __name__ == '__main__':

    data = prepare_data('healthcare-dataset-stroke-data.csv', one_hot = False, binary = True, normalize = True)

    train_data, test_data, train_labels, test_labels, = split_data(data, split_size=(0.999, 0.001))

    # Define categorial features
    n_features = np.array([True, False, True, True, True, True,True, False, False, True])

    smote_test = smote_loop(train_data, train_labels, n_features, 0.2, 1.1, 0.2)
