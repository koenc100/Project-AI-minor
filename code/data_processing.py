# Data preparation: Strokes project
#  14-06-2021
# Jana Bersee, Koen Ceton, Jeroen Dijkmans & Dominique Weltevreden


# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import itertools

def one_hot_encode(data):
    """
    This function one-hot encodes categorical features in a dataset.

    INPUT:
    data:   A Pandas dataframe

    OUTPUT:
    data:   A Pandas dataframe with the categorical values one-hot encoded
    """

    # Turn CopyWarning off because there is no problem but it keeps showing
    # up
    pd.set_option('mode.chained_assignment', None)
    
    # Replace 1 and 0 for hypertension values with actual names for the
    # classes for one hot encoding
    data['hypertension'].replace(to_replace = (0, 1), value = ('normal',
                                 'hypertension'), inplace = True)
    data['heart_disease'].replace(to_replace = (0, 1), value = ('healthy',
                                   'heart disease'), inplace = True)

    # Loop over every feature in the dataset (=the columns)
    for feature in list(data.columns):

        # Get the feature column from the dataframe
        current_feature = data[feature]

        # Find the categorical features, by seeing if the column has less
        # than 10 unique values. Do not one-hot encode the outcome variable.
        if (len(np.unique(current_feature)) < 10) & (feature != "stroke"):

            # Get dummies for one-hot encoding
            dummies = pd.get_dummies(current_feature)

            # Drop the old, not one-hot encoded column
            data = data.drop(feature, axis = 1)

            # Add the new dummy one-hot encoded columns to the dataset
            data = pd.concat([data, dummies], axis = 1)

    # Rename the unknown smoking status column to something more clear
    data = data.rename(columns={'Unknown':'unknown_smoking_status'})


    return data

def prepare_data(path, one_hot = True, binary = False, normalize = True):

    """
    This function cleans the stroke csv dataset.
    It returns the dataset cleaned as a pandas dataframe.

    Parameters upon function call are the computer location of stroke csv file
    (path) and a boolean for if data should be one-hot-encoded (one_hot, default
    True), a boolean for if features with 2 categories should be changed
    to binary values (e.g. gender 0 or 1, instead of 'male' and 'female')
    (binary, default False) and a boolean for if the numeric data should be
    normalized using z-score (normalized, default True).
    """

    # Load data into pandas dataframe
    data = pd.read_csv(path)

    # Drop ID column
    data = data.drop(['id'], axis=1)

    # Make a mask for groups with stroke and non-stroke
    stroke = data['stroke'] == 1
    not_stroke = data['stroke'] == 0

    # Get the mean BMI for the stroke and non-stroke group
    means_bmi = data.groupby("stroke")["bmi"].mean()

    # Use these masks and the bmi per group to fill the NA data with the mean
    # bmi for that group
    data[stroke] = data[stroke].fillna(means_bmi[1], axis=1)
    data[not_stroke] = data[not_stroke].fillna(means_bmi[0], axis=1)

    # Drop the 'other' sample for gender because there is only one sample and
    # we could not one-hot encode gender otherwise
    data = data[data.gender != 'Other']

    #Create columns with binary values if the binary argument is true
    if binary:

         # Replace columns with two categories with binaries
         data = data.replace({'Male': 1, 'Female': 0, 'Urban': 1, 'Rural': 0,
                             'Yes': 1, 'No': 0})

    # Change categorical columns to one-hot encoded columns
    if one_hot:

        # One hot encode the data with the function
        data = one_hot_encode(data)

    # Clean the column names of uppercase letters and spaces
    data.columns = data.columns.str.lower().str.replace(' ','_')

    # Normalize the numerical data
    if normalize:

        # Create a list with column names of numerical data as we only want
        # to normalize these
        numeric_data = ['age', 'bmi', 'avg_glucose_level']

        # Normalize the numeric values using the zscore
        data[numeric_data] = data[numeric_data].apply(zscore)

    return data

def split_data(data, split_size=(0.7, 0.3)):

    """
    This function splits the pandas dataframe for the stroke csv, given by
    the prepare_data function.

    Input:
    data, a clean pandas dataframe
    split_size = tuple of length 2 or 3, with the values of the different
    ratios that the dataset should be split up into summing up to 1.

    Depending on split size, the following happens:
    1. Tuple with 2 values: (training data, testing data) split, proportions in
    decimal.
    Returns 4 dataframes (train_data, test_data, train_labels, test_labels)
    2. Tuple with 3 values: (training data, testing data, validation data)
    split, proportions in decimal.
    Returns 6 dataframes (train_data, test_data, validation_data, train_labels,
    test_labels, validation_labels)
    The default split_size is (0.7, 0.3) (70% training data, 30% testing data)
    """
    # Check if the split_size is correct
    assert sum(split_size) == 1, 'The dataset should be split completely'

    # Remove the output column from the dataframe and save it as its own variable
    labels = data.pop('stroke')

    # If the size of the split is in train and test data only
    if len(split_size) == 2:

        # Split the data into training and testing data and labels
        # This returns a tuple which we will save and return
        labels_for_data = train_test_split(data,
        labels, train_size=split_size[0], random_state=1, stratify =
        labels)

        # This tuple contains train_data, test_data, train_labels, test_labels
        # respectively
        return labels_for_data

    # If the tuple length for split is 3 split data into test, train and
    # validation
    if len(split_size) == 3:

        # Split the training data from the testing data, with the second number
        # in the split_size tuple. For split size (0.6, 0.2, 0.2), you would have
        # 0.2 testing data here and 0.8 data left over.
        train_data, test_data, train_labels, test_labels = train_test_split(data,
        labels, test_size=split_size[1], random_state=1, stratify = labels)

        # Recalculate the remaining numbers in the split_size datasplit to a scale
        # from 0 to 1, as train_test_split works with splitting based on a ratio.
        # To do this, get the validation size, and divide this by the remaining
        # part of the 'training data'. Continuing with the 0.8 from the data left
        # over, to get 0.2 validation testing data from the total dataset, we
        # would have to split with a ratio of 0.2 / 1 - 0.2 = 0.25.
        split_val_size = split_size[2] /  (1 - split_size[1])

        # Split out the training data into training data and validation data
        # with the calculated ratio
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data, train_labels, test_size=split_val_size, random_state=1,
            stratify = train_labels)

        return train_data, test_data, val_data, train_labels, test_labels, val_labels


# Run only if script is main document
if __name__ == '__main__':

    # Prepare the data
    data = prepare_data('healthcare-dataset-stroke-data.csv')

    #print(data.info())
    #print(data.head())
    # Split the data
    train_data, test_data, val_data, train_labels, test_labels, val_labels = split_data(data,
    split_size=(0.6, 0.2, 0.2))

    # Print shapes of training and testing data
    print(f'\nShapes:\nTotal data: {data.shape}\nTrain data: '
    f'{train_data.shape}\nTest data: {test_data.shape}\nTrain labels: '
    f'{train_labels.shape}\nTest labels: {test_labels.shape}\n')

    # Print info
    print(train_data.info())
