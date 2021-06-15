# Data preparation: Strokes project

# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(path, split_size=(0.7, 0.3)):
    """
    Function that cleans and splits stroke dataset and uses the following
    input:
    path = computer location of csv file
    split_size = either tuple of length 2 or 3:
    1. tuple with 2 values: (training data, testing data) in decimal,
    returns 4 dataframes
    2. tuple with 3 values: (training data, testing data, validation data) in
    decimal, returns 6 dataframes
    Default: split_size = (0.7, 0.3)
    """

    # Check if the given split_size is 1 in total
    assert sum(split_size) == 1, 'The dataset should be split completely'

    # Load data into pandas dataframe
    data = pd.read_csv(path)

    # Create dummies objects for one-hot encoded columns
    gender = pd.get_dummies(data['gender'])
    ever_married = pd.get_dummies(data['ever_married'])
    work_type = pd.get_dummies(data['work_type'])
    residence_type = pd.get_dummies(data['Residence_type'])
    smoking_status = pd.get_dummies(data['smoking_status'])

    # Drop not one-hot endcoded columns
    data = data.drop(['gender', 'ever_married', 'work_type', 'Residence_type',
                      'smoking_status', 'id'], axis=1)

    # Create new dataframe with one-hot endcoded columns
    data = pd.concat([data, gender, ever_married, work_type, residence_type,
                      smoking_status], axis=1)

    # Rename column names
    data = data.rename(columns={'Yes':'ever_married', 'No':'never_married',
                                'Unknown':'unknown_smoking_status',
                                'Other':'other_gender'})

    # Clean column names by replacing spaces capital letters
    data.columns = data.columns.str.lower().str.replace(' ','_')

    # Remove rows with N/A values
    data.dropna(axis=0, inplace=True)

    # Split the data into target "y" and input "X"
    y = data['stroke']
    X = data.drop('stroke', axis=1)

    # If the size of the split is in train and test data only
    if len(split_size) == 2:

        # Split the data into training and testing
        train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, train_size=split_size[0], random_state=1265599650)

        return train_data, test_data, train_labels, test_labels

    # If the split is into test, train and validation data
    if len(split_size) == 3:

        # Split out the test data
        train_data, test_data, train_labels, test_labels  = train_test_split(
        X, y, test_size=split_size[1], random_state=1)

        # Calculate portion for validation data split
        split_val_size = split_size[2] /  (1 - split_size[1])

        # Split out the validation data
        train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=split_val_size, random_state=2)

        return train_data, test_data, val_data, train_labels, test_labels, val_labels

#TODO remove
# train_data, test_data, train_labels, test_labels = prepare_data('healthcare-dataset-stroke-data.csv', split_size=(0.7, 0.3))
# train_data, test_data, val_data, train_labels, test_labels, val_labels = prepare_data('healthcare-dataset-stroke-data.csv', split_size=(0.5, 0.25, 0.25))
#
# # Print shapes of training and testing data
# print(f'shapes:\nTrain data: {train_data.shape}\nTest data: {test_data.shape}')
# print(f'Train labels: {train_labels.shape}\nTest labels: {test_labels.shape}')
#
# # Print info
# print(train_data.info())
