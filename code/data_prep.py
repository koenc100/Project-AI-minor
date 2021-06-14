# Data preparation: Strokes project

# import libraries
import numpy as np
import pandas as pd
import math
import statistics
import matplotlib.pyplot as plt
import seaborn as sn
from IPython.display import display
from sklearn.model_selection import train_test_split

# Load data into pandas dataframe
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Create dummies objects
gender = pd.get_dummies(data['gender'])
ever_married = pd.get_dummies(data['ever_married'])
work_type = pd.get_dummies(data['work_type'])
residence_type = pd.get_dummies(data['Residence_type'])
smoking_status = pd.get_dummies(data['smoking_status'])

# Drop old columns
data = data.drop(['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis=1)

# Create new dataframe
data = pd.concat([data, gender, ever_married, work_type, residence_type, smoking_status], axis=1)

# Remove rows with N\A values
data.dropna(axis=0, inplace=True)

# Split the data into target "y" and input "X"
y = data['stroke']
X = data.drop('stroke', axis=1)

#Split the data into 70% training and 30% testing
train_data, test_data, train_labels, test_labels = train_test_split(X, y, train_size=0.7, random_state=1265599650)

print(f'shapes:\nTrain data: {train_data.shape}\nTest data: {test_data.shape}\nTrain labels: {train_labels.shape}\nTest labels: {test_labels.shape}')

# Visualizeren
