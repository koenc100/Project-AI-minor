# Oversampling code

#import imblearn
#from imblearn.over_sampling import RandomOverSampler, SMOTENC
import pandas as pd
from data_processing import prepare_data, split_data
import numpy as np
import scipy as sc
import sklearn as sk
from imblearn.over_sampling import RandomOverSampler

print('kaas')

#print(help(SMOTENC))

data = prepare_data('healthcare-dataset-stroke-data.csv', one_hot = False, binary = False, normalize = False)

train_data, test_data, train_labels, test_labels, = split_data(data, split_size=(0.999, 0.001))

print(train_data)
