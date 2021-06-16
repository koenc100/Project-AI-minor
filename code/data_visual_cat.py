# Visualize the categorical data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import prepare_data

# load the data without one hot encoding
data_cat = prepare_data('healthcare-dataset-stroke-data.csv', one_hot = False, binary = False)

# make sure all values are strings, so plotting with histplot works
data_cat['hypertension'].replace(to_replace = (0, 1), value = ('no', 'yes'), inplace = True)
data_cat['heart_disease'].replace(to_replace = (0, 1), value = ('no', 'yes'), inplace = True)

# for stroke a 1 means stroke and 0 means no stroke
data_cat['stroke'].replace(to_replace = (0, 1), value = ('no stroke', 'stroke'), inplace = True)

# create a figure to display multiple figures at once
fig, axes = plt.subplots(2, 3, figsize=(20,10), sharey=True)

# add a title for the figures combined
fig.suptitle('Figures to show categorical variables in dataset')

# create hisplots for al categorical data
sns.histplot(ax=axes[0, 0], data = data_cat, x = 'gender', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[0, 1], data = data_cat, x = 'hypertension', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[0, 2], data = data_cat, x = 'heart_disease', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[1, 0], data = data_cat, x = 'ever_married', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[1, 1], data = data_cat, x = 'work_type', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[1, 2], data = data_cat, x = 'residence_type', hue = 'stroke', multiple="stack")

# show the plot
plt.show()

print(data_cat.columns)

def proportion(data, column_name):

    variables = data[column_name].unique()
    data_stroke = data[data['stroke'] == 'stroke']

    for item in variables:

        data_item = data[data.column_name == item]
        print(data_item)


    # female_stroke = data_stroke[data_stroke['gender'] == 'Female']
    # male_stroke = data_stroke[data_stroke['gender'] == 'Male']
    # female = data_cat[data_cat['gender'] == 'Female']
    # male = data_cat[data_cat['gender'] == 'Male']
