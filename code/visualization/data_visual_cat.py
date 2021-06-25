# Visualize the categorical data and compute ratios
# 22-06-2021
# Jana Bersee, Koen Ceton, Jeroen Dijkmans, Dominique Weltevreden

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import prepare_data

# Load the data without one hot encoding
data_cat = prepare_data('healthcare-dataset-stroke-data.csv', one_hot = False, binary = False)

# Make sure all values are strings, so plotting with histplot works
data_cat['hypertension'].replace(to_replace = (0, 1), value = ('no', 'yes'), inplace = True)
data_cat['heart_disease'].replace(to_replace = (0, 1), value = ('no', 'yes'), inplace = True)

# For stroke a 1 means stroke and 0 means no stroke
data_cat['stroke'].replace(to_replace = (0, 1), value = ('no stroke', 'stroke'), inplace = True)

# Create a figure to display multiple figures at once
fig, axes = plt.subplots(4, 2, figsize=(20,10), sharey=True)

# Adjust the settings of the subplot so titles are readable
fig.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=None, hspace=0.25)

# Add a title for the figures combined
fig.suptitle('Figures to show categorical variables in dataset')

# Create hisplots for al categorical data
sns.histplot(ax=axes[0, 0], data = data_cat, x = 'gender', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[0, 1], data = data_cat, x = 'hypertension', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[1, 0], data = data_cat, x = 'heart_disease', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[1, 1], data = data_cat, x = 'ever_married', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[2, 0], data = data_cat, x = 'work_type', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[2, 1], data = data_cat, x = 'residence_type', hue = 'stroke', multiple="stack")
sns.histplot(ax=axes[3, 0], data = data_cat, x = 'smoking_status', hue = 'stroke', multiple="stack")

# Show the plot
plt.show()

def proportion(data, column_name):
    """
    This function prints the ratios of every unique item in a column.
    Ratio = samples with stroke / total samples
    arg:
    data = pandas dataframe
    column_name = str, column of which you want all the ratios per item
    """

    # Find unique items
    uniques = data[column_name].unique()

    # Print instructions about the ratio
    print(f'\nRatio = samples with stroke / total samples. \n'
          f'Column name: {column_name}')

    # Loop over all the unique items
    for item in uniques:

        # Split data into different frames per unique item
        data_item = data[data[column_name] == item]

        # Total samples
        item_total = data_item.shape[0]

        # Retrive rows of specific frame with stroke
        data_item_stroke = data_item[data_item['stroke'] =='stroke']

        # Samples with stroke
        item_stroke = data_item_stroke.shape[0]

        # Calculate ratio
        ratio_item = item_stroke / item_total

        print(f'Ratio of subgroup {item}: {round(ratio_item, 5)}')

# Run functions on every column of the categorical data
for column in data_cat.column:
    proportion(data_cat, column)

# proportion(data_cat, 'gender')
# proportion(data_cat, 'hypertension')
# proportion(data_cat, 'heart_disease')
# proportion(data_cat, 'ever_married')
# proportion(data_cat, 'work_type')
# proportion(data_cat, 'residence_type')
# proportion(data_cat, 'smoking_status')
