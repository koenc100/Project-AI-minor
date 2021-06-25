# Visualize the categorical data and compute ratios
# 22-06-2021
# Jana Bersee, Koen Ceton, Jeroen Dijkmans, Dominique Weltevreden

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import prepare_data

# Load the data without one hot encoding
data_cat = prepare_data('healthcare-dataset-stroke-data.csv', one_hot = False,
                        binary = False, normalize=False)

# Make sure all values are strings, so plotting with histplot works
data_cat['hypertension'].replace(to_replace = (0, 1), value = ('no', 'yes'),
                                    inplace = True)
data_cat['heart_disease'].replace(to_replace = (0, 1), value = ('no', 'yes'),
                                    inplace = True)

# For stroke a 1 means stroke and 0 means no stroke
data_cat['stroke'].replace(to_replace = (0, 1), value = ('no stroke', 'stroke'),
                            inplace = True)

# PLOTS

# Create a figure to display multiple figures at once
fig, axes = plt.subplots(4, 2, figsize=(20,10), sharey=True)

# Adjust the settings of the subplot so titles are readable
fig.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=None,
                    hspace=0.25)

# Add a title for the figures combined
fig.suptitle('Figures to show categorical variables in dataset')

# Pack parameters for each histplot:
parameters = {"data": data_cat, "hue": "stroke", "multiple": "stack"}

# Create histplots for all categorical data
sns.histplot(ax=axes[0, 0], x = 'gender', **parameters)
sns.histplot(ax=axes[0, 1], x = 'hypertension', **parameters)
sns.histplot(ax=axes[1, 0], x = 'heart_disease', **parameters)
sns.histplot(ax=axes[1, 1], x = 'ever_married', **parameters)
sns.histplot(ax=axes[2, 0], x = 'work_type', **parameters)
sns.histplot(ax=axes[2, 1], x = 'residence_type', **parameters)
sns.histplot(ax=axes[3, 0], x = 'smoking_status', **parameters)

# Turn off the bottom right axis as there is no plot there
axes[3, 1].axis("off")

# Show the plot
plt.show()

# PROPORTIONS

def proportion(data, column_name):
    """
    This function prints the ratios of stroke of every unique category in a
    feature (column_name)
    Ratio = samples in that category with stroke / total samples in category

    INPUT
    data : pandas dataframe
    column_name : str, column of which you want all the ratios per item

    OUTPUT
    Does not return anything, just prints the ratio per feature.
    """

    # Find unique items
    uniques = data[column_name].unique()

    # Print instructions about the ratio
    print(f'\nRatio = samples with stroke / total samples. \n'
          f'Feature: {column_name}')

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

        print(f'Ratio of subgroup {item}: {ratio_item:.5f}')

# Run functions on every column of the categorical data
proportion(data_cat, 'gender')
proportion(data_cat, 'hypertension')
proportion(data_cat, 'heart_disease')
proportion(data_cat, 'ever_married')
proportion(data_cat, 'work_type')
proportion(data_cat, 'residence_type')
proportion(data_cat, 'smoking_status')
