# File to visualize the dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import prepare_data

"""
DESCRIPTION:
This file visualizes the nummeric data of the stroke Dataset.
In the upper three plots, the age, average glucose level and bmi are plotted against the strokes occured.
In the 3 plots beneath, the age, average glucose level and bmi are plotted cumulatively, against its density.

DISCUSSION:
Clearly, the chances of having a stroke increase as the age increases.
The plots of the avg. gl. level tell us that

"""

# Prepare data: One-hot encoding, remove NaN, simplify column names
data = prepare_data('healthcare-dataset-stroke-data.csv')

# Create subplot
fig, axes = plt.subplots(2, 3, figsize=(20,10))

fig.suptitle('The strokes or nonstrokes for age, average glucose level and bmi ')

# Add 3 histogram plots to the subplot
sns.histplot(ax=axes[0, 0], x=data['age'], hue=data['stroke'], bins=30, kde=True)
axes[0, 0].set_title('Age for every stroke')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Stroke count')

sns.histplot(ax=axes[0, 1], x=data['avg_glucose_level'], hue=data['stroke'], bins=30, kde=True)
axes[0, 1].set_title('Average glucose level for every stroke')
axes[0, 1].set_xlabel('Average glucose level')

sns.histplot(ax=axes[0, 2], x=data['bmi'], hue=data['stroke'], bins=30, kde=True)
axes[0, 2].set_title('bmi for every stroke')
axes[0, 2].set_xlabel('bmi')

# Add 3 proportional histograms to the subpots
sns.histplot(ax=axes[1, 0], x=data['age'], hue=data['stroke'], element="step", fill=False,
    cumulative=True, stat="density", common_norm=False,)
axes[1, 0].set_title('Age for every stroke')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Stroke count')

sns.histplot(ax=axes[1, 1], x=data['avg_glucose_level'], hue=data['stroke'], element="step", fill=False,
    cumulative=True, stat="density", common_norm=False,)
axes[1, 1].set_title('Average glucose level for every stroke')
axes[1, 1].set_xlabel('Average glucose level')

sns.histplot(ax=axes[1, 2], x=data['bmi'], hue=data['stroke'], element="step", fill=False,
    cumulative=True, stat="density", common_norm=False,)
axes[1, 2].set_title('bmi for every stroke')
axes[1, 2].set_xlabel('bmi')

plt.show()
