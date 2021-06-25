# File to visualize the dataset

"""
DESCRIPTION:
This file visualizes the numeric data of the stroke Dataset.
In the upper three plots, the age, average glucose level and bmi are plotted against the strokes occured.
In the 3 plots beneath, the age, average glucose level and bmi are plotted cumulatively, against its density.

DISCUSSION:
Clearly, the chances of having a stroke increase as the age increases.
The plots of the avg. gl. level tell us that most stroke occur around 70.
Proporionally to strokes not occuring, around 200, there is also a significant chance of having a stroke.
The plots on bmi tell us that the bmi of a person has very little to do with the probability of having a stroke.
The density distribution is more or less the same.
"""
from .. import data_processing
from data_processing import prepare_data

# Prepare data: One-hot encoding, remove NaN, simplify column names
data = prepare_data('healthcare-dataset-stroke-data.csv')

# Create subplot
fig, axes = plt.subplots(2, 3, figsize=(20,10))

# Make subtitle
fig.suptitle('The strokes or nonstrokes for age, average glucose level and bmi')

# Add 3 histogram plots to the subplot
sns.histplot(ax=axes[0, 0], x=data['age'], hue=data['stroke'], bins=30, kde=True)
axes[0, 0].set_title('Age per stroke / non-stroke')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Stroke count')

sns.histplot(ax=axes[0, 1], x=data['avg_glucose_level'], hue=data['stroke'], bins=30, kde=True)
axes[0, 1].set_title('Average glucose level per stroke / non-stroke')
axes[0, 1].set_xlabel('Average glucose level (mg/DL)')

sns.histplot(ax=axes[0, 2], x=data['bmi'], hue=data['stroke'], bins=30, kde=True)
axes[0, 2].set_title('bmi per stroke / non-stroke')
axes[0, 2].set_xlabel('bmi')

# Add 3 cumulative density histograms to the subpots
sns.histplot(ax=axes[1, 0], x=data['age'], hue=data['stroke'], element="step", fill=False,
    cumulative=True, stat="density", common_norm=False,)
axes[1, 0].set_title('cumulative histogram of age vs stroke / non-stroke')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Density')

sns.histplot(ax=axes[1, 1], x=data['avg_glucose_level'], hue=data['stroke'], element="step", fill=False,
    cumulative=True, stat="density", common_norm=False,)
axes[1, 1].set_title('cumulative histogram of avg. gl. level vs stroke / non-stroke')
axes[1, 1].set_xlabel('Average glucose level (mg/DL)')

sns.histplot(ax=axes[1, 2], x=data['bmi'], hue=data['stroke'], element="step", fill=False,
    cumulative=True, stat="density", common_norm=False,)
axes[1, 2].set_title('cumulative histogram of bmi vs stroke / non-stroke ')
axes[1, 2].set_xlabel('bmi')

plt.show()
