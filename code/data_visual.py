# File to visualize the dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import prepare_data

"""
One better way to visualize the data is proptionally.
So instead of plotting the 80+ people with a stroke, you plot it proportional to the 80+ people without a stroke
"""

# Prepare data: One-hot encoding, remove NaN, simplify column names
data = prepare_data('healthcare-dataset-stroke-data.csv')

print(data.info())

# Returns all samples where a stroke did not occur
def stroke_samples(name):
    datacolumn = data[name]
    return datacolumn[data['stroke'] == 1]

# Returns all samples where a stroke did not occur
def non_stroke_samples(name):
    datacolumn = data[name]
    return datacolumn[data['stroke'] == 0]

# Get the age, aveage glucose level and bmi per stroke sample
age_stroke = stroke_samples('age')
gluc_stroke = stroke_samples('avg_glucose_level')
bmi_stroke = stroke_samples('bmi')

# Get the age, aveage glucose level and bmi per non stroke sample
age_nonstroke = non_stroke_samples('age')
gluc_nonstroke = non_stroke_samples('avg_glucose_level')
bmi_nonstroke = non_stroke_samples('bmi')

# Create subplot
fig, axes = plt.subplots(2, 3, figsize=(20,10))

fig.suptitle('The age, average glucose level and bmi for every occured stroke')

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
