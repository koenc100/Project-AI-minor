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

# Returns all samples where a stroke occured
def stroke_samples(name):
    with_stroke = pd.Series(data['stroke'] * data[name])
    return with_stroke[with_stroke !=0]

def non_stroke_samples(name):
    without_stroke = 4

# Proportional plot


# Get the age, aveage glucose level and bmi per stroke sample
age_stroke = stroke_samples('age')
gluc_stroke = stroke_samples('avg_glucose_level')
bmi_stroke = stroke_samples('bmi')

# Create subplot
fig, axes = plt.subplots(2, 3, figsize=(20,10), sharey=True)

fig.suptitle('The age, average glucose level and bmi for every occured stroke')

# Add 3 histogram plots to the subplot
sns.histplot(ax=axes[0, 0], x=age_stroke, bins=30)
axes[0, 0].set_title('Age for every stroke')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Stroke count')

sns.histplot(ax=axes[0, 1], x=gluc_stroke, bins=30)
axes[0, 1].set_title('Average glucose level for every stroke')
axes[0, 1].set_xlabel('Average glucose level')

sns.histplot(ax=axes[0, 2], x=bmi_stroke, bins=30)
axes[0, 2].set_title('bmi for every stroke')
axes[0, 2].set_xlabel('bmi')

plt.show()
