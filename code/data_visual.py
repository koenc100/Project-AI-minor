# File to visualize the dataset

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import prepare_data

# Prepare data: One-hot encoding, remove NaN, simplify column names
data = prepare_data('healthcare-dataset-stroke-data.csv')

# Returns all samples where a stroke occured
def stroke_samples(name):
    with_stroke = pd.Series(data['stroke'] * data[name])
    return with_stroke[with_stroke !=0]

# Get the age, aveage glucose level and bmi per stroke sample
age_stroke = stroke_samples('age')
gluc_stroke = stroke_samples('avg_glucose_level')
bmi_stroke = stroke_samples('bmi')

# Create subplot
fig, axes = plt.subplots(1, 3, figsize=(20,5), sharey=True)

fig.suptitle('The age, average glucose level and bmi for every occured stroke')

# Add 3 histogram plots to the subplot
sns.histplot(ax=axes[0], x=age_stroke, bins=30)
axes[0].set_title('Age for every stroke')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Stroke count')

sns.histplot(ax=axes[1], x=gluc_stroke, bins=30)
axes[1].set_title('Average glucose level for every stroke')
axes[1].set_xlabel('Average glucose level')

sns.histplot(ax=axes[2], x=bmi_stroke, bins=30)
axes[2].set_title('bmi for every stroke')
axes[2].set_xlabel('bmi')

plt.show()
