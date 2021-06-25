# File to visualize the dataset
# 25-06-21
# Koen, Jana, Dominique, Jeroen

"""
DESCRIPTION:
This file visualizes the numeric data of the stroke Dataset.
In the upper three plots, the age, average glucose level and bmi are plotted
against the strokes occured.
In the 3 plots beneath, the age, average glucose level and bmi are plotted
cumulatively, against its density.

DISCUSSION:
Clearly, the chances of having a stroke increase as the age increases.
The plots of the avg. gl. level tell us that most stroke occur around 70.
Proporionally to strokes not occuring, around 200, there is also a significant
chance of having a stroke.
The plots on bmi tell us that the bmi of a person has very little to do with the
probability of having a stroke.
The density distribution is more or less the same.
"""
from data_processing import prepare_data
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare the data: One-hot encoding, remove NaN, simplify column names
data = prepare_data('healthcare-dataset-stroke-data.csv')

# Create subplot
fig, axes = plt.subplots(2, 3, figsize=(20,10))

# Make a subtitle
fig.suptitle('The strokes or nonstrokes for age, average glucose level and bmi')

# Add 3 histogram plots to the subplot, showing the age distribution for strokes
# Pack the histogram parameters and unpack them for every histogram
hist_parameters = {'hue' : data['stroke'], 'bins' : 30, 'kde' : True}

sns.histplot(ax=axes[0, 0], x=data['age'], **hist_parameters)
axes[0, 0].set_title('Age per stroke / non-stroke')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Stroke count')

sns.histplot(ax=axes[0, 1], x=data['avg_glucose_level'], **hist_parameters)
axes[0, 1].set_title('Average glucose level per stroke / non-stroke')
axes[0, 1].set_xlabel('Average glucose level (mg/DL)')

sns.histplot(ax=axes[0, 2], x=data['bmi'], **hist_parameters)
axes[0, 2].set_title('bmi per stroke / non-stroke')
axes[0, 2].set_xlabel('bmi')

# Add 3 cumulative density histograms to the subplots
# Pack the cumulative density histograms parameters
cumudens = {"hue": data['stroke'], "element" : "step", "fill" : False,
            "cumulative" : True, "stat" : "density", "common_norm" : False}

sns.histplot(ax=axes[1, 0], x=data['age'], **cumudens )
axes[1, 0].set_title('cumulative histogram of age vs stroke / non-stroke')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Density')

sns.histplot(ax=axes[1, 1], x=data['avg_glucose_level'], **cumudens)
axes[1, 1].set_title('cumulative histogram of avg. gl. level vs stroke / non-stroke')
axes[1, 1].set_xlabel('Average glucose level (mg/DL)')

sns.histplot(ax=axes[1, 2], x=data['bmi'], **cumudens)
axes[1, 2].set_title('cumulative histogram of bmi vs stroke / non-stroke ')
axes[1, 2].set_xlabel('bmi')

plt.show()
