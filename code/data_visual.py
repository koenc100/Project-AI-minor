# visualize the data

from data_prep import prepare_data

data, empty_data, labels_data, empty_labels = prepare_data('healthcare-dataset-stroke-data.csv', split_size=(1, 0))

# print(data.info())
print(labels_data)
