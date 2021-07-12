import pickle
import numpy as np

# Load data from pickle file
data_file = open('real-data', 'rb')
data = pickle.load(data_file)

data_file.close()
unique_values = {}

# find unique values in each columns
for i in range(data.shape[1]):
    unique_values[i] = np.unique(data[:, i:i+1])

array = []

# filter out columns if number of unique values is less than 4
for i in unique_values:
    if unique_values[i].size > 3:
        column = data[:, i:i+1].reshape(2009)
        # place -1 in place for NA
        for j in range(column.shape[0]):
            if column[j] == 'NA':
                column[j] = -1
        # place nan inplace of -1
        # had to this double step because string NA was not useful for using numpy.fillna
        mod_data = column.astype(np.float64)
        mod_data[mod_data == -1] = np.nan

        # find mean of the columns and indexes of nan values and place the mean in nan
        col_mean = np.nanmean(mod_data, axis=0)
        indexes = np.where(np.isnan(mod_data))
        for k in indexes[0]:
            mod_data[k] = col_mean

        # normalize the data set
        data_normed = mod_data/mod_data.max(axis=0)
        array.append(data_normed.T)

# dump the normed data into a file
file = open('normed-data', 'ab')
pickle.dump(np.asarray(array).T, file)
file.close()


