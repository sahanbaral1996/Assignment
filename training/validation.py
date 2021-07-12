import pandas as pd
import numpy as np
import sys


# preprocess for validation data
def preprocess():
    data = pd.read_csv('validation.csv', error_bad_lines=False)
    real_data = []
    collected_data = data.values

    # filter out Nan values and split the values into columns
    for i in range(collected_data.shape[0]):
        try:
            if not np.isnan(collected_data[i][0]):
                pass
        except:
            data = collected_data[i][0].split(';')
            # filter out rows where there were missing columns data
            if not len(data) < 12:
                real_data.append(np.array(data))

    unique_values = {}

    # find unique values in each columns
    for i in range(np.array(real_data).shape[1]):
        unique_values[i] = np.unique(np.array(real_data)[:, i:i+1])

    array = []

    # filter out columns if number of unique values is less than 4
    for i in unique_values:
        if unique_values[i].size > 3:
            column = np.array(real_data)[:, i:i+1].reshape(114)

            # place -1 in place for NA
            for j in range(column.shape[0]):
                if column[j] == 'NA':
                    column[j] = -1

            # place nan inplace of -1
            # had to this double step because string NA was not useful for using numpy.fillna
            mod_data = column.astype(np.float64)
            mod_data[mod_data == -1] = np.nan

            # find mean of the columns and indexes of nan values and place the mean in dataset
            col_mean = np.nanmean(mod_data, axis=0)
            indexes = np.where(np.isnan(mod_data))
            print(indexes)
            for k in indexes[0]:
                mod_data[k] = col_mean

            # normalize the data set
            data_normed = mod_data/mod_data.max(axis=0)
            array.append(data_normed.T)

    # change label data form yes/No to 0/1
    val_label = np.array(real_data)[:, -1]

    val_label[np.where(val_label == 'yes.')] = 1
    val_label[np.where(val_label == 'no.')] = 0

    return np.asarray(array).T, val_label


preprocess()
