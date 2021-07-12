import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('../training/validation.csv', error_bad_lines=False)
real_data = []
collected_data = data.values

for i in range(collected_data.shape[0]):
    try:
        if not np.isnan(collected_data[i][0]):
            pass
    except:
        data = collected_data[i][0].split(';')
        if not len(data) < 12:
            real_data.append(np.array(data))

unique_values = {}
for i in range(np.array(real_data).shape[1]):
    unique_values[i] = np.unique(np.array(real_data)[:, i:i+1])

array = []
for i in unique_values:
    if unique_values[i].size > 3:
        column = np.array(real_data)[:, i:i+1].reshape(114)
        for j in range(column.shape[0]):
            if column[j] == 'NA':
                column[j] = -1
        mod_data = column.astype(np.float64)
        mod_data[mod_data == -1] = np.nan
        col_mean = np.nanmean(mod_data, axis=0)
        indexes = np.where(np.isnan(mod_data))
        for k in indexes[0]:
            mod_data[k] = col_mean
        data_normed = mod_data/mod_data.max(axis=0)
        array.append(data_normed.T)


file = open('normed-data', 'ab')
pickle.dump(np.asarray(array).T, file)
file.close()
