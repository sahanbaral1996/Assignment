import pandas as pd
import numpy as np
import pickle


# Load data from csv file
data = pd.read_csv('training.csv')

# create a array to hold the data
real_data = []

collected_data = data.values


# filter out Nan values and split the values into columns
'''
['"45";"a;89",....'] to ['45', .....]
'''
for i in range(collected_data.shape[0]):
    try:
        if not np.isnan(collected_data[i][0]):
            pass
    except:
        data = collected_data[i][0].split(';')
        real_data.append(np.array(data))


# convert array to numpy array and dump into a file using pickle
print(np.array(real_data).shape)
datafile = open('real-data', 'ab')
pickle.dump(np.array(real_data), datafile)
datafile.close()
