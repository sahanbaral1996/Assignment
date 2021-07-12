import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from validation import preprocess


# load data from file to get for output label
data_file = open('real-data', 'rb')
data = pickle.load(data_file)
data = np.array(data)
string_label = data[:, -1]

# load normalized input data
file = open('normed-data', 'rb')
input_data = pickle.load(file)

# change output label from yes/No to 0/1
string_label[np.where(string_label == '"yes."')] = 1
string_label[np.where(string_label == '"no."')] = 0

# split data into train and test with 75% train and 25% test
x_train, x_test, y_train, y_test = train_test_split(input_data, string_label, test_size=0.25, random_state=0)

# load logistic regression model
logisticRegr = LogisticRegression()

# train the model
logisticRegr.fit(x_train, y_train)

# get validation data
x_val, y_val = preprocess()
y_pred = logisticRegr.predict(x_val)

# print the performance metric of model
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

