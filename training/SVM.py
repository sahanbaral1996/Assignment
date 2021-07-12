from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from training.validation import preprocess

# load data to get output label
data_file = open('real-data', 'rb')
data = pickle.load(data_file)
data = np.array(data)
string_label = data[:, -1]

# load normalized data
file = open('normed-data', 'rb')
input_data = pickle.load(file)

# changed output label to 0/1
string_label[np.where(string_label == '"yes."')] = 1
string_label[np.where(string_label == '"no."')] = 0

# split data into train and test with 75 and 25 as train and test
x_train, x_test, y_train, y_test = train_test_split(input_data, string_label, test_size=0.25, random_state=0)

# load SVC model with polynomial kernel and train
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(x_train, y_train)

# load the preprocessed validation data
x_val, y_val = preprocess()
y_pred = svclassifier.predict(x_val)

# print performance metric
print(confusion_matrix(y_val, y_pred))

print(classification_report(y_val, y_pred))


