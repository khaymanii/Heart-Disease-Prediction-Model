
# Importing the libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data collection and processing

heart_data = pd.read_csv('heart_disease_data.csv')

heart_data.head()
heart_data.tail()
heart_data.shape
heart_data.info()
heart_data.isnull().sum()


# Statistical measures about the data

heart_data.describe()


# Checking the distribution of Target Variable

heart_data['target'].value_counts()


# Splitting the Features and Target

X = heart_data.drop(columns='target', axis = 1)
y = heart_data['target']


print(X)
print(y)


# Splitting into training and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 2)


print(X.shape, X_train.shape, X_test.shape)


# Training the Model

model = LogisticRegression()


# Training the model with training data

model.fit(X_train, y_train)

# Model Evaluation: accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print('Accuracy on Training data : ', training_data_accuracy)


# Model Evaluation: accuracy on training data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy on Test data : ', test_data_accuracy)


# Building the predictive system

input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)

# change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the numpy array as we are predicting for onky one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
    print('THe Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')




