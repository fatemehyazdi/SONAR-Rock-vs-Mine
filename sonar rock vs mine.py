# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 18:14:28 2022

@author: Fatemeh
"""

#Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#______________________________________________________________________________

#Data Collection and Data Processing
sonar_data = pd.read_csv('Copy of sonar data.csv', header=None)
sonar_data.head()
print(sonar_data, "\n")

#number of rows and columns
sonar_data.shape

#describe --> statistical measures of the data
describe = sonar_data.describe()
print(describe, "\n")

#value_counts for each column
count = sonar_data[60].value_counts()
print(count, "\n")

#group data based mind and rock
# M--> mind
# R--> rock
group = sonar_data.groupby(60).mean()
print(group, "\n")

#separating data and Labels
X = sonar_data.drop(columns= 60, axis=1)
y = sonar_data[60]
print(X ,"\n", y,"\n")

#Training and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)
print("X_train :" ,"\n", X_train ,"\n" ," X_test :" ,"\n", X_test , "\n" ,
      "y_train :" ,"\n", y_train , "\n" , "y_test :" ,"\n", y_test , "\n")
#______________________________________________________________________________

#Model Training Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

#Model Evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train) 
print('Accuracy on training data : ', training_data_accuracy,"\n")

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test) 
print('Accuracy on test data : ', test_data_accuracy,"\n")

#______________________________________________________________________________

input_data = (0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)

#convert the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
print(input_data_as_numpy_array,"\n")

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print(input_data_reshaped,"\n")

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')