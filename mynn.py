import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

import functions as func

dataframe = pd.read_csv('Training.csv')

rows, columns = dataframe.shape

column_names = list(dataframe.columns)

label_encoder = LabelEncoder()

"""Splitting independent 'x' and dependent 'y' variables of dataframe 'df'"""
x = dataframe[column_names[0:len(column_names)-2]]
#Converting to numpy array
x = x.values
#Create target variable
y = dataframe[column_names[len(column_names)-2]].values

y_encoded = label_encoder.fit_transform(y)
epochs = 100

#Weights
fl_weights = np.random.normal(0.0, 2 ** -0.5, (10, 132))
sl_weights = np.random.normal(0.0, 2 ** -0.5, (41, 10))


func.train(x[0],y_encoded[0], fl_weights, sl_weights)

# for e in range(epochs):
#   inputs_ = []
#   correct_predictions = []
#   i = 0
#   while i < 4920:
#       fl_weights, sl_weights = func.train(x[i],y_encoded[i], fl_weights, sl_weights)
#       print(fl_weights, sl_weights)
#       inputs_.append(np.array(x[i]))
#       correct_predictions.append(np.array(y_encoded[i]))
#       i = i + 1;
 
    
# train_loss = func.MSE(func.predict(np.array(inputs_).T,fl_weights,sl_weights ), 
#                       np.array(correct_predictions))

# print('Train_loss: ', train_loss)
 
 
       
      
  