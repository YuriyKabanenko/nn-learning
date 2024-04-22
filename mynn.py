import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

import functions as func

dataframe = pd.read_csv('Training.csv')
test_dataframe = pd.read_csv('Testing.csv')

rows, columns = dataframe.shape

column_names = list(dataframe.columns)

label_encoder = LabelEncoder()

x = dataframe[column_names[0:len(column_names)-2]]
x_test = test_dataframe[column_names[0:len(column_names)-2]]
#Converting to numpy array
x = x.values
x_test = x_test.values
#Create target variable
y = dataframe[column_names[len(column_names)-2]].values
y_test = test_dataframe[column_names[len(column_names)-2]].values

y_encoded = label_encoder.fit_transform(y)
epochs = 1

#Weights
fl_weights = np.random.normal(0.0, 5, (10, 132))
sl_weights = np.random.normal(0.0, 5, (41, 10))

unique_elements, counts = np.unique(y_encoded, return_counts=True)

func.train(x[0], y_encoded[0], fl_weights, sl_weights)

inputs_ = []
correct_predictions = []
model_prediction = []


for e in range(epochs):
  i = 0
  while i < 4920:
      fl_weights, sl_weights = func.train(x[i],y_encoded[i], fl_weights, sl_weights)
      print(fl_weights, sl_weights)
      inputs_.append(np.array(x[i]))
      correct_predictions.append(np.array(y_encoded[i]))
      print(i)
      i = i + 1;
   
print('fl_weights: ', fl_weights)
print('sl_weights: ', sl_weights) 
    
for val in x_test:
    model_prediction.append(func.predict(val,fl_weights, sl_weights))
    

 
       
      
  