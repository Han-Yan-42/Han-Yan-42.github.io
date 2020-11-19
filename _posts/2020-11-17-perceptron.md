---
title: "Bitcoin and Market Sentiment Indicator"
date: 2020-11-18
tags: [machine learning, data science, big data, neural network]
header:
  image: "/images/bitcoin.jpg"
excerpt: "Bitcoin and Market Sentiment Indicator through LSTM RNN Model"
---

First import all the necessary package for the project:
```python
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import pyspark as spark
    import sklearn as sk
    import seaborn as sns
    import os
    import datetime
    %matplotlib inline
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.layers.experimental import preprocessing
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense
    from keras.layers import Dropout
    from numpy import concatenate
    from sklearn.preprocessing import MinMaxScaler
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    from keras.constraints import max_norm
```
Then use pandas to import file and conduct data preparation and cleaning:

The following is the data after preparation and cleaning:
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture5.png" alt="">

Transform the data into MinMaxScaler between -1 to 1:
```python
  dataset = df.values
  scaler = MinMaxScaler(feature_range=(-1, 1))
  dataset = scaler.fit_transform(dataset)
```
Split the training dataset and testing dataset
```python
  train_size = int(len(dataset) * 0.80)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
```
The following concerns about defining the multivariates and the label, the shape the dataset
```python
train_X, train_y = train[:, 1:10], train[:,:1]
test_X, test_y = test[:,1:10], test[:,:1]
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
```
Define Optimizer to "Adam" and build the model. I included 1 hidden layers with 250 neuron nodes with a dropout measure of 0.2
```python
opt = keras.optimizers.Adam(learning_rate=0.01)
model = Sequential()
model.add(LSTM(250, return_sequences = True,input_shape=(train_X.shape[1], train_X.shape[2])))# Determining # of Neural Nodes
model.add(Dense(1))
model.add(Dropout(0.1))
model.compile(loss='mape', optimizer=opt)
model.summary()
MTS_RNN = model.fit(train_X, train_y, epochs=2000, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle= False)
plt.plot(MTS_RNN.history['loss'], label='train')
plt.plot(MTS_RNN.history['val_loss'], label='test')
plt.legend()
plt.show()
```
Prediction by LSTM RNN
```python
train_predict = model.predict(train_X)    
test_predict = model.predict(test_X)
```
Converting from three dimension to two dimension
```python
train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
```
Reshaping
```python
train_predict = train_predict.reshape((train_X.shape[0],1))
test_predict = test_predict.reshape((test_X.shape[0],1))
```
Concatenate
```python
inv_train_predict = concatenate((train_predict, train_X), axis=1)
inv_test_predict = concatenate((test_predict, test_X), axis=1)
```
Transforming to original scale
```python
inv_train_predict = scaler.inverse_transform(inv_train_predict)
inv_test_predict = scaler.inverse_transform(inv_test_predict)
```
Predicted values on training data
```python
inv_train_predict = inv_train_predict[:,0]
inv_train_predict
```
Predicted values on testing data
```python
inv_test_predict = inv_test_predict[:,0]
inv_test_predict
```
Inverse shaping
```python
train_y = train_y.reshape((len(train_y), 1))
inv_train_y = concatenate((train_y, train_X), axis=1)
inv_train_y = scaler.inverse_transform(inv_train_y)
inv_train_y = inv_train_y[:,0]
test_y = test_y.reshape((len(test_y), 1))
inv_test_y = concatenate((test_y, test_X), axis=1)
inv_test_y = scaler.inverse_transform(inv_test_y)
inv_test_y = inv_test_y[:,0]
```

Accordingly, I choose to use 1 hidden layer with 250 neuron nodes and a regularization dropout of 0.2, which preventing overfitting, to predict the future daily average closing price and obtain the following result with MAPE on validation set of 33%.

Here is the visualization with fitted data and original data:
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture1.png" alt="">

Prediction does not fit the general trend very well, thus I increase the repetition of training to 2000 times. And obtained the following result:
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture3.png" alt="">

As we can observe, the hourly data contains so many noises. So, I group data by its date and take the average of the hourly closing price of the day and summed everything else. Thus obtained the following result with MAPE around 30%.
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture4.png" alt="">
