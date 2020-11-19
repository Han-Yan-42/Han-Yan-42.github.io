---
title: "Bitcoin and Market Sentiment Indicator"
date: 2020-11-18
tags: [machine learning, data science, big data, neural network]
header:
  image: "/images/bitcoin.jpg"
excerpt: "Bitcoin and Market Sentiment Indicator through LSTM RNN Model"
---

# H1 Heading

## H2 Heading

### H3 Heading

First import all the necessary package for the project

Python code block:
```Python
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
