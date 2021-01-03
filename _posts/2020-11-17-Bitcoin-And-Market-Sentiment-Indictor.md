---
title: "Bitcoin and Market Sentiment Indicator"
date: 2020-11-18
tags: [machine learning, data science, big data, neural network]
header:
  image: "/images/bitcoin.jpg"
excerpt: "Bitcoin and Market Sentiment Indicator through LSTM RNN Model"
---
Part 1. Industry background
    
Is it a trend? Or is it the future of investing here to stay. Bitcoin has been the point of discussion ever since it entered the market in 2009. One thing is for certain, however -- its value. Over the course of just over a decade and two immense peaks, the price of Bitcoin has risen dramatically from its origins of over a hundredth of a penny to tens of thousands of dollars. From a business perspective, it is helpful to think of blockchain as a type of next-generation business improvement model. Financial institutions have been increasing exploration in blockchain technology in order to upend various different sectors in the modern era.

Additionally, with the rise of factor investing, more and more financial institutions and funds started to utilize multiple factors, such as macroeconomic indicators to develop the investing process. Moreover, they also started to use fundamentals and statistical measures to analyze and explain asset prices and build an investment strategy. This is especially so with recent breakthroughs in statistical means and recently raised artificial intelligence and newly developed technologies and infrastructures to support Big Data, more accurate asset pricing and prediction become a new realization in finance. Overall, through a combination of artificial intelligence and big data, a quantitative solution and data driven model to investment is possible. 

Although its value is indisputable, the persistent question of what truly drives the Bitcoin economy still remains. One unique aspect of bitcoin compared to other financial assets such as stocks is its volatility and impact of public perception. The reason why Twitter could be a powerful tool to predict bitcoin is due to the fact that bitcoin is only valuable if people think it's valuable. Therefore, I can hypothesize that the price of bitcoin is heavily dependent upon demand and sentiment. This is different from stocks because stocks need to factor in the valuation of the company. As a result, social media trends may be a valuable indicator for predicting price fluctuations in bitcoin. Since Twitter has mass amounts of data, it may be a good reflection of whether people are positive on bitcoin or have negative attitudes towards. Thus, I can then use this data to discover if public perception of bitcoin as indicated through Twitter could be used as a price predictor. Moreover, other factors such as social media volume may be very relevant as well.

Throughout the course of this analysis, I will examine several datasets monitoring Bitcoin sentiment, price, and trading volume over time, using a variety of various data tools detailed below in order to discover potential correlations between Bitcoin sentiment and trading, while eliminating potential confounding and omitted variables.

Part 2. Introduction and Screenshots of Dataset

The following screenshot is our data from kaggle:
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture6.png" alt="">
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture7.png" alt="">

The different columns of our data are defined as follows:
-    Compound score: avg sentiment for all tweets (between 0-1 with 1 being as positive as possible)
-    Total volume of tweets: total number of tweets regardless of sentiment
-    Count negatives: The total number of tweets that had a negative sentiment
-    Count positives: the total number of tweets that had a positive sentiment
-    Count neutrals: the total number of tweets that had a neutral sentiment
-    Sent negatives: the avg sentiment among all negative sentiment tweets
-    Sent positives: the avg sentiment among all positive sentiment tweets
-    Count news: the amount of tweets with a link to a news article
-    Count bots: the amount of tweets deemed as bot accounts
-    Close: closing price of bitcoin at the hour
-    Open: opening price of bitcoin at the hour
-    High: highest price of bitcoin during the hour window
-    Low: lowest price of bitcoin during the hour window
-    Volume (BTC): number of bitcoins in the crypto market
-    Volume (currency): total dollar amount of bitcoins in the crypto market

All data was on an hourly basis from August 2017 to December 2018. The data also consisted of 17.7 million tweets in total used for sentiment analysis.

Tweet sentiment was calculated using the VaderSentiment library with 30 additional key words created by the creator of the data set.

Introduction of the Core Question

Our intention is to investigate the correlation between Twitter sentiment and Bitcoin trading trends, in order to best construct and display any potential correlations between the two. Our core questions are the following:
1.    How does Twitter activity impact the prices of Bitcoin?
2.    Is there a correlation between tweet sentiment and Bitcoin prices?
3.    Can you predict the price of Bitcoin using Twitter information?

Target Audience

The target audience who could benefit from the answers to our core questions are crypto investors, crypto asset firms, and regulatory bodies of crypto assets. Especially for those utilizing factor investing techniques to pursue a more accurate stock pricing and model prediction might find this project insightful.

Motivation and Impact

Our target audience is those who are interested in making money through crypto currency and more specifically, Bitcoin. It is beneficial for any investor to know whether or not Twitter activity impacts or correlates with Bitcoin prices. If the data shows that there is a correlation, then Twitter activity should be another metric investors use to make decisions on buying or selling Bitcoin. 

Part 3: Analysis Plan & Tools
Sub-Questions surrounding correlation between bitcoin price and market sentiment

Our list of sub questions are as follows:
1.    What does the tweet sentiment look like at the high and low prices of bitcoin?
2.    How do the different variables in the data correlate with bitcoin prices
3.    How do the different variables in the data correlate with each other?

Tools and Softwares 

Some of the tools I used include:
-    SQL for exploratory queries
-    Python for machine learning models for time series predictions
-    Tableau for visualization
-    Tensorflow for machine learning library for Python
-    SkLearn for machine learning and data tool for Python
-    Pandas and Numpy as a data tool for Python

Part 4: Summary Statistics and Initial Findings

Initial Analysis Results for each of the sub-questions

Initially, I wanted to know if there was any potential for a correlation between Bitcoin prices and Twitter activity. Specifically, I wanted to look into our first subquestion to see what the tweet sentiment looked like for highs and lows of bitcoin prices. I tested the plausibility using simple SQL queries before doing any advanced analysis. I did so by looking at tweet sentiment at the times when Bitcoin was at its highest and lowest prices. I compared the measure of tweet sentiment at times when the price was at its highest to times when it was at its lowest. The 5000 data points with the lowest Bitcoin prices have an average price of $4741.27 and an average compound score of .0943. The 5000 data points with the highest Bitcoin prices have an average price of $10132.09 and an average compound score of .1038. Since the higher prices correlated with the higher compound score I deduced that on the surface, tweet sentiment has the potential to be a predictor of Bitcoin price.
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture8.png" alt="">
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture9.png" alt="">

After that, I started to perform experiments to answer the second question of comparing the correlation between variables and price of bitcoin. In terms of analyzing correlation I wanted to look at all of the different features in the dataset and see how they correlate with Bitcoin prices. To do this I put the data into a pandas dataframe in python and computed the pearson correlation coefficient between all combinations of features. The following is a code snippet of plotting the correlation matrix as a heat map using a cleaned csv that consists of all the different columns in the dataset. The dataset was cleaned through using a linear interpolation to fill any n/a values. This was the data set would use linear estimates based on the values that came before and after for each column to fill in n/a values.
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture10.png" alt="">

Through looking at this correlation matrix I notice how there appears to be a correlation with some of the price variables and tweet sentiment variables. For instance the open, high, close, and low all have higher correlations with total volume of tweets, count negatives, count positives, count neutrals, count news, and count bots then with any other variables. This showed the correlation potential that exists between volume and types of tweets with prices. One important detail here is that the sent negatives and sent positives (avg sentiment score for all negative and positive tweets respectively)  did not have much of a correlation with prices. 

This correlation matrix also answers our third question of seeing which variables within the data tend to have a correlation. I found through this heat map that all of the counts for total volume, neutrals, positives, and negatives all correlate with each other. Meaning the more positive tweets there are then there also tends to be more negative tweets as well. This adds extra complexity when using tweet sentiment as it may indicate that the types of tweets may not add much information to predicting the price of bitcoin as just using tweet volume in general. Moreover, I also found that count bots and count news showed some correlation between the counts for tweet sentiments. This finding is intuitive as more tweets in general would lead to more bots and news articles. Therefore, when doing a deeper analysis I were compelled to use  compund_score, total volume of tweets, count_negatives, count_positives, count_neutrals, sent_negatives, sent_positives, count_news, and count_bots because it consisted of high correlations with bitcoin price. I decided to keep sent negatives and sent positives despite their low correlation because it would still possibly contribute more information. This is also due to the nature of using an LSTM model where having more features for a neural network may be able to learn more complex patterns than visible with the correlation metrics.

The following is another exploratory analysis to look further into correlation focusing on close, count positive, count negative, and total volume. 
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture11.png" alt="">

When looking at this I can see the general trend of total volume, count negatives, and count positives all slightly increasing as close increases. While subtle, it is still significant enough where it may have some predictive power. As a result of this exploratory analysis I wanted to focus most of our deliverable on trying to form a high performing LSTM model since having predictive power over crypto currencies is inherently valuable.

Part 5: Deeper Analysis

Based upon the findings our deeper analysis is very intertwined with advanced analytics. Since using machine learning was one of the main parts of our projects I dedicated the most time to it. The answers to the sub questions also show potential correlation between bitcoin prices and tweet sentiment - especially in terms of volume of tweets and their respective sentiment groups. Therefore, before diving into building a recurrent neural network for time series forecasting, I used tableau to learn more about the data.

In order to better visualize and understand the data collected, I decided to create a series of Tableau visualizations in order to better depict/display the relationships found. I first looked at the relationship between positive and negative Bitcoin sentiment over a series of a year and a half.
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture12.png" alt="">

This figure allowed us to identify sentiment type as a non confounding variable to our sentiment/price analysis. Since positive and negative sentiment share what is roughly the same trend, I can see that positive and negative sentiment are highly correlated to each other, and is primarily affected by total sentiment -- meaning the more Bitcoin is discussed on Twitter, the more both positive and negative sentiment will be had and vice versa.

I then decided to visualize the effect of sentiment on total aggregated Bitcoin price to validate the correlation matrix discussed prior. In the findings below, I see that aggregate Bitcoin price follows a similar trend to both sentiment types.
<img src="{{ site.url }}{{ site.baseurl }}/images/Bitcoin/Picture13.png" alt="">

The trends in aggregate trading price and positive and negative sentiment seem to almost mirror each other which suggests that there might be great predictive performance when utilizing the count of the different types of sentiment. Overall, these visualizations helped reaffirm the feasibility of producing a relatively effective neural network. As a result our deeper analysis is the advanced analytics as explained in the next section.


Part 6: Advanced Analytics

For advanced analytics, I approached the data through recurrent neural networks'  Long Short-Term Memory Model. I chose to use the LSTM model because it excels at time series analysis. Recurrent neural networks utilize feedback loops to provide a memory within the network. This way they can use past context in light of the new prediction. This way information can persist within the network to enhance predictions. This can specifically help utilize recent information, such as past prices and twitter activity in the past, while training to help make a current prediction. This is why time series data, such as bitcoin prices, work well with recurrent neural networks. LSTM models are a special case of recurrent neural networks that are able to learn long-term dependencies. The LSTM architecture has repeated modules to give it a chain-like structure. The unique part of LSTMs is that each module contains four interacting layers. These four layers are what is responsible for holding certain information within the module to help the neural network learn. 

I chose to use compund_score, total volume of tweets, count_negatives, count_positives, count_neutrals, sent_negatives, sent_positives, count_news, and count_bots as continuous multivariates and hourly closing price of Bitcoin as the label. In the model, I choose to deploy 1 hidden layer with 250 neuron nodes and a dropout regularization of 20% in order to prevent overfitting. Additionally, for model optimizer, I chose “adam” and with a learning rate of 0.01. Finally, I assign epochs to 1000 times. For each epoch, I open a for loop that iterates over the dataset, in batches. 

Both predicted train and predicted test do not fit the actual data very well. In fact, when I first  it is evidently that the performance of the model can be improved. Thus, I decided to increase the epochs to 2000. In another word, I open a for loop that iterates over the dataset for 2000 times in order to minimize the cost function.

Both predicted training set and predicted testing set fit the actual train and actual test much better. It captures trend, seasonal behavior and cyclical behavior much better. Additionally, it achieves a mean average percentage error (MAPE calculated as: (actual test - fitted test)/actual test)) of 43%. Compared with the previous Model 1’s MAPE of 49% , Model 2’s MAPE of 43% improved whereas it still contains a vast amount of noise and does not capture many of the patterns. One potential reason could be that I used hourly data rather than daily data. People exhibit trading behavior that they respond quickly within days but not necessarily within certain hours. 

Accordingly, I rolled up the dataset through moving the dimensional hierarchy to daily, averaging hourly closing prices and summing all multivariate. Keeping same model parameters and obtaining Model 3.

I are able to observe that the predicted train fits the actual train relatively well. In fact MAPE for predicted train achieves around 5% which could be considered as a fairly small MAPE. Nonetheless, in its predicted test, the MAPE increased to 32%. That could be considered as a fairly large error. That could be attributed to relatively simple classification of sentiments tweets exemplified. Positive Tweet and Negative Tweet also contains different level of positive and negative emotion. I can further classify sentiment associated with tweet through text sentiment analysis through Deep Convolutional Neural Network.

Part 7: Impact Statement & Summary

I believe that our findings exemplify a moderately significant correlation between Twitter sentiment and Bitcoin trading price. Keeping potential confounding variables at a minimum, I were able to display a key correlation between sentiment and price while simultaneously disproving any impact of the type of correlation. By examining the graph generated by Tableau in the figure below (duplicated from prior for the purpose of readability), I can examine that both positive and negative sentiment follow precisely the same trend, concluding that sentiment type has no significant impact on Bitcoin prices (Red being aggregated Bitcoin trade volume by price, blue being negative sentiment, and orange being positive sentiment).

Our business analytics project should provide firms looking to analyze and aggregate historic Bitcoin sentiment data to predict future trends while disproving potential omitted/confounding variables. Narrowing on twitter volume relating to bitcoin seems to be most relevant. I have performed the task of validating the dataset and identifying correlations, meaning our dataset could then be fed into a neural network for crypto asset firms to be used to predict future crypto price trends should they collect further recurring data. The correlations I have identified will prove useful to individual investors as well, as they could then identify future upticks in Bitcoin sentiment to gauge upcoming investments. From a quantitative standpoint, I have achieved a MAPE of 5% for the training data. While this doesn’t generalize, it still shows that the model was able to learn predictive behavior using tweet sentiment analysis for the data it was trained on. This shows promise because I only had access to a limited amount of data. Therefore, it is expected that generalization to a test set would be difficult. However, with large amounts of data these methods show potential to perform quite well in performance. Therefore, our recommendation would be to use an LSTM model involving tweet sentiment volume but with much more data to effectively be able to generalize and predict future prices. 

As a whole, our analysis is by no means concrete. What I have demonstrated is a correlation between twitter activity and bitcoin prices. Specifically, the volume of tweets relating to bitcoin seem to be the most impactful. This does not entail any sort of causation. However, based on our results of the LSTM model, deep recurrent neural networks demonstrate promise in being able to learn to predict bitcoin prices using twitter activity - including sentiment - as features. In order to generalize unseen data, one should utilize more data for the training. This includes data that is more granular than the hour and perhaps analyze more tweets. In addition to that, more sophisticated measures for sentiment analysis could be used. Based upon the importance of twitter volume, I expect that other sources of data, such as google trends, could be useful as a predictor as well. Overall, our recommendation is that anyone investing in bitcoin should try to utilize twitter activity and tweet sentiment in their analysis on valuing future prices of bitcoin.


The following is the python code demostration

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
