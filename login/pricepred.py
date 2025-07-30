# https://www.youtube.com/watch?v=QIUxPv5PJOY&list=PLDBrQ1zemiv_xgppk1athh8rX6sAVFHSb&index=8&ab_channel=ComputerScience%28compsci112358%29
import math
import pandas_datareader as web
import yfinance as yf
import datetime as dt
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# print(plt.style.available)
plt.style.use('fivethirtyeight')

def get_prediction(ticker):
    end_date = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker , start ='2000-01-01', end=end_date)
    print(df)
    # visualize closing price
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.show()



    # create a new dataframe with close
    data = df.filter(items=[('Close', ticker)])# print(data.head())           # Check the first few rows
    print(data.shape)
    # convert the dataframe to numpy array
    dataset = data.values

    # # get number of rows to train the Model 80% of data for training
    training_data_len = math.ceil(len(dataset)* .8)


    # scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # create the training dataset
    # create scaled training dataset
    train_data = scaled_data[0:training_data_len, :]
    # split data into x train and y train 
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 60:
          print(x_train)
          print(y_train)

    #  convertx_train and y_train to numpy array
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # build lstm model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))


    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train,y_train, batch_size=1,epochs=1)

    # create testing dataset
    # create a new array containing scaled values from index 2904 to 3704
    test_data = scaled_data[training_data_len - 60: , :]
    # create the dataset x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, (len(test_data))):
        x_test.append(test_data[i-60:i, 0])

    # convert the data to a numpy array 
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # get the models predicted price value 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # get the rmse
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    rmse


    #  plot the data 
    train = data[:training_data_len]
    valid = data[training_data_len: ]
    valid['Predictions'] = predictions
    # visualize the model
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize= 18)
    plt.ylabel('Close Price', fontsize = 18)

    plt.plot(train[['Close']])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train','Val','Predictions'], loc = 'lower right')
    plt.show()



    #  get the quote
    stock_quote = yf.download(ticker, start ='2010-01-01', end=end_date)
    #  create a new data frame 
    new_df = stock_quote.filter(items=[('Close', ticker)])
    # get last 60 d close price and convert to array
    last_60_days = new_df[-60:].values
    # scale data to b 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    # create an empty list 
    X_test = []
    # append past 60 d
    X_test.append(last_60_days_scaled)
    # convert the X_test to a numpy array
    X_test = np.array(X_test)
    # reshape
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # get predicted price
    pred_price = model.predict(X_test)
    # undo scaling
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price)

    # pickle.dump(Sequential, open('model.pkl','wb'))






     












