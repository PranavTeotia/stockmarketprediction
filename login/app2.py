import math
import pickle
from flask import send_file
from flask_caching import Cache
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential ,load_model
from tensorflow.keras.layers import Dense, LSTM, Input
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for rendering

from flask import Flask, request, render_template, jsonify
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
# cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})
CORS(app)
# Constants for model paths
MODEL_PATH_LSTM = 'login/pricetrend.h5'
SCALER_PATH = 'login/scaler.pkl'


def load_lstm_model():
    if os.path.exists(MODEL_PATH_LSTM):
        print("Model already exists. Skipping training.")
        return

    print("Training the model...")



    # Fetch data for the specified ticker
    ticker = "AAPL"
    end_date = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start='2000-01-01', end=end_date)
    df1 = df.reset_index()['Close']

    # Scale the data

    scaler = MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    import pickle
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

        # split into train and test
    training_size = int(len(df1)*0.65)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size,:], df1[training_size:len(df1), :1]


    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i+time_step, 0])
        return np.array(dataX), np.array(dataY)
    # Prepare data for prediction
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], 1)
    

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape = (X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=64,verbose=1)

    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    # Reshape predictions to 2D for inverse scaling
    train_predict = train_predict.reshape(-1, train_predict.shape[-1])
    test_predict = test_predict.reshape(-1, test_predict.shape[-1])
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    from sklearn.metrics import mean_squared_error
    math.sqrt(mean_squared_error(y_train,train_predict))
    math.sqrt(mean_squared_error(ytest,test_predict))

    # plotting
    Look_back=100
    trainPredictPlot= np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[Look_back:len(train_predict)+Look_back, :] = train_predict
    # shift test prediction for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(Look_back*2)+1:len(df1)-1, :] = test_predict
    # plot baseline and prediction
    plt.figure(figsize=(12,6))

    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    x_input=test_data[2076:].reshape(1,-1)
    x_input.shape

    x_input=test_data[2076:].reshape(1,-1)
    x_input.shape

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
        # Save the model
    model.save(MODEL_PATH_LSTM)
    print(f"Model saved at {MODEL_PATH_LSTM}")


def get_prediction_lstm(ticker):
    try:
        # Load model and scaler
        model = load_model(MODEL_PATH_LSTM)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

        # Fetch data for the specified ticker
        end_date = datetime.today().strftime('%Y-%m-%d')
        df = yf.download(ticker, start='2000-01-01', end=end_date)


        if df.empty:
            raise ValueError("No data fetched for the specified ticker.")
        if 'Close' not in df.columns:
            raise ValueError("The 'Close' column is missing in the fetched data.")

        # Extract and verify the 'Close' column
        df1 = df['Close']
#         print("Type of df1:", type(df1))
#         print("Shape of df1:", df1.shape)
#         print("First few values of df1:", df1.head())
# # Ensure 'df1' is correctly formatted
        if isinstance(df1, pd.DataFrame) and df1.shape[1] == 1:
            df1 = df1.squeeze()

        # Convert to a scaled NumPy array
        df1_scaled = scaler.transform(df1.values.reshape(-1, 1))

        # Use the last 100 data points
        temp_input = df1_scaled[-100:].flatten().tolist()

        lst_output = []
        for _ in range(30):  # Predict the next 30 days
            x_input = np.array(temp_input[-100:]).reshape(1, 100, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])

        # Convert predictions back to the original scale
        lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten().tolist()

        
        df3 = df1.values.tolist() + lst_output  # Convert Series to list
        # print("Combined data length:", len(df3))

        # Generate day indices for prediction
        day_pred = np.arange(len(df1), len(df1) + 30)

        return lst_output, df3, day_pred
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise



# Route for the home page
@app.route('/')
def home():

        # return 'Hello, world!'
    return render_template('index2.html')




@app.route('/predict_lstm', methods=['POST'])
# @cache.cached(timeout=300, query_string=True)  # Cache for 5 minutes

def predict_lstm_route():
    try:
        ticker = request.form.get('ticker', 'AAPL')  # Default ticker as AAPL
        if not ticker:
            raise ValueError("Ticker symbol is missing or invalid.")

        # Call your prediction function
        predicted_prices, combined_data, day_pred = get_prediction_lstm(ticker)

        # Parameters for moving averages
        short_window = 50  # 50-day moving average
        long_window = 200  # 200-day moving average

        if len(combined_data) < long_window:
            raise ValueError(f"Not enough data to calculate a {long_window}-day moving average.")

        # Calculate moving averages
        ma_short = np.convolve(combined_data, np.ones(short_window) / short_window, mode='valid')
        ma_long = np.convolve(combined_data, np.ones(long_window) / long_window, mode='valid')

        # --- Historical Prices with Moving Averages ---
        plt.figure(figsize=(10, 6))

        ma_short_x = np.arange(short_window - 1, len(ma_short) + short_window - 1)
        ma_long_x = np.arange(long_window - 1, len(ma_long) + long_window - 1)

        plt.plot(combined_data, label="Historical Prices", color="blue")
        plt.plot(ma_short_x, ma_short, label=f"{short_window}-Day Moving Average", color="green")
        plt.plot(ma_long_x, ma_long, label=f"{long_window}-Day Moving Average", color="red")

        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"Historical Prices with Moving Averages for {ticker}")
        plt.legend(loc='best')

        historical_plot_filename = 'login/static/historical_plot.png'
        plt.savefig(historical_plot_filename)
        plt.close()

        # --- Predicted Prices ---
        plt.figure(figsize=(10, 6))

        actual_prices_x = np.arange(
            len(combined_data) - len(predicted_prices) - 100, 
            len(combined_data) - len(predicted_prices)
        )
        plt.plot(
            actual_prices_x,
            combined_data[-len(predicted_prices) - 100:-len(predicted_prices)],
            label="Actual Prices",
            color="blue"
        )
        plt.plot(day_pred, predicted_prices, label="Predicted Future Trend", color="orange")

        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"Stock Trend Prediction for {ticker}")
        plt.legend(loc='best')

        prediction_plot_filename = 'login/static/prediction_plot.png'
        plt.savefig(prediction_plot_filename)
        plt.close()

        # Pass both file names to the template
        return render_template(
            'result2.html', 
            ticker=ticker, 
            historical_plot_filename=historical_plot_filename, 
            prediction_plot_filename=prediction_plot_filename
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

    

   # Running the Flask app
if __name__ == '__main__':
    app.run(port=5001, debug=True)
