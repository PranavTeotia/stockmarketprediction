
import math
import pickle
from flask_caching import Cache
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import yfinance as yf
import tensorflow as tf
import numpy as np
from flask_cors import CORS

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, request, render_template, jsonify
import os

# Initialize Flask app
app = Flask(__name__)
# cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})
CORS(app)

# Constants
MODEL_PATH = 'login/lstmpricepred.h5'
SCALER_PATH = 'login/scaler.pkl'

# Use the 'fivethirtyeight' style for plots
plt.style.use('fivethirtyeight')

# Function to train and save the model
def train_and_save_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists. Skipping training.")
        return

    print("Training the model...")
    # Download data
    ticker = "AAPL"  # Default ticker for training
    df = yf.download(ticker, start='2000-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    # if df.empty:
    #     raise ValueError(f"No data found for {ticker}. Check the ticker symbol or API status.")

    data = df[['Close']]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Save the scaler for later use
    import pickle
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    # Prepare training data
    training_data_len = math.ceil(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_data_len]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build and train the model
    model = Sequential([
        Input(shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    

    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

# Function to get prediction
def get_prediction(ticker):
    try:
        # Load the saved model and scaler
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

        # Fetch data for the specified ticker
        end_date = datetime.today().strftime('%Y-%m-%d')
        df = yf.download(ticker, start='2000-01-01', end=end_date)
        
        data = df[['Close']]

        

        # Prepare last 60 days of data for prediction
        last_60_days = data[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Make prediction
        pred_price = model.predict(X_test)
        
        pred_price = scaler.inverse_transform(pred_price)
        
        

        return  float(pred_price[0][0])

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


@app.route('/')
def home():
    return render_template('index1.html')

# Route to handle prediction and return result
@app.route('/predict', methods=['POST'])
# @cache.cached(timeout=300, query_string=True)  # Cache for 5 minutes

def predict():
    try:
        ticker = request.form.get('ticker', 'AAPL')  # Default ticker as AAPL
        predicted_price = get_prediction(ticker)
        stock_info = yf.Ticker(ticker)

        stock_info = yf.Ticker(ticker).info

        stock_summary = {
            'Company Name': stock_info.get('longName', 'N/A'),
            'Sector': stock_info.get('sector', 'N/A'),
            'Industry': stock_info.get('industry', 'N/A'),
            'Market Cap': f"${stock_info.get('marketCap', 'N/A'):,}" if stock_info.get('marketCap') else 'N/A',
            'Employees': stock_info.get('fullTimeEmployees', 'N/A'),
            'Headquarters': f"{stock_info.get('city', 'N/A')}, {stock_info.get('state', 'N/A')}, {stock_info.get('country', 'N/A')}",
            'Website': stock_info.get('website', 'N/A'),
            '52 Week High': stock_info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': stock_info.get('fiftyTwoWeekLow', 'N/A'),
        }
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max")

        # Create a Plotly candlestick graph
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name=f"{ticker} Stock"
                )
            ]
        )

        # Update graph layout
        fig.update_layout(
            title=f"{ticker} Stock Price - All Time",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )

        # Convert the graph to HTML
        graph_html = pio.to_html(fig, full_html=False)


        # Return the result page with the stock summary and chart
        return render_template(
            'result1.html',
            ticker=ticker,
            predicted_price=round(predicted_price, 2),
            stock_summary=stock_summary,
            graph_html=graph_html
            # plot_filename=plot_filename
        )    
    

    except Exception as e:
        return jsonify({'error': str(e)})
# Running the Flask app
if __name__ == '__main__':
    # Train and save the model if not already done
    # train_and_save_model()
    app.run(port=5000, debug=True)



   







