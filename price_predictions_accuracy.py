import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

accuracies = []
percentages = []

def get_historical_data(ticker, period="2y"):
    """Fetch historical price data from Yahoo Finance for a given ticker and period."""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    return history

def prepare_lstm_data(history, window=60):
    """Prepare the data for the LSTM model with a specified time window.
    
    Args:
        history (DataFrame): Historical price data.
        window (int): Time window for the LSTM model.
        
    Returns:
        tuple: Scaled features (X), target values (y), and the scaler used.
    """
    # Extract the closing prices
    series = history['Close'].values
    # Scale the data to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series.reshape(-1, 1))
    
    X, y = [], []
    # Create sequences of length `window` and the corresponding target values
    for i in range(len(scaled_series) - window):
        X.append(scaled_series[i:i+window])
        y.append(scaled_series[i+window])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def train_lstm_model(X_train, y_train):
    """Train an LSTM model.
    
    Args:
        X_train (ndarray): Training features.
        y_train (ndarray): Training targets.
        
    Returns:
        model: Trained LSTM model.
    """
    model = Sequential()
    # Add LSTM layers with 50 units each
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50))
    # Add a Dense layer with one output unit
    model.add(Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

def make_future_predictions(model, initial_data, future_steps):
    """Make future predictions using the LSTM model.
    
    Args:
        model: Trained LSTM model.
        initial_data (ndarray): Initial data to start the predictions.
        future_steps (int): Number of future time steps to predict.
        
    Returns:
        ndarray: Future predictions.
    """
    future_predictions = []
    data = np.expand_dims(initial_data, axis=0)  # Expand dimensions for the LSTM model
    
    for _ in range(future_steps):
        prediction = model.predict(data)
        future_predictions.append(prediction[0, 0])
        # Shift the data for the next prediction
        data = np.roll(data, -1, axis=1)
        data[0, -1, 0] = prediction[0, 0]  # Update the last value with the prediction
    
    return np.array(future_predictions)

def evaluate_opportunity(ticker):
    """Evaluate if a stock is a good buying opportunity and display graphs.
    
    Args:
        ticker (str): Stock ticker symbol.
    """
    # Fetch historical data
    history = get_historical_data(ticker)
    window = 60  # Adjust the time window as needed
    X, y, scaler = prepare_lstm_data(history, window)
    
    # Split the data into training and testing sets
    X_train, X_test = X[:-10], X[-10:]
    y_train, y_test = y[:-10], y[-10:]
    
    # Train the LSTM model
    model = train_lstm_model(X_train, y_train)
    
    # Make future predictions
    future_steps = 60  
    initial_data = X[-1]
    future_predictions = make_future_predictions(model, initial_data, future_steps)
    future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1)).flatten()
    
    # Plot historical closing prices
    plt.figure(figsize=(14, 7))
    plt.plot(history.index, history['Close'], label='Historical Closing Price', color='blue')

    # Plot future prediction line
    future_dates = pd.date_range(start=history.index[-1] + pd.DateOffset(1), periods=future_steps, freq='D')
    plt.plot(future_dates, future_predictions, label=f'Future Prediction {future_steps} Future Steps', color='red', linestyle='--')

    plt.title(f'Historical Closing Price and Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate if it's a good buying opportunity
    current_price = history['Close'].iloc[-1]
    future_price = future_predictions[-1]  # Most recent prediction

    # Calculate percentage change
    percentage_change = ((future_price - current_price) / current_price) * 100

    if future_price > current_price:
        print(f"The stock {ticker} might be a good buying opportunity.")
    else:
        print(f"The stock {ticker} does not seem to be a good buying opportunity.")
    
    print(f"Current Price: {current_price:.2f}")
    print(f"Future Prediction: {future_price:.2f}")
    print(f"Percentage Change: {percentage_change:.2f}%")
    percentages.append(f"{percentage_change:.2f}%")

    # Calculate accuracy of the model on the test set
    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions).flatten()
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual_prices - test_predictions) / actual_prices)) * 100
    accuracy = 100 - mape
    print(f"Accuracy for {ticker}: {accuracy:.2f}%")
    accuracies.append(f"{accuracy:.2f}%")

# List of tickers to evaluate
tickers = ['GOOGL','LI','PDD', 'AMZN', 'MDB', 'CRWD', 'BABA']

for ticker in tickers:
    evaluate_opportunity(ticker)

# Display percentage change and accuracy for each ticker
for i in range(len(percentages)):
    print(f'{tickers[i]} -> Percentage Change: {percentages[i]}, Accuracy: {accuracies[i]}')
