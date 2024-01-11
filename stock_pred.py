# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Apple stock data 
stock_symbol = 'AAPL'
stock_data = yf.download(stock_symbol, start='2022-01-01', end='2023-01-01')

#Format
stock_data['Date'] = stock_data.index
stock_data['Year'] = stock_data['Date'].dt.year
stock_data['Month'] = stock_data['Date'].dt.month
stock_data['Day'] = stock_data['Date'].dt.day
stock_data['Day_of_week'] = stock_data['Date'].dt.dayofweek
stock_data['Weekend'] = (stock_data['Day_of_week'] >= 5).astype(int)

# Use previous day's closing price as a lag feature
stock_data['Previous_Close'] = stock_data['Close'].shift(1)

# Remove missing values
stock_data = stock_data.dropna()

#  features (X) and target (y)
X = stock_data[['Year', 'Month', 'Day', 'Day_of_week', 'Weekend', 'Previous_Close']]
y = stock_data['Close']

# Training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Close Prices')
plt.plot(stock_data.index[-len(y_test):], predictions, label='Predicted Close Prices')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
