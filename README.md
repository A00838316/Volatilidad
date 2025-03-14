# Volatilidad
# Install required libraries in Colab
!pip install yfinance pmdarima statsmodels

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

# Function to download stock data
def download_stock_data(ticker, interval, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(interval=interval, start=start_date, end=end_date)
    return df['Close'].dropna()

# Function to perform ADF test for stationarity
def adf_test(series, title=''):
    result = adfuller(series.dropna())
    print(f'ADF Test for {title}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Stationary' if result[1] < 0.05 else 'Non-Stationary')
    print('')

# Function to plot series and check trend/seasonality
def plot_series(series, title=''):
    plt.figure(figsize=(10, 4))
    plt.plot(series)
    plt.title(title)
    plt.show()

# Define tickers and time periods
tickers = ['TPR', 'RL', 'CPRI']  # Replaced 'KER.PA' with 'CPRI'
intervals = {'1m': ('2025-03-10', '2025-03-11'), '5m': ('2025-03-07', '2025-03-11')}
data = {}

# Download data (using proxy dates since March 2025 data may not be fully available)
for ticker in tickers:
    data[ticker] = {}
    for interval, (start, end) in intervals.items():
        # Adjust to historical data if future data isn't available yet
        if pd.to_datetime(end) > pd.Timestamp.now():
            start = '2025-03-01'  # Proxy start date
            end = '2025-03-13'    # Up to today
        data[ticker][interval] = download_stock_data(ticker, interval, start, end)

# Analysis for each stock and interval
for ticker in tickers:
    for interval in intervals.keys():
        series = data[ticker][interval]
        print(f"\nAnalysis for {ticker} - {interval}")
        
        # Plot series
        plot_series(series, f'{ticker} - {interval} Closing Prices')
        
        # ADF Test on original series
        adf_test(series, f'{ticker} - {interval} Original')
        
        # Difference the series if non-stationary
        diff_series = series.diff().dropna()
        adf_test(diff_series, f'{ticker} - {interval} Differenced')
        
        # ACF and PACF plots
        plt.figure(figsize=(12, 4))
        plt.subplot(121); plot_acf(diff_series, ax=plt.gca(), title='ACF')
        plt.subplot(122); plot_pacf(diff_series, ax=plt.gca(), title='PACF')
        plt.show()
        
        # Split into train (90%) and test (10%)
        train_size = int(len(series) * 0.9)
        train, test = series[:train_size], series[train_size:]
        
        # Manual ARIMA model selection (example candidates based on ACF/PACF)
        candidates = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)]
        best_aic = float('inf')
        best_model = None
        best_order = None
        
        for order in candidates:
            try:
                model = ARIMA(train, order=order).fit()
                aic = model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_model = model
                    best_order = order
            except:
                continue
        
        print(f'Best Manual ARIMA Model for {ticker} - {interval}: {best_order}, AIC: {best_aic}')
        
        # Residual analysis
        residuals = best_model.resid
        plt.figure(figsize=(10, 4))
        plt.subplot(121); plt.plot(residuals); plt.title('Residuals')
        plt.subplot(122); plot_acf(residuals, ax=plt.gca(), title='Residual ACF')
        plt.show()
        
        # Forecast with best manual model
        forecast = best_model.forecast(steps=len(test))
        plt.figure(figsize=(10, 4))
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test, label='Test')
        plt.plot(test.index, forecast, label='Forecast')
        plt.title(f'{ticker} - {interval} Forecast (Manual ARIMA)')
        plt.legend()
        plt.show()
        
        # Auto ARIMA
        auto_model = auto_arima(train, seasonal=False, stepwise=True, trace=True)
        print(f'Best Auto ARIMA Model for {ticker} - {interval}: {auto_model.order}')
        
        # Forecast with auto ARIMA
        auto_forecast = auto_model.predict(n_periods=len(test))
        plt.figure(figsize=(10, 4))
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test, label='Test')
        plt.plot(test.index, auto_forecast, label='Auto ARIMA Forecast')
        plt.title(f'{ticker} - {interval} Forecast (Auto ARIMA)')
        plt.legend()
        plt.show()
        
        # Compare performance
        manual_rmse = np.sqrt(np.mean((test - forecast) ** 2))
        auto_rmse = np.sqrt(np.mean((test - auto_forecast) ** 2))
        print(f'Manual ARIMA RMSE: {manual_rmse}')
        print(f'Auto ARIMA RMSE: {auto_rmse}')
        print(f'Residual Normality (Manual): {best_model.test_normality("jarquebera")[0][1]:.4f} (p-value)')
