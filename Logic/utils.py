import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs.fillna(0)))

def get_trading_data(period="7d", interval="1h"):
    ticker = "BTC-USD"
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure correct column order
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    return df.dropna()

def download_historical_data(ticker="BTC-USD"):
    # yfinance 1h data only available for last 730 days
    end = datetime.now()
    start = end - timedelta(days=729)

    df = yf.download(ticker, start=start, end=end, interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize column order
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Save to Data/ folder
    os.makedirs("Data", exist_ok=True)
    df.to_csv("Data/btc_data.csv")
    print(f"Downloaded {len(df)} rows of data for {ticker} (1h) to Data/ folder")
