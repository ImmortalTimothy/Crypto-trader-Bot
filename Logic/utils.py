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

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(series, window=20, std_dev=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def get_trading_data(period="7d", interval="1h", indicators=["SMA_10", "SMA_30", "RSI"]):
    ticker = "BTC-USD"
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    if "SMA_10" in indicators: df['SMA_10'] = df['Close'].rolling(window=10).mean()
    if "SMA_30" in indicators: df['SMA_30'] = df['Close'].rolling(window=30).mean()
    if "RSI" in indicators: df['RSI'] = compute_rsi(df['Close'], window=14)
    if "MACD" in indicators:
        macd, signal = compute_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
    if "Bollinger" in indicators:
        upper, lower = compute_bollinger_bands(df['Close'])
        df['BB_Upper'] = upper
        df['BB_Lower'] = lower

    return df.dropna()

def download_historical_data(ticker="BTC-USD"):
    end = datetime.now()
    start = end - timedelta(days=729)
    df = yf.download(ticker, start=start, end=end, interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    os.makedirs("Data", exist_ok=True)
    save_path = os.path.join("Data", "btc_data.csv")
    df.to_csv(save_path)
    print(f"Downloaded {len(df)} rows of data for {ticker} (1h) to {save_path}")
