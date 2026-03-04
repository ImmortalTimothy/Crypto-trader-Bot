import pandas as pd
import numpy as np

def preprocess_data(csv_path):
    # Data structure from head: Price,Close,High,Low,Open,Volume
    # Ticker,BTC-USD,BTC-USD,BTC-USD,BTC-USD,BTC-USD
    # Date,,,,,
    data = pd.read_csv(csv_path, skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)

    # Simple Technical Indicators
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['RSI'] = compute_rsi(data['Close'], window=14)

    data = data.dropna().reset_index(drop=True)
    return data

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    # Handle division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs.fillna(0)))
    return rsi

if __name__ == "__main__":
    df = preprocess_data("btc_data.csv")
    df.to_csv("btc_cleaned.csv", index=False)
    print("Data preprocessed and saved to btc_cleaned.csv")
    print(df.head())
