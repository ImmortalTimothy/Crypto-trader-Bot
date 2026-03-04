import pandas as pd
import numpy as np

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)

    # Flatten multi-index columns
    # The current structure seems to be (Price, Ticker, Date) or something similar based on head output
    # Let's re-read it more carefully or just skip the first few rows if they are problematic

    # Actually, let's try reading it with just header=0 and see
    df = pd.read_csv(csv_path)
    # Row 0: Ticker, BTC-USD, ...
    # Row 1: Date, , , ...
    # Data starts from row 2

    # Better to download again with simple format if possible, or just fix it here.
    # Let's fix it here.

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
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    df = preprocess_data("btc_data.csv")
    df.to_csv("btc_cleaned.csv", index=False)
    print("Data preprocessed and saved to btc_cleaned.csv")
    print(df.head())
