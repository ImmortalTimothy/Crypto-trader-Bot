import pandas as pd
from utils import download_historical_data, compute_rsi

def preprocess_data(csv_path):
    # Read the data back
    data = pd.read_csv(csv_path)
    # The index column is named 'Datetime' from yfinance
    if 'Datetime' in data.columns:
        data = data.rename(columns={'Datetime': 'Date'})

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)

    # Simple Technical Indicators
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['RSI'] = compute_rsi(data['Close'], window=14)

    data = data.dropna().reset_index(drop=True)
    return data

if __name__ == "__main__":
    download_historical_data()
    df = preprocess_data("btc_data.csv")
    df.to_csv("btc_cleaned.csv", index=False)
    print("Data preprocessed and saved to btc_cleaned.csv")
    print(df.head())
