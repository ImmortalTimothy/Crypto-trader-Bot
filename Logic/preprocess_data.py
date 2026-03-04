import os
import sys

# Ensure project root is in path before internal Logic imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from Logic.utils import download_historical_data, compute_rsi

def preprocess_data(csv_path=os.path.join("Data", "btc_data.csv")):
    if not os.path.exists(csv_path):
        download_historical_data()

    data = pd.read_csv(csv_path)
    if 'Datetime' in data.columns:
        data = data.rename(columns={'Datetime': 'Date'})

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)

    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['RSI'] = compute_rsi(data['Close'], window=14)
    data = data.dropna().reset_index(drop=True)

    os.makedirs("Data", exist_ok=True)
    output_path = os.path.join("Data", "btc_cleaned.csv")
    data.to_csv(output_path, index=False)
    print(f"Data preprocessed and saved to {output_path}")
    return data

if __name__ == "__main__":
    preprocess_data()
