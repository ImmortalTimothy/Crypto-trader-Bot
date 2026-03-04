import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_data(ticker="BTC-USD"):
    # yfinance 1h data only available for last 730 days
    end = datetime.now()
    start = end - timedelta(days=729)

    df = yf.download(ticker, start=start, end=end, interval="1h")
    # Flatten if multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv("btc_data.csv")
    print(f"Downloaded {len(df)} rows of data for {ticker} with interval 1h")

if __name__ == "__main__":
    download_data()
