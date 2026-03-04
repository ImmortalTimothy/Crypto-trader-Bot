import yfinance as yf
import pandas as pd

def download_data(ticker="BTC-USD", start="2020-01-01", end="2023-12-31"):
    df = yf.download(ticker, start=start, end=end)
    df.to_csv("btc_data.csv")
    print(f"Downloaded {len(df)} rows of data for {ticker}")

if __name__ == "__main__":
    download_data()
