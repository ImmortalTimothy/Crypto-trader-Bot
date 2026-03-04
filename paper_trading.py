from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import time
import os
from datetime import datetime
from utils import get_trading_data

# Alpaca API Credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'YOUR_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'YOUR_SECRET_KEY')

def run_paper_trading():
    model = PPO.load("ppo_trading_model")

    # Initialize Alpaca Trading Client
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    print("Starting Paper Trading Bot (Alpaca-py)...")

    while True:
        try:
            df = get_trading_data()
            last_row = df.iloc[-1]
            current_price = float(last_row['Close'])

            # Get current position
            try:
                position = trading_client.get_open_position('BTCUSD')
                current_pos = 1 if float(position.qty) > 0 else 0
            except:
                current_pos = 0

            obs = np.array([
                last_row['Close'], last_row['High'], last_row['Low'],
                last_row['Open'], last_row['Volume'], last_row['SMA_10'],
                last_row['SMA_30'], last_row['RSI'], current_pos
            ], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)

            if action == 1 and current_pos == 0:
                print(f"[{datetime.now()}] Action: BUY at {current_price}")
                account = trading_client.get_account()
                buying_power = float(account.cash) * 0.95
                qty = buying_power / current_price

                market_order_data = MarketOrderRequest(
                    symbol="BTCUSD",
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )
                trading_client.submit_order(order_data=market_order_data)

            elif action == 0 and current_pos == 1:
                print(f"[{datetime.now()}] Action: SELL at {current_price}")
                trading_client.close_position('BTCUSD')
            else:
                print(f"[{datetime.now()}] Action: HOLD (Position: {current_pos}) at {current_price}")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(60)

if __name__ == "__main__":
    if ALPACA_API_KEY == 'YOUR_API_KEY':
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
    else:
        run_paper_trading()
