from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import time
import os
from datetime import datetime
from Logic.utils import get_trading_data

# Alpaca API Credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'YOUR_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'YOUR_SECRET_KEY')

# Risk Management Settings
RISK_PER_TRADE = 0.02
STOP_LOSS_PCT = 0.05

def run_paper_trading():
    model_path = "Logic/ppo_trading_model"
    if not os.path.exists(model_path + ".zip"):
        print(f"⚠️ {model_path}.zip not found. Run train.py first.")
        return

    model = PPO.load(model_path)

    # Initialize Alpaca Trading Client
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    print("Starting Paper Trading Bot (Alpaca-py) with 2% Risk Rule...")

    while True:
        try:
            df = get_trading_data()
            last_row = df.iloc[-1]
            current_price = float(last_row['Close'])

            # Get current position and account info
            try:
                position = trading_client.get_open_position('BTCUSD')
                current_pos = 1 if float(position.qty) > 0 else 0
                avg_entry_price = float(position.avg_entry_price)
            except:
                current_pos = 0
                avg_entry_price = 0.0

            # Check for Stop Loss
            if current_pos == 1 and current_price <= avg_entry_price * (1 - STOP_LOSS_PCT):
                print(f"[{datetime.now()}] STOP LOSS EXIT at {current_price}")
                trading_client.close_position('BTCUSD')
                current_pos = 0

            obs = np.array([
                last_row['Close'], last_row['High'], last_row['Low'],
                last_row['Open'], last_row['Volume'], last_row['SMA_10'],
                last_row['SMA_30'], last_row['RSI'], current_pos
            ], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)

            if action == 1 and current_pos == 0:
                print(f"[{datetime.now()}] Action: BUY with 2% Risk Rule at {current_price}")
                account = trading_client.get_account()
                balance = float(account.cash)

                # Risk Amount = Balance * 2%
                risk_amount = balance * RISK_PER_TRADE
                price_risk_per_share = current_price * STOP_LOSS_PCT
                qty = risk_amount / price_risk_per_share

                # Ensure we have enough balance to cover the position size
                if qty * current_price > balance * 0.95:
                    qty = (balance * 0.95) / current_price

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
