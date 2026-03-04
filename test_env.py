from Logic.trading_env import TradingEnv
from Logic.preprocess_data import preprocess_data
import os
import pandas as pd

def test_env():
    data_path = "Data/btc_cleaned.csv"
    if not os.path.exists(data_path):
        df = preprocess_data()
    else:
        df = pd.read_csv(data_path)

    env = TradingEnv(df)
    obs, info = env.reset()
    print(f"Initial observation: {obs}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Portfolio={info['portfolio_value']:.2f}, Done={done}")
        if done:
            break

if __name__ == "__main__":
    test_env()
