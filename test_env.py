import pandas as pd
from trading_env import TradingEnv

def test_env():
    df = pd.read_csv("btc_cleaned.csv")
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
