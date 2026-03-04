import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv
import os

def train():
    # Load data
    df = pd.read_csv("btc_cleaned.csv")

    # Split into train and test (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]

    # Initialize environment
    env = TradingEnv(train_df)

    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=50000)

    # Save the model
    model_path = "ppo_trading_model"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    train()
