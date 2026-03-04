import os
import sys

# Crucially, add project root to sys.path before internal Logic imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from stable_baselines3 import PPO
from Logic.trading_env import TradingEnv
from Logic.preprocess_data import preprocess_data

def train():
    # Load or create data using cross-platform paths
    data_path = os.path.join("Data", "btc_cleaned.csv")
    if not os.path.exists(data_path):
        print(f"⚠️ {data_path} not found. Running preprocessing...")
        df = preprocess_data()
    else:
        df = pd.read_csv(data_path)

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
    model_path = os.path.join("Logic", "ppo_trading_model")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    train()
