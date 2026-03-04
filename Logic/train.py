import pandas as pd
from stable_baselines3 import PPO
from Logic.trading_env import TradingEnv
from Logic.preprocess_data import preprocess_data
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train():
    # Load or create data
    data_path = "Data/btc_cleaned.csv"
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
    model_path = "Logic/ppo_trading_model"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    train()
