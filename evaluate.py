import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from trading_env import TradingEnv

def calculate_metrics(portfolio_history, interval_hours=1):
    returns = pd.Series(portfolio_history).pct_change().dropna()

    # Sharpe Ratio (annualized)
    # 24 * 365 = 8760 trading hours in a crypto year
    annualization_factor = np.sqrt(8760 / interval_hours)
    sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor if returns.std() != 0 else 0

    # Max Drawdown
    cumulative_returns = pd.Series(portfolio_history)
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    total_return = (portfolio_history[-1] - portfolio_history[0]) / portfolio_history[0]

    return total_return, sharpe_ratio, max_drawdown

def evaluate():
    # Load data
    df = pd.read_csv("btc_cleaned.csv")

    # Split into train and test (80/20)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]

    # Initialize environment
    env = TradingEnv(test_df)

    # Load model
    model = PPO.load("ppo_trading_model")

    # Run evaluation
    obs, info = env.reset()
    done = False
    portfolio_history = [env.portfolio_value]

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        portfolio_history.append(info['portfolio_value'])

    # Calculate metrics
    total_return, sharpe, max_dd = calculate_metrics(portfolio_history)

    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd:.2%}")

    # Plot performance
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_history)
    plt.title("Portfolio Value Over Time (Test Set)")
    plt.xlabel("Hours")
    plt.ylabel("Portfolio Value (USD)")
    plt.grid(True)
    plt.savefig("portfolio_performance.png")
    print("Performance plot saved as portfolio_performance.png")

if __name__ == "__main__":
    evaluate()
