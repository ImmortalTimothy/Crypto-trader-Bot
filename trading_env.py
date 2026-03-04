import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_cost=0.001):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Action space: 0 = Flat, 1 = Long
        self.action_space = spaces.Discrete(2)

        # Observation space: OHLCV + indicators + current position
        # Features: Close, High, Low, Open, Volume, SMA_10, SMA_30, RSI (8 features) + position (1 feature)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0 # 0 for Flat, 1 for Long
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.history = []

    def _get_observation(self):
        obs = self.df.iloc[self.current_step][['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_10', 'SMA_30', 'RSI']].values
        obs = np.append(obs, self.position)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.history = [self.portfolio_value]

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']

        # Execute actions
        if action == 1 and self.position == 0: # Buy
            # Buy as many shares as possible with current balance
            self.shares_held = (self.balance * (1 - self.transaction_cost)) / current_price
            self.balance = 0
            self.position = 1
        elif action == 0 and self.position == 1: # Sell
            # Sell all shares
            self.balance = self.shares_held * current_price * (1 - self.transaction_cost)
            self.shares_held = 0
            self.position = 0

        self.current_step += 1

        # Update portfolio value
        if self.position == 1:
            self.portfolio_value = self.shares_held * self.df.iloc[self.current_step]['Close']
        else:
            self.portfolio_value = self.balance

        reward = np.log(self.portfolio_value / self.history[-1]) if self.history[-1] > 0 else 0
        self.history.append(self.portfolio_value)

        done = self.current_step >= len(self.df) - 1
        truncated = False

        observation = self._get_observation()
        info = {'portfolio_value': self.portfolio_value}

        return observation, reward, done, truncated, info

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}, Position: {self.position}')
