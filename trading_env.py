import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, risk_per_trade=0.02, stop_loss_pct=0.05):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct

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
        self.entry_price = 0
        self.stop_loss_price = 0
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
        self.entry_price = 0
        self.stop_loss_price = 0
        self.portfolio_value = self.initial_balance
        self.history = [self.portfolio_value]

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']

        # Check for Stop Loss if in position
        if self.position == 1 and current_price <= self.stop_loss_price:
            # Automatic Exit
            self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
            self.shares_held = 0
            self.position = 0
            self.entry_price = 0
            self.stop_loss_price = 0

        # Execute actions if not stopped out
        if action == 1 and self.position == 0: # Buy
            # 2% Risk Rule Position Sizing:
            # Risk Amount = Balance * 2%
            # Position Size = Risk Amount / (Entry Price - Stop Loss Price)
            # Stop Loss Price = Entry Price * (1 - stop_loss_pct)
            risk_amount = self.balance * self.risk_per_trade
            price_risk_per_share = current_price * self.stop_loss_pct

            desired_shares = risk_amount / price_risk_per_share
            cost = desired_shares * current_price * (1 + self.transaction_cost)

            # Ensure we don't exceed balance
            if cost > self.balance:
                self.shares_held = (self.balance * (1 - self.transaction_cost)) / current_price
                self.balance = 0
            else:
                self.shares_held = desired_shares
                self.balance -= cost

            self.position = 1
            self.entry_price = current_price
            self.stop_loss_price = current_price * (1 - self.stop_loss_pct)

        elif action == 0 and self.position == 1: # Sell
            # Manual Sell all shares
            self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
            self.shares_held = 0
            self.position = 0
            self.entry_price = 0
            self.stop_loss_price = 0

        self.current_step += 1

        # Update portfolio value
        if self.position == 1:
            self.portfolio_value = self.balance + (self.shares_held * self.df.iloc[self.current_step]['Close'])
        else:
            self.portfolio_value = self.balance

        reward = np.log(self.portfolio_value / self.history[-1]) if self.history[-1] > 0 else 0
        self.history.append(self.portfolio_value)

        done = self.current_step >= len(self.df) - 1
        truncated = False

        observation = self._get_observation()
        info = {'portfolio_value': self.portfolio_value, 'position': self.position}

        return observation, reward, done, truncated, info

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}, Position: {self.position}')
