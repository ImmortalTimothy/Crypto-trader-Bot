import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, risk_per_trade=0.02, stop_loss_pct=0.05, daily_profit_target=0.02):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.daily_profit_target = daily_profit_target

        self.feature_cols = [c for c in df.columns if c not in ['Date', 'index']]
        self.n_features = len(self.feature_cols)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features + 1,), dtype=np.float32
        )

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.shares_held = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.portfolio_value = self.initial_balance
        self.history = []

        # Daily tracking
        self.day_start_portfolio_value = self.initial_balance
        self.steps_per_day = 24 # 1h data

        # Visualization data
        self.viz_data = [] # List of (step, OHLC, action)

    def _get_observation(self):
        obs = self.df.iloc[self.current_step][self.feature_cols].values
        current_price = self.df.iloc[self.current_step]['Close']
        obs = obs / (current_price if current_price != 0 else 1)
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
        self.day_start_portfolio_value = self.initial_balance
        self.viz_data = []

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']

        # Record viz data
        row = self.df.iloc[self.current_step]
        self.viz_data.append({
            'step': self.current_step,
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'action': action,
            'pos': self.position
        })
        if len(self.viz_data) > 100: self.viz_data.pop(0)

        # Stop Loss Check
        if self.position == 1 and current_price <= self.stop_loss_price:
            self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
            self.shares_held = 0
            self.position = 0
            self.entry_price = 0
            self.stop_loss_price = 0

        # Execute actions
        if action == 1 and self.position == 0: # Buy
            risk_amount = self.balance * self.risk_per_trade
            price_risk_per_share = current_price * self.stop_loss_pct
            desired_shares = risk_amount / price_risk_per_share
            cost = desired_shares * current_price * (1 + self.transaction_cost)

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
            self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
            self.shares_held = 0
            self.position = 0
            self.entry_price = 0
            self.stop_loss_price = 0

        self.current_step += 1

        if self.position == 1:
            self.portfolio_value = self.balance + (self.shares_held * self.df.iloc[self.current_step]['Close'])
        else:
            self.portfolio_value = self.balance

        # 1. Base log return reward
        reward = np.log(self.portfolio_value / self.history[-1]) if self.history[-1] > 0 else 0

        # 2. Penalty for being flat to promote trading
        if self.position == 0:
            reward -= 0.00001

        # 3. Daily performance penalty
        if self.current_step % self.steps_per_day == 0:
            daily_return = (self.portfolio_value / self.day_start_portfolio_value) - 1.0
            if daily_return < 0:
                # Penalty based on distance from target
                distance = self.daily_profit_target - daily_return
                reward -= distance * 0.01
            self.day_start_portfolio_value = self.portfolio_value

        self.history.append(self.portfolio_value)
        done = self.current_step >= len(self.df) - 1
        truncated = False

        observation = self._get_observation()
        info = {'portfolio_value': self.portfolio_value, 'position': self.position, 'viz_data': self.viz_data}

        return observation, reward, done, truncated, info
