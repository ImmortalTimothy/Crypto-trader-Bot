import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, risk_per_trade=0.02, stop_loss_pct=0.05, daily_profit_target=0.02, max_steps=1000):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.daily_profit_target = daily_profit_target
        self.max_steps = max_steps

        self.feature_cols = [c for c in df.columns if c not in ['Date', 'index', 'Datetime']]
        self.n_features = len(self.feature_cols)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features + 1,), dtype=np.float32
        )

        self.current_step = 0
        self.start_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.shares_held = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.portfolio_value = self.initial_balance
        self.history = []

        self.day_start_portfolio_value = self.initial_balance
        self.steps_per_day = 24
        self.viz_data = []

    def _get_observation(self):
        obs = self.df.iloc[self.current_step][self.feature_cols].values
        current_price = self.df.iloc[self.current_step]['Close']
        obs = obs / (current_price if current_price != 0 else 1)
        obs = np.append(obs, self.position)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random start point to help learning
        # Ensure we have at least max_steps remaining
        max_start = len(self.df) - self.max_steps - 1
        if max_start > 0:
            self.start_step = np.random.randint(0, max_start)
        else:
            self.start_step = 0

        self.current_step = self.start_step
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

        # Stop Loss
        if self.position == 1 and current_price <= self.stop_loss_price:
            self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
            self.shares_held = 0
            self.position = 0
            self.entry_price = 0
            self.stop_loss_price = 0

        # Actions
        if action == 1 and self.position == 0: # Buy
            risk_amount = self.balance * self.risk_per_trade
            price_risk_per_share = current_price * self.stop_loss_pct
            desired_shares = risk_amount / (price_risk_per_share if price_risk_per_share != 0 else 1)
            cost = desired_shares * current_price * (1 + self.transaction_cost)

            if cost > self.balance:
                self.shares_held = (self.balance * (1 - self.transaction_cost)) / (current_price if current_price != 0 else 1)
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

        # Scaled log return reward
        # Multiplying by 100 helps PPO see small percentage changes
        reward = np.log(self.portfolio_value / self.history[-1]) * 100.0 if self.history[-1] > 0 else 0

        if self.position == 0:
            reward -= 0.001 # Promoting trading

        if (self.current_step - self.start_step) % self.steps_per_day == 0:
            daily_return = (self.portfolio_value / self.day_start_portfolio_value) - 1.0
            if daily_return < 0:
                distance = self.daily_profit_target - daily_return
                reward -= distance * 1.0 # Significant penalty for failure
            self.day_start_portfolio_value = self.portfolio_value

        self.history.append(self.portfolio_value)

        # End episode after max_steps or end of data
        done = (self.current_step >= len(self.df) - 1) or (self.current_step - self.start_step >= self.max_steps)
        truncated = False

        observation = self._get_observation()
        info = {'portfolio_value': self.portfolio_value, 'position': self.position, 'viz_data': list(self.viz_data)}

        return observation, reward, done, truncated, info
