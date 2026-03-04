import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from Logic.utils import get_trading_data
from Logic.preprocess_data import preprocess_data
from Logic.trading_env import TradingEnv
import os
import sys
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(page_title="Crypto RL Bot v2", layout="wide")

# Custom Callback for Visual Training
class StreamlitCallback(BaseCallback):
    def __init__(self, metrics_placeholder, progress_bar):
        super().__init__()
        self.metrics_placeholder = metrics_placeholder
        self.progress_bar = progress_bar
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            # FIX: Clip progress to [0.0, 1.0]
            progress = min(max(self.num_timesteps / self.locals['total_timesteps'], 0.0), 1.0)
            self.progress_bar.progress(progress)

            # Extract mean reward from rollout buffer if available
            if len(self.model.ep_info_buffer) > 0:
                reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            else:
                reward = 0.0

            self.rewards.append(reward)

            with self.metrics_placeholder.container():
                c1, c2 = st.columns(2)
                c1.metric("Timesteps", self.num_timesteps)
                c1.metric("Avg Episode Reward", f"{reward:.4f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=self.rewards, name="Mean Reward"))
                fig.update_layout(title="Training Reward Curve", height=300, margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig, width="stretch")
        return True

# --- Sidebar: Bot Control Center ---
st.sidebar.title("🤖 Bot Control Center")

run_mode = st.sidebar.selectbox("Run Mode", ["Simulation", "Paper Trading (Alpaca)"])

if run_mode == "Paper Trading (Alpaca)":
    api_key = st.sidebar.text_input("Alpaca API Key", type="password")
    api_secret = st.sidebar.text_input("Alpaca Secret Key", type="password")
else:
    api_key, api_secret = None, None

st.sidebar.subheader("Risk Management")
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0) / 100.0
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 5.0) / 100.0
initial_balance = st.sidebar.number_input("Starting Balance ($)", 100, 100000, 10000)

st.sidebar.subheader("Execution")
sim_speed = st.sidebar.select_slider("Refresh Rate (sec)", options=[1, 5, 10, 30, 60], value=60)

col_start, col_stop = st.sidebar.columns(2)
start_bot = col_start.button("🚀 START", width="stretch")
stop_bot = col_stop.button("🛑 STOP", width="stretch")
kill_switch = st.sidebar.button("💀 KILL SWITCH", type="primary", width="stretch")

# --- Tabs ---
tab_live, tab_train = st.tabs(["📊 Live Trading / Sim", "🧠 Training Center"])

# --- Session State ---
if 'running' not in st.session_state: st.session_state.running = False
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'current_pos' not in st.session_state: st.session_state.current_pos = 0
if 'balance' not in st.session_state: st.session_state.balance = float(initial_balance)
if 'shares' not in st.session_state: st.session_state.shares = 0.0
if 'portfolio_history' not in st.session_state: st.session_state.portfolio_history = []
if 'stop_loss_price' not in st.session_state: st.session_state.stop_loss_price = 0.0

if start_bot: st.session_state.running = True
if stop_bot: st.session_state.running = False

# --- Training Tab ---
with tab_train:
    st.header("🧠 Agent Training")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Configuration")
        model_mode = st.radio("Training Mode", ["Train Fresh", "Resume Existing"])
        indicators = st.multiselect("Technical Indicators", ["SMA_10", "SMA_30", "RSI", "MACD", "Bollinger"], default=["SMA_10", "SMA_30", "RSI"])
        timesteps = st.number_input("Total Timesteps", 1000, 1000000, 50000, step=1000)

    with col2:
        st.subheader("Hyperparameters")
        lr = st.number_input("Learning Rate", 1e-6, 1e-2, 3e-4, format="%.6f")
        gamma = st.slider("Gamma (Discount)", 0.8, 0.999, 0.99)
        ent_coef = st.slider("Entropy Coef", 0.0, 0.1, 0.01, format="%.3f")

    train_btn = st.button("🔥 Start Training Session", width="stretch")

    metrics_area = st.empty()
    prog_bar = st.progress(0)

    if train_btn:
        with st.spinner("Processing Data..."):
            df = preprocess_data(indicators=indicators)

        env = TradingEnv(df)
        model_path = "Logic/ppo_trading_model"

        if model_mode == "Resume Existing" and os.path.exists(model_path + ".zip"):
            st.info("Loading existing model...")
            model = PPO.load(model_path, env=env, learning_rate=lr, gamma=gamma, ent_coef=ent_coef)
        else:
            st.info("Initializing new MlpPolicy model...")
            model = PPO("MlpPolicy", env, verbose=0, device="cpu", learning_rate=lr, gamma=gamma, ent_coef=ent_coef)

        st.toast("Training started!")
        callback = StreamlitCallback(metrics_area, prog_bar)
        model.learn(total_timesteps=timesteps, callback=callback)

        model.save(model_path)
        st.success(f"Model saved successfully to {model_path}.zip")
        st.cache_resource.clear()

# --- Live/Sim Tab ---
with tab_live:
    st.header("📈 Real-time Dashboard")

    @st.cache_resource
    def load_trained_model():
        path = "Logic/ppo_trading_model.zip"
        if os.path.exists(path):
            return PPO.load(path)
        return None

    model = load_trained_model()
    if model is None:
        st.warning("⚠️ No model found. Please train one in the 'Training Center' tab.")
        st.stop()

    placeholder = st.empty()
    TX_COST = 0.001

    try:
        # Fetch current feature set from model's env expectation or just use selected
        # (Assuming model was trained with the same indicators selected in Training tab)
        # For robustness, we'd store indicator config with the model.
        # Here we just use default/current for demo.
        df = get_trading_data(indicators=["SMA_10", "SMA_30", "RSI", "MACD", "Bollinger"])
        current_price = float(df.iloc[-1]['Close'])
        now = datetime.now()

        if kill_switch:
            if st.session_state.current_pos == 1:
                if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
                    try: TradingClient(api_key, api_secret, paper=True).close_position('BTCUSD')
                    except: pass
                st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
                st.session_state.shares = 0
                st.session_state.current_pos = 0
                st.session_state.trade_history.append({'time': now, 'type': 'KILL SWITCH', 'price': current_price})
            st.session_state.running = False

        if st.session_state.running:
            if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
                try:
                    client = TradingClient(api_key, api_secret, paper=True)
                    pos = client.get_open_position('BTCUSD')
                    st.session_state.current_pos = 1 if float(pos.qty) > 0 else 0
                    st.session_state.shares = float(pos.qty)
                    st.session_state.balance = float(client.get_account().cash)
                except: st.session_state.current_pos = 0

            # Predict
            last_row = df.iloc[-1]
            # Match features to TradingEnv logic
            feature_cols = [c for c in df.columns if c not in ['Date', 'index']]
            obs = last_row[feature_cols].values / current_price
            obs = np.append(obs, st.session_state.current_pos).astype(np.float32)

            action, _ = model.predict(obs, deterministic=True)
            dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
            probs = dist.distribution.probs.detach().numpy()[0]
            confidence = probs[action]

            # Stop Loss
            if st.session_state.current_pos == 1 and current_price <= st.session_state.stop_loss_price:
                if run_mode == "Paper Trading (Alpaca)":
                    try: TradingClient(api_key, api_secret, paper=True).close_position('BTCUSD')
                    except: pass
                st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
                st.session_state.shares = 0
                st.session_state.current_pos = 0
                st.session_state.trade_history.append({'time': now, 'type': 'STOP LOSS', 'price': current_price})

            # Trade execution
            if action == 1 and st.session_state.current_pos == 0:
                risk_amt = st.session_state.balance * risk_per_trade
                price_risk = current_price * stop_loss_pct
                qty = risk_amt / price_risk
                cost = qty * current_price * (1 + TX_COST)
                if cost > st.session_state.balance:
                    qty = (st.session_state.balance * (1 - TX_COST)) / current_price
                    cost = st.session_state.balance

                if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
                    try:
                        req = MarketOrderRequest(symbol="BTCUSD", qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                        TradingClient(api_key, api_secret, paper=True).submit_order(req)
                    except: pass

                st.session_state.shares = qty
                st.session_state.balance -= cost
                st.session_state.current_pos = 1
                st.session_state.stop_loss_price = current_price * (1 - stop_loss_pct)
                st.session_state.trade_history.append({'time': now, 'type': 'BUY', 'price': current_price})

            elif action == 0 and st.session_state.current_pos == 1:
                if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
                    try: TradingClient(api_key, api_secret, paper=True).close_position('BTCUSD')
                    except: pass
                st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
                st.session_state.shares = 0
                st.session_state.current_pos = 0
                st.session_state.trade_history.append({'time': now, 'type': 'SELL', 'price': current_price})

        # Dashboard View
        current_val = st.session_state.balance + (st.session_state.shares * current_price)
        st.session_state.portfolio_history.append((now, current_val))
        hist_df = pd.DataFrame(st.session_state.portfolio_history, columns=['time', 'value'])
        p_all_time = current_val - initial_balance
        p_24h = current_val - (hist_df[hist_df['time'] >= (now - timedelta(days=1))]['value'].iloc[0] if not hist_df[hist_df['time'] >= (now - timedelta(days=1))].empty else current_val)
        p_1h = current_val - (hist_df[hist_df['time'] >= (now - timedelta(hours=1))]['value'].iloc[0] if not hist_df[hist_df['time'] >= (now - timedelta(hours=1))].empty else current_val)

        with placeholder.container():
            if st.session_state.running:
                st.subheader("🤖 Model Insights")
                cc1, cc2 = st.columns(2)
                cc1.info(f"**Planned Action:** {'LONG' if action==1 else 'FLAT'}")
                cc2.info(f"**Model Confidence:** {confidence:.1%}")
                with st.expander("Analysis: Input Feature Vector (Scaled)"):
                    st.write(obs)

            st.write("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Status", "🏃 Running" if st.session_state.running else "🛑 Stopped")
            m2.metric("Portfolio Value", f"${current_val:,.2f}")
            m3.metric("BTC Price", f"${current_price:,.2f}")
            m4.metric("Holding", "Bitcoin" if st.session_state.current_pos == 1 else "USD")

            m1b, m2b, m3b = st.columns(3)
            m1b.metric("Profit (All-time)", f"${p_all_time:,.2f}", f"{(p_all_time/initial_balance):.2%}")
            m2b.metric("Profit (Last 24h)", f"${p_24h:,.2f}")
            m3b.metric("Profit (Last 1h)", f"${p_1h:,.2f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="BTC Price"))
            if st.session_state.current_pos == 1:
                fig.add_hline(y=st.session_state.stop_loss_price, line_dash="dash", line_color="red", annotation_text="Stop Loss")
            fig.update_layout(height=400, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, width="stretch")

            st.subheader("Trade Log")
            if st.session_state.trade_history:
                st.table(pd.DataFrame(st.session_state.trade_history).tail(10))

    except Exception as e:
        st.error(f"Execution Error: {e}")

time.sleep(sim_speed)
st.rerun()
