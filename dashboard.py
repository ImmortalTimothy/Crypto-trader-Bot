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

# Set page config
st.set_page_config(page_title="Crypto RL Bot", layout="wide")

# Custom Callback for Visual Training
class StreamlitCallback(BaseCallback):
    def __init__(self, metrics_placeholder, progress_bar):
        super().__init__()
        self.metrics_placeholder = metrics_placeholder
        self.progress_bar = progress_bar
        self.rewards = []
        self.losses = []

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            progress = self.num_timesteps / self.locals['total_timesteps']
            self.progress_bar.progress(progress)

            # Get latest reward
            reward = np.mean(self.locals.get('rewards', [0]))
            self.rewards.append(reward)

            with self.metrics_placeholder.container():
                c1, c2 = st.columns(2)
                c1.metric("Timesteps", self.num_timesteps)
                c1.metric("Mean Reward (100 steps)", f"{reward:.4f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=self.rewards, name="Mean Reward"))
                fig.update_layout(title="Training Progress", height=300)
                st.plotly_chart(fig, width="stretch")
        return True

# --- Sidebar: Persistent Controls ---
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
sim_speed = st.sidebar.select_slider("Sim Refresh (sec)", options=[1, 5, 10, 30, 60], value=60)

col_start, col_stop = st.sidebar.columns(2)
start_bot = col_start.button("🚀 START", width="stretch")
stop_bot = col_stop.button("🛑 STOP", width="stretch")
kill_switch = st.sidebar.button("💀 KILL SWITCH", type="primary", width="stretch")

# --- Tabs ---
tab_live, tab_train = st.tabs(["📊 Live Trading / Sim", "🧠 Training Center"])

# --- Session State ---
if 'running' not in st.session_state:
    st.session_state.running = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'current_pos' not in st.session_state:
    st.session_state.current_pos = 0
if 'balance' not in st.session_state:
    st.session_state.balance = float(initial_balance)
if 'shares' not in st.session_state:
    st.session_state.shares = 0.0
if 'portfolio_history' not in st.session_state:
    st.session_state.portfolio_history = []
if 'stop_loss_price' not in st.session_state:
    st.session_state.stop_loss_price = 0.0

if start_bot: st.session_state.running = True
if stop_bot: st.session_state.running = False

# --- Training Tab ---
with tab_train:
    st.header("🧠 Agent Training")
    timesteps = st.number_input("Timesteps", 1000, 500000, 50000)
    train_btn = st.button("Train New Model")

    metrics_area = st.empty()
    prog_bar = st.progress(0)

    if train_btn:
        with st.spinner("Preparing Data..."):
            df = preprocess_data()

        env = TradingEnv(df)
        model = PPO("MlpPolicy", env, verbose=0, device="cpu")

        st.info("Training started! Watch metrics update below.")
        callback = StreamlitCallback(metrics_area, prog_bar)
        model.learn(total_timesteps=timesteps, callback=callback)

        model.save("Logic/ppo_trading_model")
        st.success("Training complete! Model saved to Logic/ppo_trading_model.zip")
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
        df = get_trading_data()
        current_price = float(df.iloc[-1]['Close'])
        now = datetime.now()

        # Kill Switch
        if kill_switch and st.session_state.current_pos == 1:
            if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
                try:
                    client = TradingClient(api_key, api_secret, paper=True)
                    client.close_position('BTCUSD')
                except: pass
            st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
            st.session_state.shares = 0
            st.session_state.current_pos = 0
            st.session_state.trade_history.append({'time': now, 'type': 'KILL SWITCH', 'price': current_price})
            st.session_state.running = False

        if st.session_state.running:
            # Live paper sync
            if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
                try:
                    client = TradingClient(api_key, api_secret, paper=True)
                    pos = client.get_open_position('BTCUSD')
                    st.session_state.current_pos = 1 if float(pos.qty) > 0 else 0
                    st.session_state.shares = float(pos.qty)
                    st.session_state.balance = float(client.get_account().cash)
                except: st.session_state.current_pos = 0

            # Logic
            last_row = df.iloc[-1]
            obs = np.array([
                last_row['Close'], last_row['High'], last_row['Low'],
                last_row['Open'], last_row['Volume'], last_row['SMA_10'],
                last_row['SMA_30'], last_row['RSI'], st.session_state.current_pos
            ], dtype=np.float32)

            action, _states = model.predict(obs, deterministic=True)

            # Predict probabilities for planning insights
            # Note: SB3 PPO get_distribution requires some internal access
            dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
            probs = dist.distribution.probs.detach().numpy()[0]
            confidence = probs[action]

            # STOP LOSS
            if st.session_state.current_pos == 1 and current_price <= st.session_state.stop_loss_price:
                if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
                    try: client.close_position('BTCUSD')
                    except: pass
                st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
                st.session_state.shares = 0
                st.session_state.current_pos = 0
                st.session_state.trade_history.append({'time': now, 'type': 'STOP LOSS', 'price': current_price})

            # TRADES
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
                        client = TradingClient(api_key, api_secret, paper=True)
                        req = MarketOrderRequest(symbol="BTCUSD", qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                        client.submit_order(req)
                    except Exception as e: st.sidebar.error(f"Alpaca: {e}")

                st.session_state.shares = qty
                st.session_state.balance -= cost
                st.session_state.current_pos = 1
                st.session_state.stop_loss_price = current_price * (1 - stop_loss_pct)
                st.session_state.trade_history.append({'time': now, 'type': 'BUY', 'price': current_price})

            elif action == 0 and st.session_state.current_pos == 1:
                if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
                    try: client.close_position('BTCUSD')
                    except: pass
                st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
                st.session_state.shares = 0
                st.session_state.current_pos = 0
                st.session_state.trade_history.append({'time': now, 'type': 'SELL', 'price': current_price})

        # --- UI DISPLAY ---
        current_val = st.session_state.balance + (st.session_state.shares * current_price)
        st.session_state.portfolio_history.append((now, current_val))
        hist_df = pd.DataFrame(st.session_state.portfolio_history, columns=['time', 'value'])
        p_all_time = current_val - initial_balance
        p_24h = current_val - (hist_df[hist_df['time'] >= (now - timedelta(days=1))]['value'].iloc[0] if not hist_df[hist_df['time'] >= (now - timedelta(days=1))].empty else current_val)
        p_1h = current_val - (hist_df[hist_df['time'] >= (now - timedelta(hours=1))]['value'].iloc[0] if not hist_df[hist_df['time'] >= (now - timedelta(hours=1))].empty else current_val)

        with placeholder.container():
            # Planning Insights
            if st.session_state.running:
                st.subheader("🤖 Model Insights")
                cc1, cc2 = st.columns(2)
                cc1.info(f"**Current Action:** {'LONG' if action==1 else 'FLAT'}")
                cc2.info(f"**Confidence:** {confidence:.1%}")

                with st.expander("Show Features (Input Vector)"):
                    st.write(obs)

            # Metrics
            st.write("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Status", "🏃 Running" if st.session_state.running else "🛑 Stopped")
            m2.metric("Portfolio Value", f"${current_val:,.2f}")
            m3.metric("BTC Price", f"${current_price:,.2f}")
            m4.metric("Holding", "Bitcoin" if st.session_state.current_pos == 1 else "USD")

            m1b, m2b, m3b = st.columns(3)
            m1b.metric("Profit (All-time)", f"${p_all_time:,.2f}", f"{(p_all_time/initial_balance):.2%}")
            m2b.metric("Profit (24h)", f"${p_24h:,.2f}")
            m3b.metric("Profit (1h)", f"${p_1h:,.2f}")

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="BTC Price"))
            if st.session_state.current_pos == 1:
                fig.add_hline(y=st.session_state.stop_loss_price, line_dash="dash", line_color="red", annotation_text="Stop Loss")
            st.plotly_chart(fig, width="stretch")

            st.subheader("Trade Log")
            if st.session_state.trade_history:
                st.table(pd.DataFrame(st.session_state.trade_history).tail(10))

    except Exception as e:
        st.error(f"Loop Error: {e}")

time.sleep(sim_speed)
st.rerun()
