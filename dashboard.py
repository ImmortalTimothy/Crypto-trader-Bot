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
import json
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(page_title="Crypto RL Bot Pro", layout="wide")

# Persistent Config to sync indicators between Training and Live tabs
CONFIG_FILE = "Data/model_config.json"

def save_config(indicators):
    os.makedirs("Data", exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump({"indicators": indicators}, f)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f).get("indicators", ["SMA_10", "SMA_30", "RSI"])
    return ["SMA_10", "SMA_30", "RSI"]

# Custom Callback for Visual Training
class StreamlitCallback(BaseCallback):
    def __init__(self, metrics_placeholder, progress_bar, viz_placeholder):
        super().__init__()
        self.metrics_placeholder = metrics_placeholder
        self.progress_bar = progress_bar
        self.viz_placeholder = viz_placeholder
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            progress = min(max(self.num_timesteps / self.locals['total_timesteps'], 0.0), 1.0)
            self.progress_bar.progress(progress)
            reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]) if len(self.model.ep_info_buffer) > 0 else 0.0
            self.rewards.append(reward)
            info = self.locals.get('infos', [{}])[0]
            viz_data = info.get('viz_data', [])

            with self.metrics_placeholder.container():
                c1, c2 = st.columns(2)
                c1.metric("Timesteps", self.num_timesteps)
                c1.metric("Avg Episode Reward", f"{reward:.4f}")
                fig_rew = go.Figure()
                fig_rew.add_trace(go.Scatter(y=self.rewards, name="Mean Reward"))
                fig_rew.update_layout(title="Reward Curve", height=250, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_rew, width="stretch")

            if viz_data:
                vdf = pd.DataFrame(viz_data)
                with self.viz_placeholder.container():
                    st.subheader("🕵️ Model Activity Review")
                    fig_cand = go.Figure(data=[go.Candlestick(x=vdf['step'],
                                    open=vdf['open'], high=vdf['high'],
                                    low=vdf['low'], close=vdf['close'], name="BTC")])
                    buys = vdf[vdf['action'] == 1]
                    if not buys.empty:
                        fig_cand.add_trace(go.Scatter(x=buys['step'], y=buys['low'] * 0.99, mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')))
                    fig_cand.update_layout(title="Live Training: Agent Actions (Rolling 100 steps)", height=450, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig_cand, width="stretch")
        return True

# --- Sidebar ---
st.sidebar.title("🤖 Bot Control Center")
run_mode = st.sidebar.selectbox("Run Mode", ["Simulation", "Paper Trading (Alpaca)"])
if run_mode == "Paper Trading (Alpaca)":
    api_key = st.sidebar.text_input("Alpaca API Key", type="password")
    api_secret = st.sidebar.text_input("Alpaca Secret Key", type="password")
else: api_key, api_secret = None, None

st.sidebar.subheader("Risk Management")
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0) / 100.0
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 5.0) / 100.0
daily_profit_target = st.sidebar.slider("Daily Profit Target (%)", 0.5, 5.0, 2.0) / 100.0
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
        st.subheader("Model Config")
        model_mode = st.radio("Mode", ["Train Fresh", "Resume Existing"])
        indicators = st.multiselect("Indicators", ["SMA_10", "SMA_30", "RSI", "MACD", "Bollinger"], default=["SMA_10", "SMA_30", "RSI"])
        timesteps = st.number_input("Timesteps", 1000, 1000000, 50000, step=1000)
    with col2:
        st.subheader("Hyperparameters")
        lr = st.number_input("Learning Rate", 1e-6, 1e-2, 3e-4, format="%.6f")
        gamma = st.slider("Gamma", 0.8, 0.999, 0.99)
        ent_coef = st.slider("Entropy", 0.0, 0.1, 0.01, format="%.3f")

    train_btn = st.button("🔥 Start Training Session", width="stretch")
    prog_bar = st.progress(0)
    c_metrics, c_viz = st.columns([1, 2])
    metrics_area = c_metrics.empty()
    viz_area = c_viz.empty()

    if train_btn:
        save_config(indicators) # Save for Live tab sync
        with st.spinner("Processing Data..."):
            df = preprocess_data(indicators=indicators)
        env = TradingEnv(df, daily_profit_target=daily_profit_target)
        model_path = "Logic/ppo_trading_model"
        if model_mode == "Resume Existing" and os.path.exists(model_path + ".zip"):
            model = PPO.load(model_path, env=env, learning_rate=lr, gamma=gamma, ent_coef=ent_coef)
        else:
            model = PPO("MlpPolicy", env, verbose=0, device="cpu", learning_rate=lr, gamma=gamma, ent_coef=ent_coef)
        st.toast("Training started!")
        callback = StreamlitCallback(metrics_area, prog_bar, viz_area)
        model.learn(total_timesteps=timesteps, callback=callback)
        model.save(model_path)
        st.success(f"Model saved to {model_path}.zip")
        st.cache_resource.clear()

# --- Live/Sim Tab ---
with tab_live:
    st.header("📈 Real-time Dashboard")
    @st.cache_resource
    def load_trained_model():
        path = "Logic/ppo_trading_model.zip"
        return PPO.load(path) if os.path.exists(path) else None

    model = load_trained_model()
    if model is None:
        st.warning("⚠️ No model found. Please train one.")
        st.stop()

    placeholder = st.empty()
    TX_COST = 0.001
    current_indicators = load_config() # Sync with training

    try:
        df = get_trading_data(indicators=current_indicators)
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

            last_row = df.iloc[-1]
            feature_cols = [c for c in df.columns if c not in ['Date', 'index']]
            obs = last_row[feature_cols].values / (current_price if current_price != 0 else 1)
            obs = np.append(obs, st.session_state.current_pos).astype(np.float32)

            action, _ = model.predict(obs, deterministic=True)
            dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
            probs = dist.distribution.probs.detach().numpy()[0]
            confidence = probs[action]

            if st.session_state.current_pos == 1 and current_price <= st.session_state.stop_loss_price:
                if run_mode == "Paper Trading (Alpaca)":
                    try: TradingClient(api_key, api_secret, paper=True).close_position('BTCUSD')
                    except: pass
                st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
                st.session_state.shares = 0
                st.session_state.current_pos = 0
                st.session_state.trade_history.append({'time': now, 'type': 'STOP LOSS', 'price': current_price})

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
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="BTC"))
            if st.session_state.current_pos == 1:
                fig.add_hline(y=st.session_state.stop_loss_price, line_dash="dash", line_color="red", annotation_text="Stop Loss")
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, width="stretch")
            st.subheader("Trade Log")
            if st.session_state.trade_history:
                st.table(pd.DataFrame(st.session_state.trade_history).tail(10))
    except Exception as e:
        st.error(f"Execution Error: {e}")

time.sleep(sim_speed)
st.rerun()
