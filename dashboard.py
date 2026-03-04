import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from Logic.utils import get_trading_data
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Set page config
st.set_page_config(page_title="Crypto RL Trader Control", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "Logic/ppo_trading_model.zip"
    if os.path.exists(model_path):
        return PPO.load(model_path)
    return None

model = load_model()

# --- Sidebar: Controls ---
st.sidebar.title("Trading Bot Controls")

# Mode Selection
run_mode = st.sidebar.selectbox("Mode", ["Simulation", "Paper Trading (Alpaca)"])
api_key = st.sidebar.text_input("Alpaca API Key", type="password") if run_mode == "Paper Trading (Alpaca)" else ""
api_secret = st.sidebar.text_input("Alpaca Secret Key", type="password") if run_mode == "Paper Trading (Alpaca)" else ""

# Risk Management Controls
st.sidebar.subheader("Risk Management")
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0) / 100.0
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 5.0) / 100.0
initial_balance = st.sidebar.number_input("Starting Balance ($)", 100, 100000, 10000)

# Bot Execution Controls
col_start, col_stop = st.sidebar.columns(2)
start_bot = col_start.button("🚀 Start Bot", use_container_width=True)
stop_bot = col_stop.button("🛑 Stop Bot", use_container_width=True)
kill_switch = st.sidebar.button("💀 KILL SWITCH (Sell All & Stop)", type="primary", use_container_width=True)

# --- Main App State ---
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

if start_bot:
    st.session_state.running = True
if stop_bot:
    st.session_state.running = False
if kill_switch:
    st.session_state.running = False

st.title("₿ Bitcoin RL Trading Dashboard")

if model is None:
    st.warning("No trained model found! Please run 'Logic/train.py' first.")

# --- Helper Functions ---
def get_action(df, current_pos):
    if model is None:
        return 0
    last_row = df.iloc[-1]
    obs = np.array([
        last_row['Close'], last_row['High'], last_row['Low'],
        last_row['Open'], last_row['Volume'], last_row['SMA_10'],
        last_row['SMA_30'], last_row['RSI'], current_pos
    ], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return action

# --- Main Logic ---
placeholder = st.empty()
TX_COST = 0.001

try:
    df = get_trading_data()
    current_price = float(df.iloc[-1]['Close'])
    now = datetime.now()

    if kill_switch and st.session_state.current_pos == 1:
        if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
            try:
                client = TradingClient(api_key, api_secret, paper=True)
                client.close_position('BTCUSD')
            except: pass
        st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
        st.session_state.shares = 0
        st.session_state.current_pos = 0
        st.session_state.trade_history.append({'time': now, 'type': 'KILL SWITCH EXIT', 'price': current_price})
        st.session_state.running = False

    if st.session_state.running and model is not None:
        if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
            try:
                client = TradingClient(api_key, api_secret, paper=True)
                pos = client.get_open_position('BTCUSD')
                st.session_state.current_pos = 1 if float(pos.qty) > 0 else 0
                st.session_state.shares = float(pos.qty)
                st.session_state.balance = float(client.get_account().cash)
            except:
                st.session_state.current_pos = 0

        if st.session_state.current_pos == 1 and current_price <= st.session_state.stop_loss_price:
            if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
                try: client.close_position('BTCUSD')
                except: pass
            st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
            st.session_state.shares = 0
            st.session_state.current_pos = 0
            st.session_state.trade_history.append({'time': now, 'type': 'STOP LOSS EXIT', 'price': current_price})

        action = get_action(df, st.session_state.current_pos)

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
                except Exception as e:
                    st.sidebar.error(f"Alpaca Error: {e}")

            st.session_state.shares = qty
            st.session_state.balance -= cost
            st.session_state.current_pos = 1
            st.session_state.stop_loss_price = current_price * (1 - stop_loss_pct)
            st.session_state.trade_history.append({'time': now, 'type': 'BUY', 'price': current_price})

        elif action == 0 and st.session_state.current_pos == 1:
            if run_mode == "Paper Trading (Alpaca)" and api_key and api_secret:
                try:
                    client = TradingClient(api_key, api_secret, paper=True)
                    client.close_position('BTCUSD')
                except: pass
            st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
            st.session_state.shares = 0
            st.session_state.current_pos = 0
            st.session_state.trade_history.append({'time': now, 'type': 'SELL', 'price': current_price})

    # UI Update
    current_val = st.session_state.balance + (st.session_state.shares * current_price)
    st.session_state.portfolio_history.append((now, current_val))
    if len(st.session_state.portfolio_history) > 1000:
        st.session_state.portfolio_history = st.session_state.portfolio_history[-1000:]

    hist_df = pd.DataFrame(st.session_state.portfolio_history, columns=['time', 'value'])
    p_all_time = current_val - initial_balance
    p_24h = current_val - (hist_df[hist_df['time'] >= (now - timedelta(days=1))]['value'].iloc[0] if not hist_df[hist_df['time'] >= (now - timedelta(days=1))].empty else current_val)
    p_1h = current_val - (hist_df[hist_df['time'] >= (now - timedelta(hours=1))]['value'].iloc[0] if not hist_df[hist_df['time'] >= (now - timedelta(hours=1))].empty else current_val)

    with placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", "🏃 Running" if st.session_state.running else "🛑 Stopped")
        col2.metric("Portfolio Value", f"${current_val:,.2f}")
        col3.metric("BTC Price", f"${current_price:,.2f}")
        col4.metric("Holding", "Bitcoin" if st.session_state.current_pos == 1 else "USD")

        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Profit (All-time)", f"${p_all_time:,.2f}", f"{(p_all_time/initial_balance):.2%}")
        m2.metric("Profit (Last 24h)", f"${p_24h:,.2f}")
        m3.metric("Profit (Last 1h)", f"${p_1h:,.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="BTC Price"))
        if st.session_state.current_pos == 1:
            fig.add_hline(y=st.session_state.stop_loss_price, line_dash="dash", line_color="red", annotation_text="Stop Loss")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trade Log")
        if st.session_state.trade_history:
            st.table(pd.DataFrame(st.session_state.trade_history).tail(10))

except Exception as e:
    st.error(f"Error in dashboard: {e}")

time.sleep(60)
st.rerun()
