import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils import get_trading_data

# Set page config
st.set_page_config(page_title="Crypto RL Trader", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return PPO.load("ppo_trading_model")

model = load_model()

# Risk Management Settings
RISK_PER_TRADE = 0.02
STOP_LOSS_PCT = 0.05
TX_COST = 0.001

def get_action(df, current_pos):
    last_row = df.iloc[-1]
    obs = np.array([
        last_row['Close'], last_row['High'], last_row['Low'],
        last_row['Open'], last_row['Volume'], last_row['SMA_10'],
        last_row['SMA_30'], last_row['RSI'], current_pos
    ], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return action

st.title("₿ Bitcoin RL Trading Dashboard (2% Risk Rule)")

# Initialize session state
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'current_pos' not in st.session_state:
    st.session_state.current_pos = 0
if 'balance' not in st.session_state:
    st.session_state.balance = 10000.0
if 'shares' not in st.session_state:
    st.session_state.shares = 0.0
if 'portfolio_history' not in st.session_state:
    st.session_state.portfolio_history = []
if 'stop_loss_price' not in st.session_state:
    st.session_state.stop_loss_price = 0.0

placeholder = st.empty()

# Main loop for real-time updates
while True:
    try:
        df = get_trading_data()
        if df.empty:
            st.error("Failed to fetch data. Retrying in 60s...")
            time.sleep(60)
            continue

        current_price = float(df.iloc[-1]['Close'])
        now = datetime.now()

        # Check for Stop Loss
        if st.session_state.current_pos == 1 and current_price <= st.session_state.stop_loss_price:
            st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
            st.session_state.shares = 0
            st.session_state.current_pos = 0
            st.session_state.stop_loss_price = 0.0
            st.session_state.trade_history.append({'time': now, 'type': 'STOP LOSS EXIT', 'price': current_price})

        # Agent decides action
        action = get_action(df, st.session_state.current_pos)

        # Execute action logic
        if action == 1 and st.session_state.current_pos == 0:
            # Buy with 2% risk rule
            risk_amount = st.session_state.balance * RISK_PER_TRADE
            price_risk_per_share = current_price * STOP_LOSS_PCT
            desired_shares = risk_amount / price_risk_per_share
            cost = desired_shares * current_price * (1 + TX_COST)

            if cost > st.session_state.balance:
                st.session_state.shares = (st.session_state.balance * (1 - TX_COST)) / current_price
                st.session_state.balance = 0
            else:
                st.session_state.shares = desired_shares
                st.session_state.balance -= cost

            st.session_state.current_pos = 1
            st.session_state.stop_loss_price = current_price * (1 - STOP_LOSS_PCT)
            st.session_state.trade_history.append({'time': now, 'type': 'BUY', 'price': current_price})

        elif action == 0 and st.session_state.current_pos == 1:
            # Manual Sell
            st.session_state.balance += st.session_state.shares * current_price * (1 - TX_COST)
            st.session_state.shares = 0
            st.session_state.current_pos = 0
            st.session_state.stop_loss_price = 0.0
            st.session_state.trade_history.append({'time': now, 'type': 'SELL', 'price': current_price})

        # Update portfolio value
        current_portfolio_value = st.session_state.balance + (st.session_state.shares * current_price)
        st.session_state.portfolio_history.append((now, current_portfolio_value))

        # Calculate metrics
        initial_value = 10000.0
        profit_all_time = current_portfolio_value - initial_value
        history_df = pd.DataFrame(st.session_state.portfolio_history, columns=['time', 'value'])

        # Daily profit
        val_24h_ago = history_df[history_df['time'] >= (now - timedelta(days=1))]['value'].iloc[0] if not history_df[history_df['time'] >= (now - timedelta(days=1))].empty else current_portfolio_value
        profit_today = current_portfolio_value - val_24h_ago

        # Hourly profit
        val_1h_ago = history_df[history_df['time'] >= (now - timedelta(hours=1))]['value'].iloc[0] if not history_df[history_df['time'] >= (now - timedelta(hours=1))].empty else current_portfolio_value
        profit_last_hour = current_portfolio_value - val_1h_ago

        with placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("BTC Price", f"${current_price:,.2f}")
            m2.metric("Portfolio Value", f"${current_portfolio_value:,.2f}")
            m3.metric("Holding", "Bitcoin" if st.session_state.current_pos == 1 else "USD")
            m4.metric("Profit All-time", f"${profit_all_time:,.2f}", f"{(profit_all_time/initial_value):.2%}")

            m1b, m2b, m3b, m4b, m5b = st.columns(5)
            m1b.metric("Profit (24h)", f"${profit_today:,.2f}")
            m2b.metric("Profit (1h)", f"${profit_last_hour:,.2f}")
            m3b.metric("Risk per Trade", f"{RISK_PER_TRADE:.1%}")
            m4b.metric("Stop Loss", f"{STOP_LOSS_PCT:.1%}")
            m5b.metric("SL Price", f"${st.session_state.stop_loss_price:,.2f}" if st.session_state.current_pos == 1 else "N/A")

            st.subheader("BTC-USD Live Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="BTC Price"))
            if st.session_state.current_pos == 1:
                fig.add_hline(y=st.session_state.stop_loss_price, line_dash="dash", line_color="red", annotation_text="Stop Loss")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Recent Activity")
            if st.session_state.trade_history:
                st.table(pd.DataFrame(st.session_state.trade_history).tail(10))
            else:
                st.info("Waiting for first trade signal...")

            st.caption(f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"Error: {e}")

    time.sleep(60)
