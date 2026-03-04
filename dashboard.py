import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Crypto RL Trader", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return PPO.load("ppo_trading_model")

model = load_model()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    # Handle division by zero
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs.fillna(0)))

def get_live_data():
    # Fetch 1h data to match training interval
    # Period 7d is enough for indicators
    df = yf.download("BTC-USD", period="7d", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    return df.dropna()

def get_action(df, current_pos):
    last_row = df.iloc[-1]
    # Observation space: Close, High, Low, Open, Volume, SMA_10, SMA_30, RSI, position
    obs = np.array([
        last_row['Close'], last_row['High'], last_row['Low'],
        last_row['Open'], last_row['Volume'], last_row['SMA_10'],
        last_row['SMA_30'], last_row['RSI'], current_pos
    ], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return action

st.title("₿ Bitcoin RL Trading Dashboard (1h Interval)")

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

placeholder = st.empty()

# Transaction cost
TX_COST = 0.001

# Main loop for real-time updates
while True:
    try:
        df = get_live_data()
        if df.empty:
            st.error("Failed to fetch data. Retrying in 60s...")
            time.sleep(60)
            continue

        current_price = float(df.iloc[-1]['Close'])
        now = datetime.now()

        # Agent decides action
        action = get_action(df, st.session_state.current_pos)

        # Execute action logic
        if action == 1 and st.session_state.current_pos == 0:
            # Buy
            st.session_state.shares = (st.session_state.balance * (1 - TX_COST)) / current_price
            st.session_state.balance = 0
            st.session_state.current_pos = 1
            st.session_state.trade_history.append({'time': now, 'type': 'BUY', 'price': current_price})
        elif action == 0 and st.session_state.current_pos == 1:
            # Sell
            st.session_state.balance = st.session_state.shares * current_price * (1 - TX_COST)
            st.session_state.shares = 0
            st.session_state.current_pos = 0
            st.session_state.trade_history.append({'time': now, 'type': 'SELL', 'price': current_price})

        # Update portfolio value
        current_portfolio_value = st.session_state.balance + (st.session_state.shares * current_price)
        st.session_state.portfolio_history.append((now, current_portfolio_value))

        # Calculate metrics
        initial_value = 10000.0
        profit_all_time = current_portfolio_value - initial_value

        history_df = pd.DataFrame(st.session_state.portfolio_history, columns=['time', 'value'])

        # Today's profit (since 24h ago)
        val_24h_ago = history_df[history_df['time'] >= (now - timedelta(days=1))]['value'].iloc[0] if not history_df[history_df['time'] >= (now - timedelta(days=1))].empty else current_portfolio_value
        profit_today = current_portfolio_value - val_24h_ago

        # Last hour profit
        val_1h_ago = history_df[history_df['time'] >= (now - timedelta(hours=1))]['value'].iloc[0] if not history_df[history_df['time'] >= (now - timedelta(hours=1))].empty else current_portfolio_value
        profit_last_hour = current_portfolio_value - val_1h_ago

        with placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("BTC Price", f"${current_price:,.2f}")
            m2.metric("Portfolio Value", f"${current_portfolio_value:,.2f}")
            m3.metric("Current Holding", "Bitcoin" if st.session_state.current_pos == 1 else "USD")
            m4.metric("All-time Profit", f"${profit_all_time:,.2f}", f"{(profit_all_time/initial_value):.2%}")

            m1b, m2b, m3b = st.columns(3)
            m1b.metric("Profit (Last 24h)", f"${profit_today:,.2f}")
            m2b.metric("Profit (Last 1h)", f"${profit_last_hour:,.2f}")
            m3b.metric("Active Trades", len(st.session_state.trade_history))

            st.subheader("BTC-USD 1h Chart with Signals")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="BTC Price", line=dict(color='royalblue', width=2)))

            buys = [t for t in st.session_state.trade_history if t['type'] == 'BUY']
            sells = [t for t in st.session_state.trade_history if t['type'] == 'SELL']

            if buys:
                fig.add_trace(go.Scatter(x=[t['time'] for t in buys], y=[t['price'] for t in buys],
                                         mode='markers', name='Entry', marker=dict(symbol='triangle-up', size=12, color='green')))
            if sells:
                fig.add_trace(go.Scatter(x=[t['time'] for t in sells], y=[t['price'] for t in sells],
                                         mode='markers', name='Exit', marker=dict(symbol='triangle-down', size=12, color='red')))

            fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Recent Activity")
            if st.session_state.trade_history:
                history_display = pd.DataFrame(st.session_state.trade_history).tail(10)
                st.table(history_display)
            else:
                st.info("Waiting for first trade signal...")

            st.caption(f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}. Auto-refreshing every 60 seconds.")

    except Exception as e:
        st.error(f"Error: {e}")

    time.sleep(60)
