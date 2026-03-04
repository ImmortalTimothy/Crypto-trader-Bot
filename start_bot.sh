#!/bin/bash

# Launcher for the Crypto RL Trading Bot Dashboard

echo "🚀 Starting the Crypto RL Trading Bot Dashboard..."

# Check if dependencies are installed
if ! python3 -c "import streamlit, stable_baselines3, alpaca" 2>/dev/null; then
    echo "📦 Installing missing dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

# Check if model exists
if [ ! -f "Logic/ppo_trading_model.zip" ]; then
    echo "⚠️  Model not found! Running training script first..."
    # Ensure data exists for training
    python3 Logic/train.py
fi

# Run the dashboard
echo "📊 Opening dashboard..."
streamlit run dashboard.py
