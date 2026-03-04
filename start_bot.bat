@echo off
echo 🚀 Starting the Crypto RL Trading Bot Dashboard...

:: Check if venv exists and activate
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: Check dependencies
python -c "import streamlit, stable_baselines3, alpaca" 2>nul
if %errorlevel% neq 0 (
    echo 📦 Installing missing dependencies...
    pip install -r requirements.txt
)

:: Check model
if not exist Logic\ppo_trading_model.zip (
    echo ⚠️  Model not found! Running training script first...
    python Logic\train.py
)

:: Run dashboard
echo 📊 Opening dashboard...
streamlit run dashboard.py
pause
