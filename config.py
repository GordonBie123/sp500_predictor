"""
Configuration file for Stock Price Predictor
Store your API keys and settings here
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys - Try Streamlit secrets first, then environment variables
try:
    # In Streamlit Cloud
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    ALPHA_VANTAGE_KEY = st.secrets["ALPHA_VANTAGE_KEY"]
except:
    # Local development
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'your_newsapi_key_here')
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'your_alpha_vantage_key_here')

# Rest of your config remains the same...
DATABASE_PATH = 'stock_predictor.db'
CACHE_DIR = 'cache/'

# Model Parameters
PREDICTION_DAYS = 30
LOOKBACK_DAYS = 60
TRAIN_TEST_SPLIT = 0.8

# Technical Indicators Settings
INDICATORS = {
    'SMA': [10, 20, 50],
    'EMA': [12, 26],
    'RSI': 14,
    'MACD': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    }
}

# News Settings
NEWS_SOURCES = [
    'bloomberg',
    'reuters',
    'the-wall-street-journal',
    'financial-times',
    'cnbc'
]
MAX_NEWS_ARTICLES = 100
NEWS_LOOKBACK_DAYS = 7

# S&P 500 Top Stocks
SP500_SYMBOLS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
    'ABBV', 'PFE', 'LLY', 'WMT', 'DIS', 'CSCO', 'ACN', 'CRM', 'AVGO',
    'MCD', 'COST', 'TMO', 'ABT', 'PEP', 'NFLX', 'CMCSA', 'VZ', 'DHR',
    'INTC', 'ADBE', 'NKE', 'TXN', 'PM', 'UNP', 'NEE', 'T', 'COP',
    'HON', 'MS', 'LOW', 'UPS', 'RTX', 'BMY', 'QCOM', 'CVS', 'SBUX'
]

# Model Hyperparameters
LSTM_PARAMS = {
    'units': [128, 64, 32],
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.01,
    'random_state': 42
}
