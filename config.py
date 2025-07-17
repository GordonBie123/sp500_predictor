"""
Configuration file for Stock Price Predictor
Store your API keys and settings here
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys (Store in .env file for security)
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '93fea06edd8b41148e5c64949494225c')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'AS77BW0DBO45OEBN')

# Database Configuration
DATABASE_PATH = 'stock_predictor.db'
CACHE_DIR = 'cache/'

# Model Parameters
PREDICTION_DAYS = 30  # Number of days to predict
LOOKBACK_DAYS = 60   # Historical data for training
TRAIN_TEST_SPLIT = 0.8

# Technical Indicators Settings
INDICATORS = {
    'SMA': [10, 20, 50],  # Simple Moving Average periods
    'EMA': [12, 26],      # Exponential Moving Average periods
    'RSI': 14,            # Relative Strength Index period
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

# S&P 500 Top Stocks (Can be expanded)
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