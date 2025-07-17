"""
Utility functions for the Stock Price Predictor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import os
import json

def get_company_info(symbol: str) -> Dict:
    """
    Get company information for a stock symbol
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with company information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'employees': info.get('fullTimeEmployees', 0),
            'description': info.get('longBusinessSummary', '')[:200] + '...'
        }
    except:
        return {
            'name': symbol,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
            'employees': 0,
            'description': 'Information not available'
        }

def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate risk metrics for a stock
    
    Args:
        returns: Series of returns
        
    Returns:
        Dictionary with risk metrics
    """
    # Annual trading days
    trading_days = 252
    
    # Calculate metrics
    daily_returns = returns.dropna()
    
    metrics = {
        'volatility': daily_returns.std() * np.sqrt(trading_days),
        'sharpe_ratio': (daily_returns.mean() * trading_days) / (daily_returns.std() * np.sqrt(trading_days)),
        'max_drawdown': calculate_max_drawdown(returns),
        'var_95': np.percentile(daily_returns, 5),
        'cvar_95': daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean()
    }
    
    return metrics

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def validate_api_key(api_key: str, api_type: str = 'newsapi') -> bool:
    """
    Validate API key
    
    Args:
        api_key: API key to validate
        api_type: Type of API ('newsapi', 'alphavantage')
        
    Returns:
        Boolean indicating if key is valid
    """
    if not api_key or api_key == 'your_newsapi_key_here':
        return False
    
    if api_type == 'newsapi':
        # Test NewsAPI key
        try:
            import requests
            response = requests.get(
                f'https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'
            )
            return response.status_code == 200
        except:
            return False
    
    return True

def create_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create trading signals based on technical indicators
    
    Args:
        df: DataFrame with price and technical indicators
        
    Returns:
        DataFrame with added signal columns
    """
    df = df.copy()
    
    # Moving Average Crossover
    df['MA_Signal'] = 0
    df.loc[df['SMA_10'] > df['SMA_20'], 'MA_Signal'] = 1
    df.loc[df['SMA_10'] < df['SMA_20'], 'MA_Signal'] = -1
    
    # RSI Signal
    df['RSI_Signal'] = 0
    df.loc[df['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold - Buy
    df.loc[df['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought - Sell
    
    # MACD Signal
    df['MACD_Signal_Line'] = 0
    df.loc[df['MACD'] > df['MACD_signal'], 'MACD_Signal_Line'] = 1
    df.loc[df['MACD'] < df['MACD_signal'], 'MACD_Signal_Line'] = -1
    
    # Combined Signal
    df['Combined_Signal'] = (
        df['MA_Signal'] * 0.3 +
        df['RSI_Signal'] * 0.3 +
        df['MACD_Signal_Line'] * 0.4
    )
    
    return df

def format_large_number(num: float) -> str:
    """Format large numbers with K, M, B suffixes"""
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def save_prediction_history(symbol: str, prediction: Dict, filepath: str = 'predictions_history.json'):
    """
    Save prediction to history file
    
    Args:
        symbol: Stock symbol
        prediction: Prediction data
        filepath: Path to history file
    """
    # Load existing history
    history = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            history = json.load(f)
    
    # Add new prediction
    if symbol not in history:
        history[symbol] = []
    
    prediction['timestamp'] = datetime.now().isoformat()
    history[symbol].append(prediction)
    
    # Keep only last 100 predictions per symbol
    history[symbol] = history[symbol][-100:]
    
    # Save updated history
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)

def load_prediction_history(symbol: str, filepath: str = 'predictions_history.json') -> List[Dict]:
    """Load prediction history for a symbol"""
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r') as f:
        history = json.load(f)
    
    return history.get(symbol, [])

def calculate_prediction_accuracy(history: List[Dict]) -> Dict[str, float]:
    """
    Calculate accuracy metrics from prediction history
    
    Args:
        history: List of historical predictions
        
    Returns:
        Dictionary with accuracy metrics
    """
    if len(history) < 5:
        return {
            'accuracy': 0,
            'avg_error': 0,
            'directional_accuracy': 0
        }
    
    errors = []
    directional_correct = 0
    
    for pred in history:
        if 'actual_price' in pred and 'predicted_price' in pred:
            error = abs(pred['actual_price'] - pred['predicted_price']) / pred['actual_price']
            errors.append(error)
            
            # Check directional accuracy
            pred_direction = pred['predicted_price'] > pred['current_price']
            actual_direction = pred['actual_price'] > pred['current_price']
            if pred_direction == actual_direction:
                directional_correct += 1
    
    return {
        'accuracy': 1 - np.mean(errors) if errors else 0,
        'avg_error': np.mean(errors) if errors else 0,
        'directional_accuracy': directional_correct / len(errors) if errors else 0
    }

# Example usage
if __name__ == "__main__":
    # Test company info
    info = get_company_info('AAPL')
    print("Company Info:", info)
    
    # Test number formatting
    print(f"Market Cap: {format_large_number(2.5e12)}")  # $2.50T