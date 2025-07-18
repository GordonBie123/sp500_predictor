"""
Stock Data Fetcher Module
Handles all stock data collection using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from typing import Dict, List, Optional
import os
import joblib
from tqdm import tqdm
import config  # Add this line

class StockDataFetcher:
    """
    Class to fetch and process stock data
    """
    
    def __init__(self, cache_dir: str = 'cache/'):
        """
        Initialize the data fetcher
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def fetch_stock_data(self, symbol: str, period: str = '2y', 
                    interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical stock data using Alpha Vantage API
        """
        try:
            import requests
            import config  # Add this import
            
            # Get API key
            api_key = config.ALPHA_VANTAGE_KEY
            
            # Alpha Vantage endpoint
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full'
            
            print(f"Fetching from Alpha Vantage: {symbol}")
            response = requests.get(url, timeout=10)
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data:
                print(f"API Error: {data['Error Message']}")
                return pd.DataFrame()
            
            if 'Note' in data:
                print(f"API Rate limit: {data['Note']}")
                return pd.DataFrame()
                
            if 'Time Series (Daily)' not in data:
                print(f"No time series data in response")
                return pd.DataFrame()
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Reset index and rename
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Add symbol
            df['Symbol'] = symbol
            
            print(f"✅ Successfully fetched {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # Exponential Moving Averages
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # RSI (Relative Strength Index)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = bb.bollinger_wband()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # Support and Resistance levels
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Support_Resistance_Ratio'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])
        
        print("✅ Technical indicators added successfully")
        
        return df
    
    def get_market_data(self) -> pd.DataFrame:
        """
        Get market-wide indicators (VIX, S&P 500, etc.)
        
        Returns:
            DataFrame with market indicators
        """
        market_symbols = {
            'SP500': '^GSPC',
            'VIX': '^VIX',
            'DXY': 'DX-Y.NYB',  # Dollar Index
            'GOLD': 'GC=F',     # Gold Futures
            'OIL': 'CL=F'       # Crude Oil Futures
        }
        
        market_data = {}
        
        for name, symbol in market_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1mo')
                if not hist.empty:
                    market_data[name] = hist['Close'].iloc[-1]
                    market_data[f'{name}_Change'] = hist['Close'].pct_change().iloc[-1]
            except:
                print(f"Warning: Could not fetch {name} data")
                
        return pd.DataFrame([market_data])
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final feature set for ML models
        
        Args:
            df: DataFrame with stock data and indicators
            
        Returns:
            DataFrame with cleaned features
        """
        # Remove any NaN values created by indicators
        df = df.dropna()
        
        # Select relevant features for modeling
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
            'Volume_ratio', 'High_Low_Pct',
            'Price_Change', 'Price_Change_5d', 'Price_Change_20d',
            'Volatility', 'Support_Resistance_Ratio'
        ]
        
        # Keep only available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        return df[['Date'] + available_features].copy()
    
    def save_to_cache(self, df: pd.DataFrame, filename: str):
        """Save dataframe to cache"""
        filepath = os.path.join(self.cache_dir, f"{filename}.pkl")
        joblib.dump(df, filepath)
        print(f"📁 Data saved to cache: {filepath}")
    
    def load_from_cache(self, filename: str) -> Optional[pd.DataFrame]:
        """Load dataframe from cache"""
        filepath = os.path.join(self.cache_dir, f"{filename}.pkl")
        if os.path.exists(filepath):
            print(f"📂 Loading data from cache: {filepath}")
            return joblib.load(filepath)
        return None

# Example usage
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = StockDataFetcher()
    
    # Fetch data for UNH
    symbol = 'UNH'
    data = fetcher.fetch_stock_data(symbol)
    
    # Add technical indicators
    data = fetcher.add_technical_indicators(data)
    
    # Prepare features
    features = fetcher.prepare_features(data)
    
    print(f"\nFinal feature shape: {features.shape}")
    print(f"Features: {features.columns.tolist()}")
