"""
Data Preprocessing Module for Stock Price Predictor
Handles data cleaning, feature engineering, and preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class StockDataPreprocessor:
    """
    Class for preprocessing stock data for ML models
    """
    
    def __init__(self, scaling_method: str = 'minmax'):
        """
        Initialize preprocessor
        
        Args:
            scaling_method: Method for scaling ('minmax', 'standard', 'robust')
        """
        self.scaling_method = scaling_method
        self.scaler = self._get_scaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names = []
        
    def _get_scaler(self):
        """Get scaler based on method"""
        if self.scaling_method == 'minmax':
            return MinMaxScaler(feature_range=(0, 1))
        elif self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for better predictions
        
        Args:
            df: DataFrame with basic OHLCV and technical indicators
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Price-based features
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_Pct'] = df['Price_Range'] / df['Close']
        df['Close_to_High'] = (df['High'] - df['Close']) / df['High']
        df['Close_to_Low'] = (df['Close'] - df['Low']) / df['Low']
        
        # Volume features
        df['Volume_Price_Trend'] = df['Volume'] * df['Close']
        df['Volume_Weighted_Price'] = (df['Volume'] * df['Close']).rolling(window=10).sum() / df['Volume'].rolling(window=10).sum()
        
        # Trend features
        df['Price_Momentum_3d'] = df['Close'].pct_change(periods=3)
        df['Price_Momentum_7d'] = df['Close'].pct_change(periods=7)
        df['Price_Momentum_14d'] = df['Close'].pct_change(periods=14)
        
        # Volatility features
        df['Return_Volatility_7d'] = df['Price_Change'].rolling(window=7).std()
        df['Return_Volatility_14d'] = df['Price_Change'].rolling(window=14).std()
        df['Return_Volatility_30d'] = df['Price_Change'].rolling(window=30).std()
        
        # Support/Resistance features
        df['Distance_from_20d_High'] = (df['Close'] - df['High'].rolling(window=20).max()) / df['Close']
        df['Distance_from_20d_Low'] = (df['Close'] - df['Low'].rolling(window=20).min()) / df['Close']
        
        # Pattern recognition features
        df['Bullish_Engulfing'] = self._detect_bullish_engulfing(df)
        df['Bearish_Engulfing'] = self._detect_bearish_engulfing(df)
        df['Hammer'] = self._detect_hammer(df)
        df['Shooting_Star'] = self._detect_shooting_star(df)
        
        # Market regime features
        df['Trend_Strength'] = self._calculate_trend_strength(df)
        df['Market_Regime'] = self._classify_market_regime(df)
        
        return df
    
    def _detect_bullish_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect bullish engulfing pattern"""
        prev_open = df['Open'].shift(1)
        prev_close = df['Close'].shift(1)
        
        pattern = (
            (prev_close < prev_open) &  # Previous candle is bearish
            (df['Close'] > df['Open']) &  # Current candle is bullish
            (df['Open'] < prev_close) &  # Current open is below previous close
            (df['Close'] > prev_open)  # Current close is above previous open
        ).astype(int)
        
        return pattern
    
    def _detect_bearish_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect bearish engulfing pattern"""
        prev_open = df['Open'].shift(1)
        prev_close = df['Close'].shift(1)
        
        pattern = (
            (prev_close > prev_open) &  # Previous candle is bullish
            (df['Close'] < df['Open']) &  # Current candle is bearish
            (df['Open'] > prev_close) &  # Current open is above previous close
            (df['Close'] < prev_open)  # Current close is below previous open
        ).astype(int)
        
        return pattern
    
    def _detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect hammer pattern"""
        body = abs(df['Close'] - df['Open'])
        lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
        upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
        
        pattern = (
            (lower_shadow > 2 * body) &  # Long lower shadow
            (upper_shadow < 0.1 * body)  # Small upper shadow
        ).astype(int)
        
        return pattern
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect shooting star pattern"""
        body = abs(df['Close'] - df['Open'])
        lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
        upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
        
        pattern = (
            (upper_shadow > 2 * body) &  # Long upper shadow
            (lower_shadow < 0.1 * body)  # Small lower shadow
        ).astype(int)
        
        return pattern
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using ADX concept"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        trend_strength = df['Price_Change'].rolling(window=14).mean() / atr
        return trend_strength.fillna(0)
    
    def _classify_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify market regime (trending up/down, ranging)"""
        sma_short = df['Close'].rolling(window=10).mean()
        sma_long = df['Close'].rolling(window=30).mean()
        
        regime = pd.Series(index=df.index, dtype=int)
        regime[sma_short > sma_long * 1.02] = 1  # Uptrend
        regime[sma_short < sma_long * 0.98] = -1  # Downtrend
        regime.fillna(0, inplace=True)  # Ranging
        
        return regime
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in the dataset
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with handled missing values
        """
        # Forward fill for time series continuity
        df = df.fillna(method='ffill')
        
        # Backward fill for any remaining NaN at the beginning
        df = df.fillna(method='bfill')
        
        # If still any NaN, use imputer
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if df[numeric_columns].isnull().any().any():
            df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None,
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in the data
        
        Args:
            df: DataFrame
            columns: Columns to check for outliers
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outliers = pd.DataFrame(index=df.index)
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[f'{col}_outlier'] = (
                    (df[col] < (Q1 - threshold * IQR)) |
                    (df[col] > (Q3 + threshold * IQR))
                )
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[f'{col}_outlier'] = z_scores > threshold
        
        return outliers
    
    def create_target_variables(self, df: pd.DataFrame, target_col: str = 'Close',
                               horizons: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Create multiple target variables for different prediction horizons
        
        Args:
            df: DataFrame with price data
            target_col: Column to create targets from
            horizons: List of prediction horizons (days)
            
        Returns:
            DataFrame with target variables
        """
        for horizon in horizons:
            # Future price
            df[f'Target_{horizon}d'] = df[target_col].shift(-horizon)
            
            # Future return
            df[f'Target_Return_{horizon}d'] = (
                df[f'Target_{horizon}d'] / df[target_col] - 1
            ) * 100
            
            # Binary classification (up/down)
            df[f'Target_Direction_{horizon}d'] = (
                df[f'Target_Return_{horizon}d'] > 0
            ).astype(int)
            
            # Multi-class (strong down, down, neutral, up, strong up)
            df[f'Target_Class_{horizon}d'] = pd.cut(
                df[f'Target_Return_{horizon}d'],
                bins=[-np.inf, -2, -0.5, 0.5, 2, np.inf],
                labels=[0, 1, 2, 3, 4]
            )
        
        return df
    
    def prepare_final_dataset(self, df: pd.DataFrame, feature_cols: List[str],
                            target_col: str, remove_outliers: bool = False) -> Tuple:
        """
        Prepare final dataset for modeling
        
        Args:
            df: DataFrame with all features
            feature_cols: List of feature columns to use
            target_col: Target column name
            remove_outliers: Whether to remove outliers
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Handle missing data
        df = self.handle_missing_data(df)
        
        # Remove outliers if requested
        if remove_outliers:
            outliers = self.detect_outliers(df, columns=feature_cols)
            outlier_mask = ~outliers.any(axis=1)
            df = df[outlier_mask]
        
        # Select features and target
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Remove rows with NaN target
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Store feature names
        self.feature_names = feature_cols
        
        return X, y, feature_cols

# Example usage
if __name__ == "__main__":
    # Example preprocessing pipeline
    preprocessor = StockDataPreprocessor(scaling_method='minmax')
    
    print("Stock Data Preprocessor initialized")
    print("Available methods:")
    print("- create_advanced_features()")
    print("- handle_missing_data()")
    print("- detect_outliers()")
    print("- create_target_variables()")
    print("- prepare_final_dataset()")