"""
Machine Learning Models for Stock Price Prediction
Implements LSTM, Random Forest, XGBoost, and Ensemble models
"""
import os
# Limit TensorFlow memory and CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_CPU'] = '1'

import tensorflow as tf
# Configure TensorFlow to use less memory
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

# Then use keras directly
Sequential = keras.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
EarlyStopping = keras.callbacks.EarlyStopping
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

class StockPricePredictor:
    """
    Main class for stock price prediction using multiple models
    """
    
    def __init__(self):
        """Initialize the predictor with models and scalers"""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close',
                    lookback_days: int = 60, prediction_days: int = 30) -> Tuple:
        """
        Prepare data for training models
        
        Args:
            df: DataFrame with features
            target_col: Column to predict
            lookback_days: Days to look back for sequences
            prediction_days: Days to predict ahead
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, dates_test)
        """
        # Sort by date
        df = df.sort_values('Date').copy()
        
        # Separate date column
        dates = df['Date'].values
        
        # Remove date from features
        feature_df = df.drop(['Date'], axis=1)
        
        # Scale features
        self.scalers['features'] = MinMaxScaler()
        scaled_features = self.scalers['features'].fit_transform(feature_df)
        
        # Create sequences for LSTM
        X, y, seq_dates = self._create_sequences(
            scaled_features, 
            feature_df.columns.get_loc(target_col),
            lookback_days,
            prediction_days
        )
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_test = seq_dates[split_idx:]
        
        print(f"✅ Data prepared:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Features per sample: {X_train.shape[1:]}")
        
        return X_train, X_test, y_train, y_test, dates_test
    
    def _create_sequences(self, data: np.ndarray, target_idx: int,
                         lookback: int, prediction_days: int) -> Tuple:
        """Create sequences for time series prediction"""
        X, y, dates = [], [], []
        
        for i in range(lookback, len(data) - prediction_days):
            X.append(data[i-lookback:i])
            y.append(data[i+prediction_days-1, target_idx])
            dates.append(i+prediction_days-1)
            
        return np.array(X), np.array(y), dates
    
    def build_lstm_model(self, input_shape: Tuple):
        """
        Build LSTM neural network model
        
        Args:
            input_shape: Shape of input data (lookback_days, n_features)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_models(self, X_train: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Train all models and return performance metrics
        
        Args:
            X_train, X_test: Training and testing features
            y_train, y_test: Training and testing targets
            
        Returns:
            Dictionary with model performances
        """
        performances = {}
        
        # 1. Train LSTM Model
        print("\n🧠 Training LSTM Model...")
        lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks for better training
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        
        history = lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        self.models['lstm'] = lstm_model
        lstm_pred = lstm_model.predict(X_test).flatten()
        performances['LSTM'] = self._calculate_metrics(y_test, lstm_pred)
        print(f"   LSTM R² Score: {performances['LSTM']['r2']:.4f}")
        
        # 2. Train Random Forest
        print("\n🌲 Training Random Forest Model...")
        # Flatten sequences for tree-based models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_flat, y_train)
        self.models['random_forest'] = rf_model
        
        rf_pred = rf_model.predict(X_test_flat)
        performances['Random Forest'] = self._calculate_metrics(y_test, rf_pred)
        print(f"   Random Forest R² Score: {performances['Random Forest']['r2']:.4f}")
        
        # Store feature importance
        self.feature_importance['random_forest'] = rf_model.feature_importances_
        
        # 3. Train XGBoost
        print("\n🚀 Training XGBoost Model...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            random_state=42
        )
        xgb_model.fit(X_train_flat, y_train)
        self.models['xgboost'] = xgb_model
        
        xgb_pred = xgb_model.predict(X_test_flat)
        performances['XGBoost'] = self._calculate_metrics(y_test, xgb_pred)
        print(f"   XGBoost R² Score: {performances['XGBoost']['r2']:.4f}")
        
        # 4. Create Ensemble Predictions
        print("\n🎯 Creating Ensemble Model...")
        ensemble_pred = self._ensemble_predict(X_test, X_test_flat)
        performances['Ensemble'] = self._calculate_metrics(y_test, ensemble_pred)
        print(f"   Ensemble R² Score: {performances['Ensemble']['r2']:.4f}")
        
        self.is_trained = True
        return performances
    
    def _ensemble_predict(self, X_lstm: np.ndarray, X_flat: np.ndarray) -> np.ndarray:
        """Create ensemble predictions from all models"""
        # Get predictions from each model
        lstm_pred = self.models['lstm'].predict(X_lstm).flatten()
        rf_pred = self.models['random_forest'].predict(X_flat)
        xgb_pred = self.models['xgboost'].predict(X_flat)
        
        # Weighted average (can be optimized)
        weights = [0.4, 0.3, 0.3]  # LSTM, RF, XGBoost
        ensemble_pred = (
            weights[0] * lstm_pred +
            weights[1] * rf_pred +
            weights[2] * xgb_pred
        )
        
        return ensemble_pred
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def predict(self, X: np.ndarray, model_name: str = 'ensemble') -> np.ndarray:
        """
        Make predictions using specified model
        
        Args:
            X: Input features
            model_name: Model to use ('lstm', 'random_forest', 'xgboost', 'ensemble')
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        if model_name == 'ensemble':
            X_flat = X.reshape(X.shape[0], -1)
            return self._ensemble_predict(X, X_flat)
        elif model_name == 'lstm':
            return self.models['lstm'].predict(X).flatten()
        else:
            X_flat = X.reshape(X.shape[0], -1)
            return self.models[model_name].predict(X_flat)
    
    def save_models(self, path: str = 'models/'):
        """Save trained models"""
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save scalers
        joblib.dump(self.scalers, f'{path}/scalers.pkl')
        
        # Save tree-based models
        joblib.dump(self.models['random_forest'], f'{path}/random_forest.pkl')
        joblib.dump(self.models['xgboost'], f'{path}/xgboost.pkl')
        
        # Save LSTM
        self.models['lstm'].save(f'{path}/lstm_model.h5')
        
        print(f"✅ Models saved to {path}")
    
    def load_models(self, path: str = 'models/'):
        """Load trained models"""
        # Load scalers
        self.scalers = joblib.load(f'{path}/scalers.pkl')
        
        # Load tree-based models
        self.models['random_forest'] = joblib.load(f'{path}/random_forest.pkl')
        self.models['xgboost'] = joblib.load(f'{path}/xgboost.pkl')
        
        # Load LSTM
        from tensorflow import keras
        self.models['lstm'] = tf.keras.models.load_model(f'{path}/lstm_model.h5')
        
        self.is_trained = True
        print(f"✅ Models loaded from {path}")

# Example usage
if __name__ == "__main__":
    # This would be used with actual data
    print("Stock Price Predictor Model Module")
    print("Use with actual stock data for training")
