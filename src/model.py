import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import date, timedelta
from typing import Tuple, Optional, Dict, List
import os
import json
from pathlib import Path


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LEGACY_MODELS_DIR = PROJECT_ROOT / "notebooks"

# Supported cryptocurrencies
SUPPORTED_CRYPTOS = {
    "BTC-USD": {"name": "Bitcoin", "symbol": "BTC", "color": "#F7931A"},
    "ETH-USD": {"name": "Ethereum", "symbol": "ETH", "color": "#627EEA"},
    "SOL-USD": {"name": "Solana", "symbol": "SOL", "color": "#00FFA3"},
    "XRP-USD": {"name": "Ripple", "symbol": "XRP", "color": "#00AAE4"},
    "ADA-USD": {"name": "Cardano", "symbol": "ADA", "color": "#0033AD"},
}

DEFAULT_SEQUENCE_LENGTH = 60


def get_model_path(ticker: str) -> Optional[str]:
    """
    Get the model path for a specific cryptocurrency.
    Checks models/ directory first, then falls back to notebooks/ for legacy models.
    """
    crypto_info = SUPPORTED_CRYPTOS.get(ticker)
    if not crypto_info:
        return None
    
    symbol = crypto_info['symbol'].lower()
    
    # Check new models directory first
    new_model_path = MODELS_DIR / f"{symbol}_lstm_model.keras"
    if new_model_path.exists():
        return str(new_model_path)
    
    # Fall back to legacy location (BTC only)
    if symbol == "btc":
        legacy_path = LEGACY_MODELS_DIR / "btc_lstm_model.keras"
        if legacy_path.exists():
            return str(legacy_path)
    
    return None


def get_training_metrics() -> Optional[Dict]:
    """Load training metrics from the models directory"""
    metrics_file = MODELS_DIR / "training_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def get_available_models() -> List[str]:
    """Get list of tickers that have trained models available"""
    available = []
    for ticker in SUPPORTED_CRYPTOS.keys():
        if get_model_path(ticker):
            available.append(ticker)
    return available


class CryptoPredictor:
    """Main class for cryptocurrency price prediction"""
    
    def __init__(self, model_path: Optional[str] = None, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load a pre-trained model"""
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def fetch_data(self, ticker: str = "BTC-USD", start_date: str = "2020-01-01", 
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical price data from Yahoo Finance"""
        if end_date is None:
            end_date = str(date.today())
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            return df[['Close']].dropna()
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for LSTM model"""
        scaled_data = self.scaler.fit_transform(df[['Close']])
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X).reshape((-1, self.sequence_length, 1))
        y = np.array(y)
        
        return X, y
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare data for prediction only (no y values)"""
        scaled_data = self.scaler.fit_transform(df[['Close']])
        
        X = []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i, 0])
        
        return np.array(X).reshape((-1, self.sequence_length, 1))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions and inverse transform to original scale"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        predictions_scaled = self.model.predict(X, verbose=0)
        return self.scaler.inverse_transform(predictions_scaled)
    
    def predict_next_days(self, df: pd.DataFrame, days: int = 1) -> List[Dict]:
        """Predict the next N days of prices"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Fit scaler on full data
        scaled_data = self.scaler.fit_transform(df[['Close']])
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:].flatten().tolist()
        
        last_date = df.index[-1]
        
        for i in range(days):
            # Prepare input
            X = np.array(current_sequence[-self.sequence_length:]).reshape((1, self.sequence_length, 1))
            
            # Predict
            pred_scaled = self.model.predict(X, verbose=0)[0][0]
            pred_price = self.scaler.inverse_transform([[pred_scaled]])[0][0]
            
            # Calculate prediction date (skip weekends for display, crypto trades 24/7)
            pred_date = last_date + timedelta(days=i + 1)
            
            predictions.append({
                "date": pred_date,
                "predicted_price": float(pred_price),
                "day_number": i + 1
            })
            
            # Add prediction to sequence for next iteration
            current_sequence.append(pred_scaled)
        
        return predictions
    
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate various performance metrics"""
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        avg_price = float(np.mean(actual))
        rmse_pct = (rmse / avg_price) * 100
        mae_pct = (mae / avg_price) * 100
        
        # Calculate directional accuracy
        if len(actual) > 1:
            actual_direction = np.diff(actual.flatten()) > 0
            pred_direction = np.diff(predicted.flatten()) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        return {
            "rmse": rmse,
            "mae": mae,
            "avg_price": avg_price,
            "rmse_pct": rmse_pct,
            "mae_pct": mae_pct,
            "directional_accuracy": directional_accuracy
        }
    
    @staticmethod
    def build_model(sequence_length: int = DEFAULT_SEQUENCE_LENGTH, 
                    lstm_units: int = 50, 
                    dropout_rate: float = 0.2) -> Sequential:
        """Build a new LSTM model architecture"""
        model = Sequential([
            Input(shape=(sequence_length, 1)),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model


def get_price_change_info(df: pd.DataFrame) -> Dict:
    """Get price change statistics"""
    if len(df) < 2:
        return {}
    
    current_price = float(df['Close'].iloc[-1])
    
    # 24h change
    if len(df) >= 2:
        prev_price = float(df['Close'].iloc[-2])
        change_24h = current_price - prev_price
        change_24h_pct = (change_24h / prev_price) * 100
    else:
        change_24h = change_24h_pct = 0
    
    # 7d change
    if len(df) >= 7:
        price_7d = float(df['Close'].iloc[-7])
        change_7d = current_price - price_7d
        change_7d_pct = (change_7d / price_7d) * 100
    else:
        change_7d = change_7d_pct = 0
    
    # 30d change
    if len(df) >= 30:
        price_30d = float(df['Close'].iloc[-30])
        change_30d = current_price - price_30d
        change_30d_pct = (change_30d / price_30d) * 100
    else:
        change_30d = change_30d_pct = 0
    
    # All-time high/low in dataset
    all_time_high = float(df['Close'].max())
    all_time_low = float(df['Close'].min())
    
    return {
        "current_price": current_price,
        "change_24h": change_24h,
        "change_24h_pct": change_24h_pct,
        "change_7d": change_7d,
        "change_7d_pct": change_7d_pct,
        "change_30d": change_30d,
        "change_30d_pct": change_30d_pct,
        "all_time_high": all_time_high,
        "all_time_low": all_time_low,
    }

