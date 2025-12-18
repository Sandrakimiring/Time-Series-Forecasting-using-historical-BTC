import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SUPPORTED_CRYPTOS = {
    "BTC-USD": {"name": "Bitcoin", "symbol": "BTC"},
    "ETH-USD": {"name": "Ethereum", "symbol": "ETH"},
    "SOL-USD": {"name": "Solana", "symbol": "SOL"},
    "XRP-USD": {"name": "Ripple", "symbol": "XRP"},
    "ADA-USD": {"name": "Cardano", "symbol": "ADA"},
}

SEQUENCE_LENGTH = 60
TRAINING_YEARS = 2
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"
METRICS_FILE = MODELS_DIR / "training_metrics.json"


def fetch_data(ticker: str, years: int = TRAINING_YEARS) -> pd.DataFrame:
    """Fetch historical price data from Yahoo Finance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    logger.info(f"Fetching {ticker} data from {start_date.date()} to {end_date.date()}")
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[['Close']].dropna()
        logger.info(f"Fetched {len(df)} days of data for {ticker}")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        raise


def prepare_data(df: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH):
    """Prepare data for LSTM training"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X).reshape((-1, sequence_length, 1))
    y = np.array(y)
    
    # Split data
    split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler


def build_model(sequence_length: int = SEQUENCE_LENGTH) -> Sequential:
    """Build LSTM model architecture"""
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(ticker: str, crypto_info: dict) -> dict:
    """Train a model for a specific cryptocurrency"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training model for {crypto_info['name']} ({ticker})")
    logger.info(f"{'='*60}")
    
    try:
        # Fetch and prepare data
        df = fetch_data(ticker)
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Build model
        model = build_model()
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train
        logger.info(f"Starting training for {EPOCHS} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        predictions = model.predict(X_test, verbose=0)
        predictions_unscaled = scaler.inverse_transform(predictions)
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions_unscaled))
        mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
        avg_price = float(np.mean(y_test_unscaled))
        rmse_pct = (rmse / avg_price) * 100
        
        # Calculate directional accuracy
        actual_direction = np.diff(y_test_unscaled.flatten()) > 0
        pred_direction = np.diff(predictions_unscaled.flatten()) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Save model
        model_path = MODELS_DIR / f"{crypto_info['symbol'].lower()}_lstm_model.keras"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        metrics = {
            "ticker": ticker,
            "name": crypto_info['name'],
            "symbol": crypto_info['symbol'],
            "rmse": float(rmse),
            "mae": float(mae),
            "rmse_pct": float(rmse_pct),
            "avg_price": float(avg_price),
            "directional_accuracy": float(directional_accuracy),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "epochs_trained": len(history.history['loss']),
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "trained_at": datetime.now().isoformat(),
            "model_path": str(model_path)
        }
        
        logger.info(f" {crypto_info['name']} - RMSE: ${rmse:.2f} ({rmse_pct:.2f}%), Direction Accuracy: {directional_accuracy:.1f}%")
        
        return metrics
    
    except Exception as e:
        logger.error(f" Failed to train {ticker}: {e}")
        return {
            "ticker": ticker,
            "name": crypto_info['name'],
            "symbol": crypto_info['symbol'],
            "error": str(e),
            "trained_at": datetime.now().isoformat()
        }


def main():
    """Main training pipeline"""
    logger.info("Starting automated model training pipeline")
    logger.info(f"Training {len(SUPPORTED_CRYPTOS)} cryptocurrency models")
    logger.info(f"Using {TRAINING_YEARS} years of historical data")
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Train all models
    all_metrics = {
        "last_updated": datetime.now().isoformat(),
        "training_config": {
            "sequence_length": SEQUENCE_LENGTH,
            "training_years": TRAINING_YEARS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "validation_split": VALIDATION_SPLIT
        },
        "models": {}
    }
    
    successful = 0
    failed = 0
    
    for ticker, info in SUPPORTED_CRYPTOS.items():
        try:
            metrics = train_model(ticker, info)
            all_metrics["models"][info['symbol']] = metrics
            
            if "error" not in metrics:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"Unexpected error training {ticker}: {e}")
            failed += 1
            all_metrics["models"][info['symbol']] = {
                "ticker": ticker,
                "error": str(e),
                "trained_at": datetime.now().isoformat()
            }
    
    # Save metrics
    with open(METRICS_FILE, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {METRICS_FILE}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Successful: {successful}/{len(SUPPORTED_CRYPTOS)}")
    logger.info(f" Failed: {failed}/{len(SUPPORTED_CRYPTOS)}")
    
    if successful > 0:
        logger.info("\nModel Performance:")
        for symbol, metrics in all_metrics["models"].items():
            if "error" not in metrics:
                logger.info(f"  {metrics['name']:12} | RMSE: ${metrics['rmse']:>10,.2f} ({metrics['rmse_pct']:.2f}%) | Dir Acc: {metrics['directional_accuracy']:.1f}%")
    
    logger.info("\n Training pipeline completed!")
    
    # Exit with error code if any failed (for GitHub Actions)
    if failed > 0:
        sys.exit(1)
    
    return all_metrics


if __name__ == "__main__":
    main()

