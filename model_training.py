"""
model_training.py
=================
Script to collect historical SOL/USDT data, engineer features, train the ML model,
and optionally schedule weekly retraining.
"""
import os
import time
import traceback
import schedule
import pandas as pd

from data_manager import data_manager
from trading_logic import TechnicalAnalyzer
from feature_engineering import generate_features
from ml_model import train_model

DATA_DIR   = "data"
MODEL_DIR  = "model"
SYMBOL     = "SOLUSDT"
TIMEFRAME  = "15m"
LIMIT      = 1000

def collect_training_data(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    limit: int = LIMIT
) -> str:
    """
    Fetches OHLCV data, calculates indicators via TechnicalAnalyzer,
    and saves CSV. Returns the CSV file path.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"historical_{symbol}_{timeframe}.csv")

    print(f"Collecting {limit} bars of {timeframe} data for {symbol}...")
    df = data_manager.fetch_klines(symbol, timeframe, limit)
    if df.empty:
        raise RuntimeError("Failed to fetch any OHLCV data.")

    # Calculate advanced indicators
    analyzer = TechnicalAnalyzer()
    df = analyzer.calculate_advanced_indicators(df)
    if df.empty:
        raise RuntimeError("Indicator calculation returned empty DataFrame.")

    df.to_csv(path)
    print(f"Data saved to {path}")
    return path

def training_workflow():
    """One-shot training workflow."""
    try:
        data_path = collect_training_data()
        print("Training machine learning model...")
        metrics = train_model(data_path)
        print("Model training completed!")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        return metrics
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        return None

def scheduled_retraining():
    """Scheduled weekly retraining."""
    print("\n=== Scheduled Retraining Started ===")
    training_workflow()
    print("=== Retraining Finished ===\n")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Initial training
    training_workflow()

    # Weekly schedule
    schedule.every().sunday.at("02:00").do(scheduled_retraining)
    print("Scheduler started. Model will retrain weekly on Sundays at 02:00 AM.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("Model training scheduler stopped by user.")
