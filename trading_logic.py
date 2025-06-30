"""
trading_logic.py
Enhanced trading logic with advanced indicators and ML integration
"""
import logging
from datetime import datetime
import numpy as np
import pandas as pd

from data_manager import data_manager
from feature_engineering import generate_features
from ml_model import load_model, predict_signal
from config import DEFAULT_SYMBOL

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and add all required indicators: RSI, ATR, EMA_20, EMA_50, MACD, etc.
        """
        if df.empty or len(df) < 20:
            return df

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14, min_periods=14).mean()

        # EMA 20, EMA 50
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # MACD and MACD Signal
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Volume Ratio (for volume spike detection)
        df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()

        # Fill any remaining NaNs with 0 for safety
        df.fillna(0, inplace=True)
        return df

    def analyze_market(self, symbol: str = DEFAULT_SYMBOL) -> dict:
        try:
            # 1. Fetch multi-timeframe data
            df_15m = data_manager.fetch_klines(symbol, '15m', 500)
            df_1h  = data_manager.fetch_klines(symbol, '1h', 200)
            df_4h  = data_manager.fetch_klines(symbol, '4h', 100)

            # 2. Insufficient data check
            if df_15m.empty or len(df_15m) < 20:
                return self._get_insufficient_data_response()

            # 3. Calculate technical indicators
            df_15m = self.calculate_advanced_indicators(df_15m)
            df_1h  = self.calculate_advanced_indicators(df_1h)
            df_4h  = self.calculate_advanced_indicators(df_4h)

            latest_15m = df_15m.iloc[-1]
            latest_1h  = df_1h.iloc[-1] if not df_1h.empty else latest_15m
            latest_4h  = df_4h.iloc[-1] if not df_4h.empty else latest_15m

            # 4. Machine learning signal (safe wrapping)
            try:
                features = generate_features(df_15m)
                if load_model() and not features.empty:
                    ml_score = predict_signal(None, features)
                else:
                    ml_score = 0.5
            except Exception:
                logger.exception("ML feature or prediction error")
                ml_score = 0.5

            # 5. Simple decision logic example
            decision = 'HOLD'
            confidence = 0
            close = latest_15m['close']
            if close > latest_15m['EMA_20'] and ml_score > 0.6:
                decision = 'BUY'
                confidence = int(ml_score * 100)
            elif close < latest_15m['EMA_20'] and ml_score < 0.4:
                decision = 'SELL'
                confidence = int((1 - ml_score) * 100)

            # 6. Dynamic TP/SL
            tp = sl = None
            if decision in ('BUY', 'SELL'):
                atr = latest_15m.get('ATR', 0)
                if decision == 'BUY':
                    tp = round(close + atr * 2, 3)
                    sl = round(close - atr * 1.5, 3)
                else:
                    tp = round(close - atr * 2, 3)
                    sl = round(close + atr * 1.5, 3)

            # 7. Build result dict (ensure ema_20 & ema_50 always present)
            ema_20 = float(latest_15m.get('EMA_20', 0))
            ema_50 = float(latest_15m.get('EMA_50', 0))

            return {
                'decision': decision,
                'confidence': confidence,
                'price': round(close, 3),
                'tp': tp,
                'sl': sl,
                'ml_score': round(ml_score, 3),
                'rsi_15m': round(latest_15m.get('RSI', 0), 2),
                'rsi_1h': round(latest_1h.get('RSI', 0), 2),
                'atr': round(latest_15m.get('ATR', 0), 3),
                'ema_20': round(ema_20, 3),
                'ema_50': round(ema_50, 3),
                'macd': round(latest_15m.get('MACD', 0), 3),
                'macd_signal': round(latest_15m.get('MACD_Signal', 0), 3),
                'volume': int(latest_15m.get('volume', 0)),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'volume_spike': bool(latest_15m.get('Volume_Ratio', 0) > 1.5),
                'indicators_count': len([
                    c for c in df_15m.columns 
                    if c not in ['open', 'high', 'low', 'close', 'volume']
                ])
            }

        except Exception as e:
            logger.exception("Market analysis failed")
            return self._get_error_response(str(e))

    def _get_insufficient_data_response(self) -> dict:
        return {
            'decision': 'INSUFFICIENT_DATA',
            'error': 'Not enough data',
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _get_error_response(self, msg: str) -> dict:
        return {
            'decision': 'ANALYSIS_ERROR',
            'error': msg,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


# Global instance
market_analyzer = MarketAnalyzer()
