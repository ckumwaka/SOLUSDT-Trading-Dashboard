"""
Advanced feature engineering for machine learning models
"""

import pandas as pd
import numpy as np
from typing import List
import logging
from config import ML_FEATURES

logger = logging.getLogger(__name__)

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive ML features from market data"""
    
    if df.empty or len(df) < 50:
        logger.warning("Insufficient data for feature generation")
        return pd.DataFrame()
    
    try:
        features = df.copy()
        
        # Ensure required columns exist
        required_cols = ['RSI', 'MACD', 'MACD_Signal', 'volume', 'close', 'EMA_20', 'EMA_50', 'ATR']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Basic signal features
        features['RSI_30'] = (features['RSI'] < 30).astype(int)
        features['RSI_70'] = (features['RSI'] > 70).astype(int)
        features['RSI_Oversold'] = (features['RSI'] < 25).astype(int)
        features['RSI_Overbought'] = (features['RSI'] > 75).astype(int)
        
        # MACD features
        features['MACD_Cross'] = (features['MACD'] > features['MACD_Signal']).astype(int)
        features['MACD_Bullish'] = ((features['MACD'] > 0) & (features['MACD'] > features['MACD_Signal'])).astype(int)
        features['MACD_Bearish'] = ((features['MACD'] < 0) & (features['MACD'] < features['MACD_Signal'])).astype(int)
        
        # Volume analysis features
        features['Volume_Spike'] = (features['volume'] > features['volume'].rolling(20).mean() * 1.5).astype(int)
        features['Volume_Drought'] = (features['volume'] < features['volume'].rolling(20).mean() * 0.5).astype(int)
        features['Volume_Ratio'] = features['volume'] / features['volume'].rolling(20).mean()
        
        # Price position features
        features['Above_EMA20'] = (features['close'] > features['EMA_20']).astype(int)
        features['Above_EMA50'] = (features['close'] > features['EMA_50']).astype(int)
        features['EMA_Cross'] = (features['EMA_20'] > features['EMA_50']).astype(int)
        
        # Momentum features
        features['Momentum_5'] = features['close'].pct_change(5)
        features['Momentum_15'] = features['close'].pct_change(15)
        features['Momentum_30'] = features['close'].pct_change(30)
        
        # Volatility features
        features['ATR_Normalized'] = features['ATR'] / features['close']
        features['High_Volatility'] = (features['ATR_Normalized'] > features['ATR_Normalized'].rolling(20).quantile(0.8)).astype(int)
        features['Low_Volatility'] = (features['ATR_Normalized'] < features['ATR_Normalized'].rolling(20).quantile(0.2)).astype(int)
        
        # Price pattern features
        if 'BB_Position' in features.columns:
            features['BB_Squeeze'] = (features['BB_Width'] < features['BB_Width'].rolling(20).quantile(0.2)).astype(int)
            features['BB_Expansion'] = (features['BB_Width'] > features['BB_Width'].rolling(20).quantile(0.8)).astype(int)
            features['Near_BB_Upper'] = (features['BB_Position'] > 0.8).astype(int)
            features['Near_BB_Lower'] = (features['BB_Position'] < 0.2).astype(int)
        
        # Support/Resistance features
        if 'Price_Position' in features.columns:
            features['Near_Resistance'] = (features['Price_Position'] > 0.8).astype(int)
            features['Near_Support'] = (features['Price_Position'] < 0.2).astype(int)
            features['Mid_Range'] = ((features['Price_Position'] > 0.4) & (features['Price_Position'] < 0.6)).astype(int)
        
        # Trend strength features
        features['Strong_Uptrend'] = ((features['close'] > features['EMA_20']) & 
                                    (features['EMA_20'] > features['EMA_50']) & 
                                    (features['RSI'] > 50) & 
                                    (features['MACD'] > features['MACD_Signal'])).astype(int)
        
        features['Strong_Downtrend'] = ((features['close'] < features['EMA_20']) & 
                                      (features['EMA_20'] < features['EMA_50']) & 
                                      (features['RSI'] < 50) & 
                                      (features['MACD'] < features['MACD_Signal'])).astype(int)
        
        # Divergence features
        features['Price_RSI_Divergence'] = calculate_divergence(features['close'], features['RSI'])
        features['Price_MACD_Divergence'] = calculate_divergence(features['close'], features['MACD'])
        
        # Market regime features
        features['Trending_Market'] = calculate_trending_regime(features)
        features['Ranging_Market'] = 1 - features['Trending_Market']
        
        # Higher timeframe alignment (simulated)
        features['HTF_Bullish'] = simulate_htf_alignment(features, 'bullish')
        features['HTF_Bearish'] = simulate_htf_alignment(features, 'bearish')
        
        # Select final features for ML
        available_features = [feat for feat in ML_FEATURES if feat in features.columns]
        
        if len(available_features) < len(ML_FEATURES) * 0.8:
            logger.warning(f"Only {len(available_features)}/{len(ML_FEATURES)} features available")
        
        # Add additional computed features
        computed_features = [
            'RSI_Oversold', 'RSI_Overbought', 'MACD_Bullish', 'MACD_Bearish',
            'Volume_Drought', 'High_Volatility', 'Low_Volatility', 'Strong_Uptrend',
            'Strong_Downtrend', 'Price_RSI_Divergence', 'Trending_Market', 'HTF_Bullish'
        ]
        
        final_features = available_features + [f for f in computed_features if f in features.columns]
        
        result = features[final_features].dropna()
        logger.info(f"Generated {len(final_features)} features from {len(df)} data points")
        
        return result
        
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        return pd.DataFrame()

def calculate_divergence(price_series: pd.Series, indicator_series: pd.Series, 
                        window: int = 20) -> pd.Series:
    """Calculate divergence between price and indicator"""
    try:
        price_slope = price_series.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        indicator_slope = indicator_series.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Divergence occurs when slopes have opposite signs
        divergence = ((price_slope > 0) & (indicator_slope < 0)) | ((price_slope < 0) & (indicator_slope > 0))
        return divergence.astype(int)
    except:
        return pd.Series(0, index=price_series.index)

def calculate_trending_regime(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Determine if market is in trending or ranging regime"""
    try:
        # Use ADX-like calculation
        if 'ATR' in df.columns and 'close' in df.columns:
            price_change = df['close'].diff().abs()
            trending_strength = price_change.rolling(window).sum() / (df['ATR'].rolling(window).sum() + 1e-8)
            trending_regime = (trending_strength > trending_strength.quantile(0.6)).astype(int)
            return trending_regime
        else:
            return pd.Series(0, index=df.index)
    except:
        return pd.Series(0, index=df.index)

def simulate_htf_alignment(df: pd.DataFrame, direction: str) -> pd.Series:
    """Simulate higher timeframe alignment"""
    try:
        if direction == 'bullish':
            # Simulate bullish HTF alignment
            condition = ((df['EMA_20'] > df['EMA_50']) & 
                        (df['RSI'] > 45) & 
                        (df['MACD'] > 0))
        else:
            # Simulate bearish HTF alignment
            condition = ((df['EMA_20'] < df['EMA_50']) & 
                        (df['RSI'] < 55) & 
                        (df['MACD'] < 0))
        
        return condition.astype(int)
    except:
        return pd.Series(0, index=df.index)

def get_feature_importance(model, feature_names: List[str]) -> dict:
    """Get feature importance from trained model"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        return {}

def validate_features(features: pd.DataFrame) -> bool:
    """Validate feature DataFrame for ML training"""
    if features.empty:
        return False
    
    # Check for infinite values
    if np.isinf(features.values).any():
        logger.warning("Infinite values detected in features")
        return False
    
    # Check for excessive NaN values
    nan_percentage = features.isnull().mean().mean()
    if nan_percentage > 0.1:  # More than 10% NaN
        logger.warning(f"High NaN percentage in features: {nan_percentage:.2%}")
        return False
    
    # Check feature variance
    low_variance_features = features.columns[features.var() < 1e-6].tolist()
    if low_variance_features:
        logger.warning(f"Low variance features detected: {low_variance_features}")
    
    return True
