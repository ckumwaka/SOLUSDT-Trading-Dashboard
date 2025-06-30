"""
Configuration settings for SOL/USDT Trading Dashboard
"""

# API Configuration
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/"
RATE_LIMIT_DELAY = 0.1

# Trading Configuration
DEFAULT_SYMBOL = "SOLUSDT"
TIMEFRAMES = {
    '1m': '1m',
    '5m': '5m', 
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}

# Technical Indicators Configuration
RSI_PERIOD = 14
ATR_PERIOD = 14
EMA_FAST = 12
EMA_SLOW = 26
EMA_SIGNAL = 9
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ML Configuration
ML_FEATURES = [
    'RSI_30', 'RSI_70', 'MACD_Cross', 'Volume_Spike',
    'Above_EMA20', 'Above_EMA50', 'Momentum_5', 'Momentum_15',
    'ATR_Normalized', 'Price_Position', 'Volume_Ratio'
]

# Risk Management
DEFAULT_RISK_MULTIPLIER = 2.0
MAX_RISK_PER_TRADE = 0.02
POSITION_SIZE_METHOD = 'kelly'

# Application Configuration
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
DEBUG_MODE = False
CACHE_TIMEOUT = 30

# Dashboard Configuration
AUTO_REFRESH_INTERVAL = 30000  # milliseconds
MAX_DATA_POINTS = 500
CHART_THEME = 'dark'

# File Paths
DATA_DIR = 'data'
MODEL_DIR = 'model'
TEMPLATES_DIR = 'templates'
STATIC_DIR = 'static'

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_dashboard.log'
