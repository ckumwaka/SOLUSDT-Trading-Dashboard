"""
Real-time data streaming via WebSocket connections
"""

import websocket
import json
import threading
import time
import logging
from typing import Callable, Dict, List
from collections import deque
import queue
from config import BINANCE_WS_URL, DEFAULT_SYMBOL

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self, symbol: str = DEFAULT_SYMBOL):
        self.symbol = symbol.lower()
        self.ws = None
        self.callbacks = {}
        self.data_queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        
    def start(self):
        """Start WebSocket connection"""
        if self.is_running:
            return
        
        self.is_running = True
        self.reconnect_attempts = 0
        self._connect()
    
    def stop(self):
        """Stop WebSocket connection"""
        self.is_running = False
        if self.ws:
            self.ws.close()
    
    def _connect(self):
        """Establish WebSocket connection"""
        try:
            # Create streams for real-time data
            streams = [
                f"{self.symbol}@ticker",      # 24hr ticker statistics
                f"{self.symbol}@trade",       # Trade data
                f"{self.symbol}@kline_1m",    # 1-minute klines
                f"{self.symbol}@bookTicker"   # Best bid/ask
            ]
            
            ws_url = f"{BINANCE_WS_URL}stream?streams=" + "/".join(streams)
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start in separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self._schedule_reconnect()
    
    def _on_open(self, ws):
        """Called when WebSocket connection opens"""
        logger.info(f"WebSocket connected for {self.symbol}")
        self.reconnect_attempts = 0
        
        # Notify callbacks
        if 'connection' in self.callbacks:
            self.callbacks['connection']({'status': 'connected'})
    
    def _on_message(self, ws, message):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'stream' in data:
                stream_name = data['stream']
                stream_data = data['data']
                
                # Process different stream types
                if '@ticker' in stream_name:
                    self._process_ticker_data(stream_data)
                elif '@trade' in stream_name:
                    self._process_trade_data(stream_data)
                elif '@kline' in stream_name:
                    self._process_kline_data(stream_data)
                elif '@bookTicker' in stream_name:
                    self._process_book_ticker_data(stream_data)
                
                # Add to queue for processing
                if not self.data_queue.full():
                    self.data_queue.put({
                        'stream': stream_name,
                        'data': stream_data,
                        'timestamp': time.time()
                    })
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def _process_ticker_data(self, data):
        """Process 24hr ticker statistics"""
        ticker_info = {
            'symbol': data['s'],
            'price': float(data['c']),
            'change': float(data['P']),
            'volume': float(data['v']),
            'high': float(data['h']),
            'low': float(data['l']),
            'timestamp': int(data['E'])
        }
        
        if 'ticker' in self.callbacks:
            self.callbacks['ticker'](ticker_info)
    
    def _process_trade_data(self, data):
        """Process individual trade data"""
        trade_info = {
            'symbol': data['s'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'time': int(data['T']),
            'is_buyer_maker': data['m']
        }
        
        # Update price history
        self.price_history.append(trade_info['price'])
        
        if 'trade' in self.callbacks:
            self.callbacks['trade'](trade_info)
    
    def _process_kline_data(self, data):
        """Process kline/candlestick data"""
        kline = data['k']
        kline_info = {
            'symbol': kline['s'],
            'open_time': int(kline['t']),
            'close_time': int(kline['T']),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'is_closed': kline['x']
        }
        
        # Update volume history
        if kline_info['is_closed']:
            self.volume_history.append(kline_info['volume'])
        
        if 'kline' in self.callbacks:
            self.callbacks['kline'](kline_info)
    
    def _process_book_ticker_data(self, data):
        """Process best bid/ask data"""
        book_info = {
            'symbol': data['s'],
            'best_bid': float(data['b']),
            'best_bid_qty': float(data['B']),
            'best_ask': float(data['a']),
            'best_ask_qty': float(data['A']),
            'timestamp': int(data['u'])
        }
        
        if 'book' in self.callbacks:
            self.callbacks['book'](book_info)
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        
        if 'error' in self.callbacks:
            self.callbacks['error']({'error': str(error)})
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure"""
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        if 'connection' in self.callbacks:
            self.callbacks['connection']({'status': 'disconnected'})
        
        if self.is_running:
            self._schedule_reconnect()
    
    def _schedule_reconnect(self):
        """Schedule reconnection attempt"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
            
            logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts})")
            
            def reconnect():
                time.sleep(delay)
                if self.is_running:
                    self._connect()
            
            thread = threading.Thread(target=reconnect)
            thread.daemon = True
            thread.start()
        else:
            logger.error("Max reconnection attempts reached")
            self.is_running = False
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for specific event type"""
        self.callbacks[event_type] = callback
    
    def get_latest_price(self) -> float:
        """Get latest price from history"""
        return self.price_history[-1] if self.price_history else 0.0
    
    def get_price_change(self, periods: int = 10) -> float:
        """Calculate price change over specified periods"""
        if len(self.price_history) < periods:
            return 0.0
        
        current = self.price_history[-1]
        past = self.price_history[-(periods + 1)]
        return (current - past) / past * 100
    
    def get_volume_spike(self) -> bool:
        """Detect volume spikes"""
        if len(self.volume_history) < 20:
            return False
        
        recent_volume = self.volume_history[-1]
        avg_volume = sum(list(self.volume_history)[:-1]) / (len(self.volume_history) - 1)
        
        return recent_volume > avg_volume * 1.5

# Global instance
ws_manager = WebSocketManager()
