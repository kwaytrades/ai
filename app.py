from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
import traceback
import logging
import numpy as np
import re
from datetime import datetime, timedelta
import requests
import os
from typing import Dict, List, Optional, Tuple, Any


# Import Redis cache manager
from cache_manager import get_cached_analysis, cache_analysis, cache_manager, invalidate_ticker_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ADD THIS ENTIRE SECTION HERE ↓

import time
from collections import defaultdict, deque
from threading import Lock

# Global metrics storage
class MetricsCollector:
    def __init__(self):
        self.lock = Lock()
        self.start_time = datetime.now()
        
        # Request metrics
        self.total_requests = 0
        self.requests_by_endpoint = defaultdict(int)
        self.requests_by_ticker = defaultdict(int)
        self.recent_requests = deque(maxlen=100)  # Last 100 requests
        
        # Performance metrics
        self.response_times = defaultdict(list)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Error tracking
        self.errors_by_endpoint = defaultdict(int)
        self.recent_errors = deque(maxlen=50)
        
        # Popular tickers tracking
        self.ticker_request_count = defaultdict(int)
        
    def record_request(self, endpoint: str, ticker: str = None, response_time: float = 0, cache_status: str = None, error: bool = False):
        with self.lock:
            self.total_requests += 1
            self.requests_by_endpoint[endpoint] += 1
            
            if ticker:
                self.requests_by_ticker[ticker.upper()] += 1
                self.ticker_request_count[ticker.upper()] += 1
            
            # Record recent request
            request_info = {
                "timestamp": datetime.now().isoformat(),
                "endpoint": endpoint,
                "ticker": ticker,
                "response_time_ms": round(response_time * 1000, 2),
                "cache_status": cache_status,
                "error": error
            }
            self.recent_requests.append(request_info)
            
            # Performance tracking
            if response_time > 0:
                self.response_times[endpoint].append(response_time)
                # Keep only last 50 response times per endpoint
                if len(self.response_times[endpoint]) > 50:
                    self.response_times[endpoint] = self.response_times[endpoint][-50:]
            
            # Cache tracking
            if cache_status == "hit":
                self.cache_hits += 1
            elif cache_status == "miss":
                self.cache_misses += 1
            
            # Error tracking
            if error:
                self.errors_by_endpoint[endpoint] += 1
                error_info = {
                    "timestamp": datetime.now().isoformat(),
                    "endpoint": endpoint,
                    "ticker": ticker
                }
                self.recent_errors.append(error_info)
    
    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            uptime = datetime.now() - self.start_time
            
            # Calculate average response times
            avg_response_times = {}
            for endpoint, times in self.response_times.items():
                if times:
                    avg_response_times[endpoint] = round(sum(times) / len(times) * 1000, 2)  # Convert to ms
            
            # Get top tickers
            top_tickers = sorted(
                self.ticker_request_count.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # Calculate requests per minute
            total_minutes = max(uptime.total_seconds() / 60, 1)
            requests_per_minute = round(self.total_requests / total_minutes, 2)
            
            # Cache hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = round((self.cache_hits / total_cache_requests * 100), 2) if total_cache_requests > 0 else 0
            
            return {
                "uptime": {
                    "seconds": int(uptime.total_seconds()),
                    "formatted": str(uptime).split('.')[0]  # Remove microseconds
                },
                "requests": {
                    "total": self.total_requests,
                    "per_minute": requests_per_minute,
                    "by_endpoint": dict(self.requests_by_endpoint),
                    "recent": list(self.recent_requests)[-10:]  # Last 10 requests
                },
                "performance": {
                    "avg_response_times_ms": avg_response_times,
                    "cache_hit_rate": cache_hit_rate,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses
                },
                "tickers": {
                    "top_requested": top_tickers,
                    "unique_count": len(self.ticker_request_count)
                },
                "errors": {
                    "by_endpoint": dict(self.errors_by_endpoint),
                    "recent": list(self.recent_errors)[-5:]  # Last 5 errors
                }
            }

# Global metrics instance
metrics = MetricsCollector()

# Middleware to track requests
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        response_time = time.time() - request.start_time
        
        # Extract ticker from request args
        ticker = request.args.get('ticker', None)
        
        # Get cache status from response (if it's JSON)
        cache_status = None
        try:
            if response.content_type == 'application/json':
                response_data = response.get_json()
                if response_data and isinstance(response_data, dict):
                    cache_status = response_data.get('cache_status')
        except:
            pass
        
        # Record metrics
        metrics.record_request(
            endpoint=request.endpoint or request.path,
            ticker=ticker,
            response_time=response_time,
            cache_status=cache_status,
            error=response.status_code >= 400
        )
    
    return response


# Configuration
class Config:
    DEFAULT_INTERVAL = '1d'
    DEFAULT_PERIOD = '1mo'
    MIN_DATA_POINTS = 20
    GAP_THRESHOLD_PERCENT = 1.0
    PIVOT_WINDOW = 5
    MAX_SUPPORT_RESISTANCE_LEVELS = 5
    
    # Technical indicator parameters
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    EMA_PERIODS = [20, 50, 200]
    BBANDS_PERIOD = 20
    BBANDS_STD = 2.0
    ATR_PERIOD = 14

config = Config()

# ---------------------------
# EODHD Data Provider
# ---------------------------

class EODHDClient:
    def __init__(self):
        self.api_token = os.environ.get('EODHD_API_TOKEN', '687129905386e2.08237337')
        self.base_url = "https://eodhd.com/api"
        logger.info(f"EODHD client initialized")
    
    def get_historical_data(self, ticker: str, period: str = "1mo") -> Tuple[pd.DataFrame, bool]:
        """Get historical OHLCV data from EODHD."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            period_days = {
                '1d': 2, '5d': 7, '1mo': 35, '3mo': 95, 
                '6mo': 185, '1y': 370, '2y': 735
            }
            days = period_days.get(period, 35)
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/eod/{ticker}.US"
            params = {
                'api_token': self.api_token,
                'fmt': 'json',
                'from': start_date,
                'to': end_date
            }
            
            logger.info(f"Fetching EODHD data for {ticker}")
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if not data:
                    return pd.DataFrame(), False
                
                df_data = []
                for item in data:
                    try:
                        df_data.append({
                            'Open': float(item['open']),
                            'High': float(item['high']),
                            'Low': float(item['low']),
                            'Close': float(item['adjusted_close']),
                            'Volume': int(item['volume']) if item['volume'] else 0
                        })
                    except (KeyError, ValueError, TypeError):
                        continue
                
                if not df_data:
                    return pd.DataFrame(), False
                
                dates = pd.to_datetime([item['date'] for item in data[:len(df_data)]])
                df = pd.DataFrame(df_data, index=dates)
                df.sort_index(inplace=True)
                
                return df, True
            else:
                return pd.DataFrame(), False
                
        except Exception as e:
            logger.error(f"EODHD fetch failed for {ticker}: {str(e)}")
            return pd.DataFrame(), False

# Initialize EODHD client
eodhd = EODHDClient()

# ---------------------------
# Validation Functions
# ---------------------------

def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """Validate ticker symbol format."""
    if not ticker or not isinstance(ticker, str):
        return False, "Ticker must be a non-empty string"
    
    ticker = ticker.strip().upper()
    
    if not re.match(r'^[A-Z0-9.\-]+$', ticker):
        return False, "Ticker contains invalid characters"
    
    if len(ticker) > 10:
        return False, "Ticker symbol too long"
        
    return True, ticker

def validate_interval(interval: str) -> Tuple[bool, str]:
    """Validate interval parameter."""
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    if interval not in valid_intervals:
        return False, f"Invalid interval. Must be one of: {', '.join(valid_intervals)}"
    
    return True, interval

def validate_period(period: str) -> Tuple[bool, str]:
    """Validate period parameter."""
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    
    if period not in valid_periods:
        return False, f"Invalid period. Must be one of: {', '.join(valid_periods)}"
    
    return True, period

def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate OHLCV data quality."""
    if df.empty:
        return False, "Empty dataset"
        
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    invalid_high_low = (df['High'] < df['Low']).any()
    if invalid_high_low:
        return False, "Invalid data: High < Low detected"
        
    invalid_close = ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).any()
    if invalid_close:
        return False, "Invalid data: Close outside High/Low range"
        
    negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).any().any()
    if negative_prices:
        return False, "Invalid data: Negative prices detected"
    
    if len(df) < 5:
        return False, f"Insufficient data points. Got {len(df)}, minimum required: 5"
        
    return True, "Data validation passed"

# ---------------------------
# Data Fetching with Redis Cache
# ---------------------------

def fetch_ohlcv_robust(ticker: str, interval: str = '1d', period: str = '1mo') -> Tuple[pd.DataFrame, str]:
    """Production data fetching with Redis caching - NO FALLBACK TO SAMPLE DATA."""
    
    # Check Redis cache first
    cached_result = get_cached_analysis(ticker, interval, period)
    if cached_result is not None:
        logger.info(f"Redis cache HIT for {ticker}")
        # Reconstruct DataFrame from cached data
        if 'ohlcv_data' in cached_result:
            df_data = cached_result['ohlcv_data']
            df = pd.DataFrame.from_dict({pd.to_datetime(k): v for k, v in df_data.items()}, orient='index')
            df.sort_index(inplace=True)
            return df, cached_result.get('data_source', 'cached')
    
    try:
        # Method 1: Try EODHD (real market data ONLY)
        logger.info(f"Attempting EODHD fetch for {ticker}")
        eodhd_data, eodhd_success = eodhd.get_historical_data(ticker, period)
        
        if eodhd_success and not eodhd_data.empty:
            logger.info(f"EODHD SUCCESS: Got {len(eodhd_data)} rows for {ticker}")
            
            data_valid, validation_msg = validate_ohlcv_data(eodhd_data)
            if data_valid:
                # Cache raw OHLCV data for later use
                cache_ohlcv_data = {
                    'ohlcv_data': {k.isoformat(): v for k, v in eodhd_data.to_dict('index').items()},
                    'data_source': 'eodhd',
                    'cached_at': datetime.now().isoformat()
                }
                cache_analysis(ticker, interval, period, cache_ohlcv_data)
                
                return eodhd_data, "eodhd"
            else:
                logger.error(f"EODHD data validation failed for {ticker}: {validation_msg}")
                return pd.DataFrame(), "validation_failed"
        
        # NO FALLBACK - Return empty DataFrame to indicate failure
        logger.error(f"EODHD failed for {ticker} - no fallback data")
        return pd.DataFrame(), "eodhd_failed"
        
    except Exception as e:
        logger.error(f"Critical error in fetch_ohlcv_robust for {ticker}: {str(e)}")
        return pd.DataFrame(), "error"

# ---------------------------
# Technical Analysis Functions
# ---------------------------

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using pure pandas."""
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=prices.index, dtype=float)

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate EMA using pure pandas."""
    try:
        return prices.ewm(span=period, adjust=False).mean()
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        return pd.Series(index=prices.index, dtype=float)

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[pd.DataFrame]:
    """Calculate MACD using pure pandas."""
    try:
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD_12_26_9': macd_line,
            'MACDs_12_26_9': signal_line,
            'MACDh_12_26_9': histogram
        })
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return None

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Optional[pd.DataFrame]:
    """Calculate Bollinger Bands using pure pandas."""
    try:
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        
        return pd.DataFrame({
            'BBU_20_2.0': upper_band,
            'BBM_20_2.0': rolling_mean,
            'BBL_20_2.0': lower_band
        })
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        return None

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ATR using pure pandas."""
    try:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        return pd.Series(index=high.index, dtype=float)

def calculate_daily_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    try:
        if df.empty or 'Volume' not in df.columns or df['Volume'].sum() == 0:
            return pd.Series(index=df.index, dtype=float)
        
        volume_sum = df['Volume'].cumsum()
        price_volume_sum = (df['Close'] * df['Volume']).cumsum()
        vwap = price_volume_sum / volume_sum
        
        return vwap.ffill()
        
    except Exception as e:
        logger.error(f"Error calculating VWAP: {str(e)}")
        return pd.Series(index=df.index, dtype=float)

def find_pivot_levels(df: pd.DataFrame, window: int = None, num_levels: int = None) -> Dict[str, List[float]]:
    """Find support and resistance levels using pivot point analysis."""
    if window is None:
        window = min(config.PIVOT_WINDOW, len(df) // 4)
    if num_levels is None:
        num_levels = min(config.MAX_SUPPORT_RESISTANCE_LEVELS, 3)
        
    if len(df) < window * 2 + 1:
        return {"support_levels": [], "resistance_levels": []}
    
    try:
        # Find pivot highs (resistance levels)
        pivot_high_conditions = []
        for i in range(1, window + 1):
            pivot_high_conditions.append(df['High'].shift(i) < df['High'])
            pivot_high_conditions.append(df['High'].shift(-i) < df['High'])
        
        if pivot_high_conditions:
            pivot_high_mask = pd.concat(pivot_high_conditions, axis=1).all(axis=1)
            pivot_highs = df.loc[pivot_high_mask, 'High'].dropna()
        else:
            pivot_highs = pd.Series(dtype=float)
        
        # Find pivot lows (support levels)
        pivot_low_conditions = []
        for i in range(1, window + 1):
            pivot_low_conditions.append(df['Low'].shift(i) > df['Low'])
            pivot_low_conditions.append(df['Low'].shift(-i) > df['Low'])
        
        if pivot_low_conditions:
            pivot_low_mask = pd.concat(pivot_low_conditions, axis=1).all(axis=1)
            pivot_lows = df.loc[pivot_low_mask, 'Low'].dropna()
        else:
            pivot_lows = pd.Series(dtype=float)
        
        # Get most significant levels
        current_price = df['Close'].iloc[-1]
        
        if not pivot_highs.empty:
            resistance_levels = sorted(
                pivot_highs[pivot_highs > current_price].nsmallest(num_levels),
                reverse=True
            )
        else:
            resistance_levels = []
        
        if not pivot_lows.empty:
            support_levels = sorted(
                pivot_lows[pivot_lows < current_price].nlargest(num_levels)
            )
        else:
            support_levels = []
        
        return {
            "support_levels": [round(float(s), 2) for s in support_levels],
            "resistance_levels": [round(float(r), 2) for r in resistance_levels]
        }
        
    except Exception as e:
        logger.error(f"Error finding pivot levels: {str(e)}")
        return {"support_levels": [], "resistance_levels": []}

def detect_price_gaps(df: pd.DataFrame, threshold_percent: float = None) -> List[Dict[str, Any]]:
    """Detect significant price gaps."""
    if threshold_percent is None:
        threshold_percent = config.GAP_THRESHOLD_PERCENT
        
    if df.empty or len(df) < 2:
        return []

    try:
        prev_close = df['Close'].shift(1)
        gap_percent = (df['Open'] - prev_close) / prev_close * 100
        
        significant_mask = (abs(gap_percent) > threshold_percent) & gap_percent.notna()
        
        if not significant_mask.any():
            return []
            
        significant_gaps = df[significant_mask].copy()
        significant_gaps['gap_percent'] = gap_percent[significant_mask]
        
        gaps = []
        for index, row in significant_gaps.tail(5).iterrows():
            try:
                gap_info = {
                    "datetime": index.isoformat(),
                    "date": str(index.date()),
                    "type": "gap_up" if row['gap_percent'] > 0 else "gap_down",
                    "gap_percent": round(row['gap_percent'], 2),
                    "prev_close": round(prev_close.loc[index], 2),
                    "open": round(row['Open'], 2),
                    "gap_size": round(abs(row['Open'] - prev_close.loc[index]), 2)
                }
                
                if hasattr(index, 'time'):
                    gap_info["time"] = str(index.time())
                    
                gaps.append(gap_info)
            except Exception:
                continue
        
        gaps.sort(key=lambda x: x['datetime'], reverse=True)
        return gaps
        
    except Exception as e:
        logger.error(f"Error detecting gaps: {str(e)}")
        return []

def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all technical indicators with robust error handling."""
    if df.empty:
        return {}
    
    indicators = {}
    
    try:
        # RSI
        if len(df) >= config.RSI_PERIOD:
            try:
                rsi_series = calculate_rsi(df['Close'], config.RSI_PERIOD)
                if rsi_series is not None and not rsi_series.empty:
                    rsi_value = rsi_series.iloc[-1]
                    if pd.notna(rsi_value):
                        indicators['rsi'] = {
                            'value': round(float(rsi_value), 2),
                            'signal': 'overbought' if rsi_value > 70 else 'oversold' if rsi_value < 30 else 'neutral'
                        }
            except Exception as e:
                logger.error(f"Error calculating RSI: {str(e)}")
        
        # MACD
        if len(df) >= config.MACD_SLOW:
            try:
                macd_data = calculate_macd(df['Close'], config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
                
                if macd_data is not None and not macd_data.empty:
                    latest_macd = macd_data.iloc[-1]
                    
                    macd_val = latest_macd['MACD_12_26_9']
                    signal_val = latest_macd['MACDs_12_26_9']
                    hist_val = latest_macd['MACDh_12_26_9']
                    
                    if all(pd.notna([macd_val, signal_val, hist_val])):
                        indicators['macd'] = {
                            'macd': round(float(macd_val), 4),
                            'signal': round(float(signal_val), 4),
                            'histogram': round(float(hist_val), 4),
                            'trend': 'bullish' if macd_val > signal_val else 'bearish'
                        }
            except Exception as e:
                logger.error(f"Error calculating MACD: {str(e)}")
        
        # EMAs
        ema_data = {}
        for period in config.EMA_PERIODS:
            if len(df) >= period:
                try:
                    ema_series = calculate_ema(df['Close'], period)
                    if ema_series is not None and not ema_series.empty:
                        ema_value = ema_series.iloc[-1]
                        if pd.notna(ema_value):
                            ema_data[f'ema_{period}'] = round(float(ema_value), 2)
                except Exception as e:
                    logger.error(f"Error calculating EMA {period}: {str(e)}")
        
        if ema_data:
            indicators['ema'] = ema_data
        
        # Bollinger Bands
        if len(df) >= config.BBANDS_PERIOD:
            try:
                bbands = calculate_bollinger_bands(df['Close'], config.BBANDS_PERIOD, config.BBANDS_STD)
                
                if bbands is not None and not bbands.empty:
                    latest_bb = bbands.iloc[-1]
                    
                    upper = latest_bb['BBU_20_2.0']
                    middle = latest_bb['BBM_20_2.0']
                    lower = latest_bb['BBL_20_2.0']
                    
                    if all(pd.notna([upper, middle, lower])):
                        current_price = df['Close'].iloc[-1]
                        bb_position = ((current_price - lower) / (upper - lower)) * 100
                        
                        indicators['bollinger_bands'] = {
                            'upper': round(float(upper), 2),
                            'middle': round(float(middle), 2),
                            'lower': round(float(lower), 2),
                            'bb_position': round(bb_position, 2),
                            'signal': 'overbought' if bb_position > 80 else 'oversold' if bb_position < 20 else 'neutral'
                        }
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        
        # ATR
        if len(df) >= config.ATR_PERIOD:
            try:
                atr_series = calculate_atr(df['High'], df['Low'], df['Close'], config.ATR_PERIOD)
                if atr_series is not None and not atr_series.empty:
                    atr_value = atr_series.iloc[-1]
                    if pd.notna(atr_value):
                        indicators['atr'] = {
                            'value': round(float(atr_value), 4),
                            'volatility': 'high' if atr_value > df['Close'].iloc[-1] * 0.02 else 'low'
                        }
            except Exception as e:
                logger.error(f"Error calculating ATR: {str(e)}")
        
        # VWAP
        try:
            vwap_series = calculate_daily_vwap(df)
            if not vwap_series.empty:
                vwap_value = vwap_series.iloc[-1]
                if pd.notna(vwap_value):
                    current_price = df['Close'].iloc[-1]
                    indicators['vwap'] = {
                        'value': round(float(vwap_value), 2),
                        'signal': 'bullish' if current_price > vwap_value else 'bearish'
                    }
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
    
    return indicators

def generate_trading_signals(data: Dict[str, Any], indicators: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate comprehensive trading signals."""
    signals = []
    
    try:
        current_price = data.get('last_price')
        if current_price is None:
            return [{"type": "error", "message": "No current price available"}]
        
        # RSI Signals
        rsi_data = indicators.get('rsi')
        if rsi_data:
            rsi_val = rsi_data['value']
            if rsi_val > 70:
                signals.append({
                    "type": "warning",
                    "indicator": "RSI",
                    "message": f"Overbought condition (RSI: {rsi_val:.1f}). Consider selling.",
                    "strength": "strong" if rsi_val > 80 else "moderate"
                })
            elif rsi_val < 30:
                signals.append({
                    "type": "opportunity", 
                    "indicator": "RSI",
                    "message": f"Oversold condition (RSI: {rsi_val:.1f}). Potential buying opportunity.",
                    "strength": "strong" if rsi_val < 20 else "moderate"
                })
        
        # MACD Signals
        macd_data = indicators.get('macd')
        if macd_data:
            if macd_data['trend'] == 'bullish' and macd_data['histogram'] > 0:
                signals.append({
                    "type": "bullish",
                    "indicator": "MACD",
                    "message": "Bullish momentum confirmed by MACD crossover.",
                    "strength": "moderate"
                })
            elif macd_data['trend'] == 'bearish' and macd_data['histogram'] < 0:
                signals.append({
                    "type": "bearish",
                    "indicator": "MACD", 
                    "message": "Bearish momentum confirmed by MACD crossover.",
                    "strength": "moderate"
                })
        
        # EMA Trend Signals
        ema_data = indicators.get('ema', {})
        if 'ema_20' in ema_data and 'ema_50' in ema_data:
            ema_20, ema_50 = ema_data['ema_20'], ema_data['ema_50']
            
            if current_price > ema_20 > ema_50:
                signals.append({
                    "type": "bullish",
                    "indicator": "EMA",
                    "message": "Strong uptrend: Price above EMA-20 above EMA-50.",
                    "strength": "strong"
                })
            elif current_price < ema_20 < ema_50:
                signals.append({
                    "type": "bearish",
                    "indicator": "EMA",
                    "message": "Strong downtrend: Price below EMA-20 below EMA-50.",
                    "strength": "strong"
                })
        
        # VWAP Signals
        vwap_data = indicators.get('vwap')
        if vwap_data:
            vwap_val = vwap_data['value']
            if current_price > vwap_val:
                signals.append({
                    "type": "bullish",
                    "indicator": "VWAP",
                    "message": f"Price above VWAP ({vwap_val:.2f}). Bullish intraday sentiment.",
                    "strength": "moderate"
                })
            else:
                signals.append({
                    "type": "bearish",
                    "indicator": "VWAP",
                    "message": f"Price below VWAP ({vwap_val:.2f}). Bearish intraday sentiment.",
                    "strength": "moderate"
                })
        
        # Support/Resistance Signals
        support_levels = data.get('support_levels', [])
        resistance_levels = data.get('resistance_levels', [])
        
        for support in support_levels:
            if abs(current_price - support) / support * 100 < 2:  # Within 2%
                signals.append({
                    "type": "opportunity",
                    "indicator": "Support",
                    "message": f"Price near support level ({support:.2f}). Potential bounce.",
                    "strength": "moderate"
                })
                break
        
        for resistance in resistance_levels:
            if abs(current_price - resistance) / resistance * 100 < 2:  # Within 2%
                signals.append({
                    "type": "warning",
                    "indicator": "Resistance", 
                    "message": f"Price near resistance level ({resistance:.2f}). Potential reversal.",
                    "strength": "moderate"
                })
                break
        
        # Bollinger Bands Signals
        bb_data = indicators.get('bollinger_bands')
        if bb_data:
            bb_pos = bb_data['bb_position']
            if bb_pos > 80:
                signals.append({
                    "type": "warning",
                    "indicator": "Bollinger Bands",
                    "message": f"Price in upper band ({bb_pos:.1f}%). Potential reversal zone.",
                    "strength": "moderate"
                })
            elif bb_pos < 20:
                signals.append({
                    "type": "opportunity",
                    "indicator": "Bollinger Bands", 
                    "message": f"Price in lower band ({bb_pos:.1f}%). Potential bounce zone.",
                    "strength": "moderate"
                })
        
        # Gap Analysis
        gaps = data.get('gaps', [])
        recent_gaps = [g for g in gaps if 
                      (datetime.now() - datetime.fromisoformat(g['datetime'].replace('Z', '+00:00'))).days <= 5]
        
        for gap in recent_gaps[:2]:
            if gap['type'] == 'gap_up' and gap['gap_percent'] > 3:
                signals.append({
                    "type": "info",
                    "indicator": "Gap Analysis",
                    "message": f"Recent gap up ({gap['gap_percent']:.1f}%). Watch for gap fill.",
                    "strength": "weak"
                })
            elif gap['type'] == 'gap_down' and gap['gap_percent'] < -3:
                signals.append({
                    "type": "info", 
                    "indicator": "Gap Analysis",
                    "message": f"Recent gap down ({gap['gap_percent']:.1f}%). Watch for gap fill.",
                    "strength": "weak"
                })
        
        if not signals:
            signals.append({
                "type": "neutral",
                "indicator": "Overall",
                "message": "No strong signals detected. Market appears neutral.",
                "strength": "neutral"
            })
    
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        signals.append({
            "type": "error",
            "indicator": "System",
            "message": "Error generating trading signals.",
            "strength": "unknown"
        })
    
    return signals

def analyze_ticker(ticker: str, interval: str = '1d', period: str = '1mo') -> Dict[str, Any]:
    """Complete technical analysis with Redis caching - REAL DATA ONLY."""
    
    # Validate inputs
    ticker_valid, ticker_result = validate_ticker(ticker)
    if not ticker_valid:
        return {"error": ticker_result}
    ticker = ticker_result
    
    interval_valid, interval_result = validate_interval(interval)
    if not interval_valid:
        return {"error": interval_result}
    
    period_valid, period_result = validate_period(period)
    if not period_valid:
        return {"error": period_result}
    
    # Fetch data with Redis caching
    df, source = fetch_ohlcv_robust(ticker, interval, period)
    
    # Handle data fetch failures
    if df.empty:
        error_messages = {
            "eodhd_failed": f"Unable to fetch real market data for {ticker} from EODHD. Please verify ticker symbol and try again.",
            "validation_failed": f"Data validation failed for {ticker}. The retrieved data appears to be corrupted or incomplete.",
            "error": f"Technical error occurred while fetching data for {ticker}. Please try again later."
        }
        
        error_msg = error_messages.get(source, f"No real market data available for {ticker}")
        return {
            "error": error_msg,
            "ticker": ticker,
            "data_source_attempted": "eodhd",
            "suggestion": "Verify ticker symbol is correct and that the market is open. For non-US stocks, try adding the appropriate exchange suffix (e.g., 'TICKER.L' for London)."
        }
    
    # Only proceed if we have REAL data
    if source not in ["eodhd", "cached"]:
        return {
            "error": f"Only real market data is accepted. Source '{source}' is not permitted.",
            "ticker": ticker
        }
    
    # Validate data quality
    data_valid, validation_msg = validate_ohlcv_data(df)
    if not data_valid:
        return {
            "error": f"Data quality validation failed: {validation_msg}",
            "ticker": ticker,
            "data_source": source
        }
    
    try:
        # Calculate gaps BEFORE any data modification
        gaps = detect_price_gaps(df)
        
        # Find support/resistance levels
        pivot_levels = find_pivot_levels(df)
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(df)
        
        # Get latest price data
        latest = df.iloc[-1]
        
        # Build response - REAL DATA CONFIRMED
        response_data = {
            "ticker": ticker,
            "interval": interval,
            "period": period,
            "data_source": "eodhd" if source == "cached" else source,  # Show original source
            "cache_status": "hit" if source == "cached" else "miss",
            "last_updated": datetime.now().isoformat(),
            "data_points": len(df),
            "last_price": round(float(latest['Close']), 2),
            "price_change": {
                "amount": round(float(latest['Close'] - df['Close'].iloc[-2]), 2) if len(df) > 1 else 0,
                "percent": round(float((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100), 2) if len(df) > 1 else 0
            },
            "volume": int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
            "support_levels": pivot_levels["support_levels"],
            "resistance_levels": pivot_levels["resistance_levels"],
            "gaps": gaps,
            "technical_indicators": indicators,
            "data_quality": "real_market_data"  # Explicit confirmation
        }
        
        # Generate trading signals
        signals = generate_trading_signals(response_data, indicators)
        response_data["signals"] = signals
        response_data["signal_summary"] = {
            "total_signals": len(signals),
            "bullish": len([s for s in signals if s.get('type') == 'bullish']),
            "bearish": len([s for s in signals if s.get('type') == 'bearish']),
            "neutral": len([s for s in signals if s.get('type') == 'neutral']),
            "warnings": len([s for s in signals if s.get('type') == 'warning']),
            "opportunities": len([s for s in signals if s.get('type') == 'opportunity'])
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        return {
            "error": f"Technical analysis failed for {ticker}: {str(e)}",
            "data_source": source,
            "ticker": ticker
        }

# ---------------------------
# API Endpoints (Updated with Redis Cache Management)
# ---------------------------

@app.route("/", methods=["GET"])
def root():
    """Health check endpoint."""
    return jsonify({
        "service": "Technical Analysis Microservice - Production (Redis Cache)",
        "status": "healthy",
        "version": "2.4.0-redis-cache",
        "timestamp": datetime.now().isoformat(),
        "data_policy": "REAL MARKET DATA ONLY - No sample/demo data",
        "data_source": "EODHD Professional Market Data",
        "caching": "Redis with market-aware TTL",
        "features": [
            "RSI", "MACD", "EMA", "Bollinger Bands", "VWAP", "ATR",
            "Support/Resistance Levels",
            "Gap Analysis", "Trading Signals",
            "Smart Redis Caching"
        ],
        "cache_status": cache_manager.get_cache_stats(),
        "endpoints": {
            "analysis": "/analysis?ticker=SYMBOL&interval=1d&period=1mo",
            "signals": "/signals?ticker=SYMBOL",
            "test_eodhd": "/test-eodhd?ticker=SYMBOL",
            "cache": "/cache/stats",
            "cache_clear": "/cache/clear (POST)",
            "cache_invalidate": "/cache/invalidate/TICKER (DELETE)",
            "health": "/health"
        }
    })

@app.route("/analysis", methods=["GET"])
def analysis_endpoint():
    """Main technical analysis endpoint with Redis caching."""
    ticker = request.args.get("ticker", "AAPL").strip()
    interval = request.args.get("interval", config.DEFAULT_INTERVAL)
    period = request.args.get("period", config.DEFAULT_PERIOD)
    
    try:
        result = analyze_ticker(ticker, interval, period)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in analysis endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route("/signals", methods=["GET"])
def signals_endpoint():
    """Simplified signals-only endpoint."""
    ticker = request.args.get("ticker", "AAPL").strip()
    interval = request.args.get("interval", config.DEFAULT_INTERVAL)
    period = request.args.get("period", config.DEFAULT_PERIOD)
    
    try:
        result = analyze_ticker(ticker, interval, period)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "ticker": result["ticker"],
            "data_source": result["data_source"],
            "cache_status": result.get("cache_status", "unknown"),
            "last_price": result["last_price"],
            "support_levels": result["support_levels"],
            "resistance_levels": result["resistance_levels"],
            "signals": result["signals"],
            "signal_summary": result["signal_summary"],
            "last_updated": result["last_updated"]
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in signals endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error", 
            "details": str(e)
        }), 500

@app.route("/cache/stats", methods=["GET"])
def cache_stats():
    """Get comprehensive Redis cache statistics."""
    return jsonify(cache_manager.get_cache_stats())

@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear all Redis cache."""
    try:
        if cache_manager.redis_client:
            cache_manager.redis_client.flushdb()
            return jsonify({
                "message": "Redis cache cleared successfully",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Redis not available"}), 503
    except Exception as e:
        return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500

@app.route("/cache/invalidate/<ticker>", methods=["DELETE"])
def invalidate_cache(ticker):
    """Invalidate cache for specific ticker."""
    success = invalidate_ticker_cache(ticker.upper())
    return jsonify({
        "success": success,
        "ticker": ticker.upper(),
        "message": f"Cache {'invalidated' if success else 'not found'} for {ticker}",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/test-eodhd", methods=["GET"])
def test_eodhd_endpoint():
    """Test EODHD connection and features."""
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    
    try:
        # Test data fetch
        data, success = eodhd.get_historical_data(ticker, "1mo")
        
        result = {
            "ticker": ticker,
            "eodhd_success": success,
            "data_points": len(data) if success else 0,
            "cache_backend": "Redis",
            "features_included": [
                "Historical OHLCV",
                "Support/Resistance Detection",
                "Technical Indicators",
                "Gap Analysis",
                "Trading Signals",
                "Redis Caching"
            ]
        }
        
        if success and not data.empty:
            # Test support/resistance calculation
            pivot_levels = find_pivot_levels(data)
            
            result.update({
                "sample_data": {
                    "latest_close": float(data['Close'].iloc[-1]),
                    "latest_volume": int(data['Volume'].iloc[-1])
                },
                "support_resistance_test": {
                    "support_levels_found": len(pivot_levels["support_levels"]),
                    "resistance_levels_found": len(pivot_levels["resistance_levels"]),
                    "support_levels": pivot_levels["support_levels"],
                    "resistance_levels": pivot_levels["resistance_levels"]
                }
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "eodhd_success": False
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Detailed health check with cache status."""
    cache_stats = cache_manager.get_cache_stats()
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.4.0-redis-cache",
        "data_policy": "REAL MARKET DATA ONLY",
        "features": {
            "support_resistance": True,
            "technical_indicators": True,
            "gap_analysis": True,
            "trading_signals": True,
            "redis_caching": True,
            "market_aware_ttl": True,
            "popular_ticker_optimization": True,
            "real_data_only": True,
            "sample_data_fallback": False
        },
        "data_source": "EODHD Professional Feed",
        "cache_backend": {
            "type": "Redis",
            "status": cache_stats.get("redis_status", "unknown"),
            "market_aware": True
        },
        "error_policy": "Returns errors when real data unavailable"
    })

# Legacy endpoints
@app.route("/ta", methods=["GET"])
def legacy_ta():
    return analysis_endpoint()

@app.route("/debug", methods=["GET"])
def legacy_debug():
    return health_check()

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Production Technical Analysis Microservice - Redis Cache v2.4.0")
    logger.info("Data Policy: EODHD real market data only with Redis caching")
    app.run(host="0.0.0.0", port=port, debug=False)

@app.route("/debug/cache-write", methods=["GET"])
def debug_cache_write():
    """Debug cache write operations."""
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    
    try:
        # Test basic Redis write
        if cache_manager.redis_client:
            # Test 1: Basic Redis write
            cache_manager.redis_client.set("test_key", "test_value", ex=60)
            test_read = cache_manager.redis_client.get("test_key")
            
            # Test 2: Cache key generation
            cache_key = cache_manager.get_cache_key(ticker, "1d", "1mo")
            
            # Test 3: TTL calculation
            ttl = cache_manager.get_cache_ttl(ticker)
            
            # Test 4: Try to cache some sample data
            sample_data = {
                "test": "data",
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker
            }
            
            write_success = cache_manager.set_cached_data(ticker, "1d", "1mo", sample_data)
            
            # Test 5: Try to read it back
            cached_data = cache_manager.get_cached_data(ticker, "1d", "1mo")
            
            return jsonify({
                "redis_connected": True,
                "basic_write_test": test_read == "test_value",
                "cache_key_generated": cache_key,
                "calculated_ttl": ttl,
                "sample_write_success": write_success,
                "sample_read_back": cached_data is not None,
                "cached_data_preview": str(cached_data)[:200] if cached_data else None,
                "redis_keys_count": len(cache_manager.redis_client.keys("*")),
                "all_keys": cache_manager.redis_client.keys("*")
            })
        else:
            return jsonify({"error": "Redis not connected"})
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        })

@app.route("/debug/full-flow", methods=["GET"])  
def debug_full_flow():
    """Debug the complete cache flow with real data."""
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    
    try:
        # Step 1: Clear any existing cache for this ticker
        invalidate_ticker_cache(ticker)
        
        # Step 2: Fetch data (should be cache miss)
        df, source = fetch_ohlcv_robust(ticker, "1d", "1mo")
        
        # Step 3: Check if data was cached
        cached_after = get_cached_analysis(ticker, "1d", "1mo")
        
        # Step 4: Check Redis directly
        if cache_manager.redis_client:
            redis_keys = cache_manager.redis_client.keys("*")
            cache_key = cache_manager.get_cache_key(ticker, "1d", "1mo")
            direct_redis_get = cache_manager.redis_client.get(cache_key)
        else:
            redis_keys = []
            direct_redis_get = None
            
        return jsonify({
            "step_1_cache_cleared": True,
            "step_2_data_fetch": {
                "success": not df.empty,
                "source": source,
                "data_points": len(df) if not df.empty else 0
            },
            "step_3_cache_check": {
                "cached_data_found": cached_after is not None,
                "cached_data_type": type(cached_data).__name__ if cached_after else None
            },
            "step_4_redis_direct": {
                "total_keys_in_redis": len(redis_keys),
                "redis_keys": redis_keys,
                "direct_get_result": direct_redis_get is not None,
                "cache_key_used": cache_key if cache_manager.redis_client else None
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route("/debug/env", methods=["GET"])
def debug_env():
    """Check environment variables."""
    redis_url = os.getenv('REDIS_URL', 'NOT_SET')
    
    return jsonify({
        "redis_url_exists": redis_url != 'NOT_SET',
        "redis_url_length": len(redis_url) if redis_url != 'NOT_SET' else 0,
        "redis_url_preview": redis_url[:30] + "..." if len(redis_url) > 30 else redis_url,
        "all_env_vars": {k: v[:20] + "..." if len(str(v)) > 20 else v 
                        for k, v in os.environ.items() 
                        if 'REDIS' in k.upper() or 'EODHD' in k.upper()}
    })

@app.route("/debug/redis-connect", methods=["GET"])
def debug_redis_connect():
    """Test Redis connection with detailed error reporting."""
    import redis
    import traceback
    
    redis_url = os.getenv('REDIS_URL')
    
    results = {
        "redis_url_length": len(redis_url),
        "tests": {}
    }
    
    # Test 1: Basic connection
    try:
        r1 = redis.from_url(redis_url, decode_responses=True)
        ping1 = r1.ping()
        results["tests"]["basic_connection"] = {"success": True, "ping": ping1}
    except Exception as e:
        results["tests"]["basic_connection"] = {
            "success": False, 
            "error": str(e),
            "error_type": type(e).__name__
        }
    
    # Test 2: Connection with longer timeout
    try:
        r2 = redis.from_url(
            redis_url, 
            decode_responses=True,
            socket_connect_timeout=15,
            socket_timeout=15
        )
        ping2 = r2.ping()
        results["tests"]["long_timeout"] = {"success": True, "ping": ping2}
    except Exception as e:
        results["tests"]["long_timeout"] = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
    
    # Test 3: Try with SSL (rediss://)
    try:
        ssl_url = redis_url.replace('redis://', 'rediss://')
        r3 = redis.from_url(ssl_url, decode_responses=True)
        ping3 = r3.ping()
        results["tests"]["ssl_connection"] = {"success": True, "ping": ping3}
    except Exception as e:
        results["tests"]["ssl_connection"] = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
    
    # Test 4: Check what cache_manager shows
    results["cache_manager_status"] = {
        "redis_client_exists": cache_manager.redis_client is not None,
        "cache_manager_type": type(cache_manager.redis_client).__name__ if cache_manager.redis_client else "None"
    }
    
    return jsonify(results)

@app.route("/debug/simple-redis", methods=["GET"])
def debug_simple_redis():
    """Super simple Redis test."""
    import redis
    
    redis_url = os.getenv('REDIS_URL')
    
    try:
        # Create client
        client = redis.from_url(redis_url, decode_responses=True)
        
        # Test ping
        ping_result = client.ping()
        
        # Test simple set/get
        client.set("test_key_simple", "hello_world", ex=60)
        get_result = client.get("test_key_simple")
        
        return jsonify({
            "connection_success": True,
            "ping_result": ping_result,
            "set_get_test": get_result == "hello_world",
            "get_result": get_result
        })
        
    except Exception as e:
        return jsonify({
            "connection_success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        })
@app.route("/debug/rest-api-fixed", methods=["GET"])
def debug_rest_api_fixed():
    """Test Upstash REST API with correct format."""
    upstash_url = os.getenv('UPSTASH_REDIS_REST_URL')
    upstash_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
    
    try:
        # Test ping
        ping_response = requests.post(
            f"{upstash_url}/ping",
            headers={"Authorization": f"Bearer {upstash_token}"},
            timeout=10
        )
        
        # Test set with correct format (like the curl example)
        set_response = requests.post(
            f"{upstash_url}/setex/test_key/60",
            headers={"Authorization": f"Bearer {upstash_token}"},
            data='"test_value"',  # Quoted string like curl example
            timeout=5
        )
        
        # Test get
        get_response = requests.get(
            f"{upstash_url}/get/test_key",
            headers={"Authorization": f"Bearer {upstash_token}"},
            timeout=5
        )
        
        return jsonify({
            "token_updated": True,
            "ping_test": {
                "status_code": ping_response.status_code,
                "success": ping_response.status_code == 200,
                "response": ping_response.text[:100]
            },
            "set_test": {
                "status_code": set_response.status_code,
                "success": set_response.status_code == 200,
                "response": set_response.text[:100]
            },
            "get_test": {
                "status_code": get_response.status_code,
                "success": get_response.status_code == 200,
                "result": get_response.json() if get_response.status_code == 200 else get_response.text[:100]
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route("/debug/auth-formats", methods=["GET"])
def debug_auth_formats():
    """Test different authentication formats."""
    upstash_url = os.getenv('UPSTASH_REDIS_REST_URL')
    upstash_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
    
    if not upstash_url or not upstash_token:
        return jsonify({"error": "Credentials not set"})
    
    results = {}
    
    # Test different auth header formats
    auth_formats = [
        {"name": "Bearer", "header": f"Bearer {upstash_token}"},
        {"name": "Basic", "header": f"Basic {upstash_token}"},
        {"name": "Token", "header": f"Token {upstash_token}"},
        {"name": "Plain", "header": upstash_token}
    ]
    
    for auth_format in auth_formats:
        try:
            response = requests.post(
                f"{upstash_url}/ping",
                headers={"Authorization": auth_format["header"]},
                timeout=5
            )
            
            results[auth_format["name"]] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.text[:100] if response.text else None
            }
            
        except Exception as e:
            results[auth_format["name"]] = {
                "error": str(e)
            }
    
    return jsonify({
        "url_preview": upstash_url,
        "token_length": len(upstash_token),
        "token_preview": upstash_token[:10] + "..." + upstash_token[-10:],
        "auth_tests": results
    })

@app.route("/debug/basic-auth", methods=["GET"])
def debug_basic_auth():
    """Test Upstash with Basic Authentication."""
    upstash_url = os.getenv('UPSTASH_REDIS_REST_URL')
    upstash_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
    
    try:
        import base64
        
        # Try Basic Auth with empty username and token as password
        auth_string = base64.b64encode(f":{upstash_token}".encode()).decode()
        
        response = requests.post(
            f"{upstash_url}/ping",
            headers={"Authorization": f"Basic {auth_string}"},
            timeout=5
        )
        
        return jsonify({
            "basic_auth_test": {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.text if response.status_code != 200 else "SUCCESS"
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/debug/cache-keys", methods=["GET"])
def debug_cache_keys():
    """Debug what's actually in Redis cache."""
    try:
        if cache_manager.use_rest_api:
            # Get all keys from Redis
            response = requests.get(
                f"{cache_manager.upstash_url}/keys/*",
                headers={"Authorization": f"Bearer {cache_manager.upstash_token}"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                all_keys = result.get('result', [])
                
                # Get values for a few keys to see the data
                key_data = {}
                for key in all_keys[:5]:  # Check first 5 keys
                    get_response = requests.get(
                        f"{cache_manager.upstash_url}/get/{key}",
                        headers={"Authorization": f"Bearer {cache_manager.upstash_token}"},
                        timeout=5
                    )
                    if get_response.status_code == 200:
                        key_result = get_response.json()
                        key_data[key] = {
                            "exists": bool(key_result.get('result')),
                            "data_preview": str(key_result.get('result', ''))[:100]
                        }
                
                # Test key generation for a known ticker
                test_key = cache_manager.get_cache_key("AAPL", "1d", "1mo")
                
                return jsonify({
                    "total_keys": len(all_keys),
                    "all_keys": all_keys,
                    "key_data_sample": key_data,
                    "test_key_generation": {
                        "aapl_key": test_key,
                        "key_in_redis": test_key in all_keys
                    },
                    "cache_manager_info": {
                        "use_rest_api": cache_manager.use_rest_api,
                        "upstash_url": cache_manager.upstash_url[:30] + "..."
                    }
                })
            else:
                return jsonify({"error": f"Failed to get keys: {response.status_code}"})
                
        elif cache_manager.redis_client:
            # Direct Redis
            all_keys = cache_manager.redis_client.keys("*")
            return jsonify({
                "total_keys": len(all_keys),
                "all_keys": all_keys,
                "connection_type": "direct_redis"
            })
        else:
            return jsonify({"error": "No Redis connection"})
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route("/debug/analysis-flow", methods=["GET"])
def debug_analysis_flow():
    """Debug the complete analysis flow to find cache issues."""
    ticker = request.args.get("ticker", "DEBUGTEST").upper()
    interval = "1d"
    period = "1mo"
    
    debug_info = {
        "ticker": ticker,
        "steps": [],
        "cache_operations": [],
        "final_result": None
    }
    
    try:
        # Step 1: Check cache before analysis
        debug_info["steps"].append("1. Checking cache before analysis")
        cache_key = cache_manager.get_cache_key(ticker, interval, period)
        cached_before = cache_manager.get_cached_data(ticker, interval, period)
        
        debug_info["cache_operations"].append({
            "operation": "get_cached_data_before",
            "cache_key": cache_key,
            "result": "HIT" if cached_before else "MISS",
            "data_exists": cached_before is not None,
            "data_type": type(cached_before).__name__ if cached_before else None
        })
        
        if cached_before:
            debug_info["steps"].append("2. Cache HIT - should return cached data")
            debug_info["final_result"] = "CACHE_HIT"
            return jsonify(debug_info)
        
        # Step 2: Simulate fetching new data (don't actually call EODHD to save API calls)
        debug_info["steps"].append("2. Cache MISS - simulating new data fetch")
        
        # Create mock data similar to what EODHD returns
        mock_data = {
            'ohlcv_data': {
                '2025-07-20T00:00:00': {'Open': 150.0, 'High': 155.0, 'Low': 148.0, 'Close': 152.0, 'Volume': 1000000},
                '2025-07-21T00:00:00': {'Open': 152.0, 'High': 156.0, 'Low': 150.0, 'Close': 154.0, 'Volume': 1100000},
                '2025-07-22T00:00:00': {'Open': 154.0, 'High': 158.0, 'Low': 152.0, 'Close': 156.0, 'Volume': 1200000}
            },
            'data_source': 'mock_eodhd',
            'cached_at': datetime.now().isoformat()
        }
        
        # Step 3: Try to cache the data
        debug_info["steps"].append("3. Attempting to cache mock data")
        cache_success = cache_manager.set_cached_data(ticker, interval, period, mock_data)
        
        debug_info["cache_operations"].append({
            "operation": "set_cached_data",
            "cache_key": cache_key,
            "success": cache_success,
            "data_written": mock_data
        })
        
        # Step 4: Immediately try to read it back
        debug_info["steps"].append("4. Immediately reading back cached data")
        cached_after = cache_manager.get_cached_data(ticker, interval, period)
        
        debug_info["cache_operations"].append({
            "operation": "get_cached_data_after",
            "cache_key": cache_key,
            "result": "HIT" if cached_after else "MISS",
            "data_exists": cached_after is not None,
            "data_matches": cached_after == mock_data if cached_after else False,
            "data_read": cached_after
        })
        
        # Step 5: Check what's actually in Redis
        debug_info["steps"].append("5. Checking Redis directly")
        if cache_manager.use_rest_api:
            direct_response = requests.get(
                f"{cache_manager.upstash_url}/get/{cache_key}",
                headers={"Authorization": f"Bearer {cache_manager.upstash_token}"},
                timeout=5
            )
            
            if direct_response.status_code == 200:
                redis_result = direct_response.json()
                debug_info["cache_operations"].append({
                    "operation": "direct_redis_check",
                    "redis_response": redis_result,
                    "raw_data": redis_result.get('result'),
                    "raw_data_type": type(redis_result.get('result')).__name__
                })
            else:
                debug_info["cache_operations"].append({
                    "operation": "direct_redis_check",
                    "error": f"HTTP {direct_response.status_code}",
                    "response": direct_response.text
                })
        
        # Final assessment
        if cached_after == mock_data:
            debug_info["final_result"] = "SUCCESS - Cache working correctly"
        elif cached_after is not None:
            debug_info["final_result"] = "PARTIAL - Data cached but doesn't match"
        else:
            debug_info["final_result"] = "FAILED - Data not retrievable from cache"
        
        return jsonify(debug_info)
        
    except Exception as e:
        debug_info["steps"].append(f"ERROR: {str(e)}")
        debug_info["final_result"] = "ERROR"
        debug_info["error"] = str(e)
        debug_info["traceback"] = traceback.format_exc()
        return jsonify(debug_info)

@app.route("/debug/fetch-function", methods=["GET"])
def debug_fetch_function():
    """Debug the fetch_ohlcv_robust function specifically."""
    ticker = request.args.get("ticker", "TESTFETCH").upper()
    
    try:
        # Call the actual fetch function with debug logging
        logger.info(f"DEBUG: Starting fetch for {ticker}")
        
        df, source = fetch_ohlcv_robust(ticker, "1d", "1mo")
        
        return jsonify({
            "ticker": ticker,
            "fetch_result": {
                "data_found": not df.empty,
                "data_points": len(df) if not df.empty else 0,
                "source": source,
                "columns": list(df.columns) if not df.empty else []
            },
            "cache_check": {
                "cache_key": cache_manager.get_cache_key(ticker, "1d", "1mo"),
                "cached_data_exists": cache_manager.get_cached_data(ticker, "1d", "1mo") is not None
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route("/debug/simple-cache-test", methods=["GET"])
def debug_simple_cache_test():
    """Ultra-simple cache test with minimal data."""
    
    try:
        # Use a simple key and simple data
        test_key = "simple_test"
        simple_data = {"message": "hello", "number": 42, "timestamp": datetime.now().isoformat()}
        
        # Manual cache operations
        cache_key = f"ta:{test_key}:1d:1mo"
        
        # Write directly using cache manager
        write_result = cache_manager.set_cached_data(test_key, "1d", "1mo", simple_data)
        
        # Read directly using cache manager  
        read_result = cache_manager.get_cached_data(test_key, "1d", "1mo")
        
        # Also test the convenience functions
        convenience_write = cache_analysis(test_key, "1d", "1mo", simple_data)
        convenience_read = get_cached_analysis(test_key, "1d", "1mo")
        
        return jsonify({
            "test_data": simple_data,
            "cache_key": cache_key,
            "direct_test": {
                "write_success": write_result,
                "read_success": read_result is not None,
                "data_matches": read_result == simple_data if read_result else False,
                "read_data": read_result
            },
            "convenience_test": {
                "write_success": convenience_write,
                "read_success": convenience_read is not None,
                "data_matches": convenience_read == simple_data if convenience_read else False,
                "read_data": convenience_read
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route("/debug/test-cache-flow", methods=["GET"])
def debug_test_cache_flow():
    """Test the complete cache read/write flow."""
    ticker = request.args.get("ticker", "TEST").upper()
    
    try:
        # Step 1: Generate cache key
        cache_key = cache_manager.get_cache_key(ticker, "1d", "1mo")
        
        # Step 2: Test data to cache
        test_data = {
            "ticker": ticker,
            "test_timestamp": datetime.now().isoformat(),
            "data": {"price": 123.45, "volume": 1000000}
        }
        
        # Step 3: Write to cache
        write_success = cache_manager.set_cached_data(ticker, "1d", "1mo", test_data)
        
        # Step 4: Immediately try to read it back
        read_data = cache_manager.get_cached_data(ticker, "1d", "1mo")
        
        # Step 5: Check if key exists in Redis directly
        key_exists = False
        if cache_manager.use_rest_api:
            check_response = requests.get(
                f"{cache_manager.upstash_url}/get/{cache_key}",
                headers={"Authorization": f"Bearer {cache_manager.upstash_token}"},
                timeout=5
            )
            key_exists = check_response.status_code == 200 and check_response.json().get('result')
        
        return jsonify({
            "cache_key": cache_key,
            "write_attempt": {
                "success": write_success,
                "data_written": test_data
            },
            "read_attempt": {
                "success": read_data is not None,
                "data_read": read_data,
                "matches_written": read_data == test_data if read_data else False
            },
            "redis_direct_check": {
                "key_exists": key_exists,
                "cache_key_used": cache_key
            },
            "debug_info": {
                "ttl": cache_manager.get_cache_ttl(ticker),
                "use_rest_api": cache_manager.use_rest_api
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Get comprehensive service metrics."""
    try:
        service_metrics = metrics.get_metrics()
        cache_stats = cache_manager.get_cache_stats()
        
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "service": service_metrics,
            "cache": cache_stats,
            "system": {
                "version": "2.4.0-redis-cache",
                "data_source": "EODHD Professional",
                "cache_backend": "Upstash Redis REST API"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Serve the dashboard HTML page."""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Technical Analysis Microservice Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card h3 {
            color: #4a5568;
            margin-bottom: 15px;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 10px;
        }
        
        .stat-label {
            color: #718096;
            font-size: 0.9rem;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-connected {
            background-color: #48bb78;
            animation: pulse 2s infinite;
        }
        
        .status-error {
            background-color: #f56565;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .recent-activity {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .activity-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.9rem;
        }
        
        .activity-item:last-child {
            border-bottom: none;
        }
        
        .ticker-badge {
            background: #4299e1;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .cache-hit {
            color: #48bb78;
            font-weight: bold;
        }
        
        .cache-miss {
            color: #ed8936;
            font-weight: bold;
        }
        
        .auto-refresh {
            text-align: center;
            color: white;
            margin-top: 20px;
            opacity: 0.8;
        }
        
        .popular-tickers {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        
        .ticker-item {
            background: #f7fafc;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #4299e1;
        }
        
        .ticker-symbol {
            font-weight: bold;
            color: #2d3748;
        }
        
        .ticker-count {
            font-size: 0.8rem;
            color: #718096;
        }
        
        .error-badge {
            background: #f56565;
            color: white;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 0.7rem;
        }
        
        .loading {
            text-align: center;
            color: white;
            font-size: 1.2rem;
            margin: 50px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Technical Analysis Dashboard</h1>
            <p>Real-time monitoring of your microservice performance</p>
        </div>
        
        <div id="loading" class="loading">
            🔄 Loading dashboard data...
        </div>
        
        <div id="dashboard-content" style="display: none;">
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>🔗 Service Status</h3>
                    <div class="stat-value">
                        <span class="status-indicator" id="status-indicator"></span>
                        <span id="service-status">Loading...</span>
                    </div>
                    <div class="stat-label">
                        Uptime: <span id="uptime">--</span>
                    </div>
                </div>
                
                <div class="stat-card">
                    <h3>📈 Total Requests</h3>
                    <div class="stat-value" id="total-requests">0</div>
                    <div class="stat-label">
                        <span id="requests-per-minute">0</span> requests/min
                    </div>
                </div>
                
                <div class="stat-card">
                    <h3>⚡ Cache Performance</h3>
                    <div class="stat-value" id="cache-hit-rate">0%</div>
                    <div class="stat-label">
                        <span id="cache-hits">0</span> hits | <span id="cache-misses">0</span> misses
                    </div>
                </div>
                
                <div class="stat-card">
                    <h3>💾 Redis Status</h3>
                    <div class="stat-value" id="redis-status">Connecting...</div>
                    <div class="stat-label">
                        <span id="cache-keys">0</span> cached items
                    </div>
                </div>
            </div>
            
            <div class="recent-activity">
                <h3>🕒 Recent API Requests</h3>
                <div id="recent-requests">
                    <div class="activity-item">No recent requests</div>
                </div>
            </div>
            
            <div class="recent-activity">
                <h3>🏆 Most Requested Tickers</h3>
                <div class="popular-tickers" id="popular-tickers">
                    <div class="ticker-item">No data yet</div>
                </div>
            </div>
        </div>
        
        <div class="auto-refresh">
            🔄 Auto-refreshing every 5 seconds
        </div>
    </div>

    <script>
        let refreshInterval;
        
        async function fetchMetrics() {
            try {
                const response = await fetch('/metrics');
                const data = await response.json();
                updateDashboard(data);
                
                // Hide loading and show content
                document.getElementById('loading').style.display = 'none';
                document.getElementById('dashboard-content').style.display = 'block';
                
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
                document.getElementById('loading').innerHTML = '❌ Failed to load dashboard data';
            }
        }
        
        function updateDashboard(data) {
            // Service status
            const isHealthy = data.cache && data.cache.redis_status === 'connected';
            document.getElementById('service-status').textContent = isHealthy ? 'Online' : 'Issues Detected';
            document.getElementById('status-indicator').className = 
                'status-indicator ' + (isHealthy ? 'status-connected' : 'status-error');
            document.getElementById('uptime').textContent = data.service.uptime.formatted;
            
            // Request metrics
            document.getElementById('total-requests').textContent = data.service.requests.total.toLocaleString();
            document.getElementById('requests-per-minute').textContent = data.service.requests.per_minute;
            
            // Cache performance
            document.getElementById('cache-hit-rate').textContent = data.service.performance.cache_hit_rate + '%';
            document.getElementById('cache-hits').textContent = data.service.performance.cache_hits;
            document.getElementById('cache-misses').textContent = data.service.performance.cache_misses;
            
            // Redis status
            if (data.cache) {
                document.getElementById('redis-status').textContent = 
                    data.cache.redis_status === 'connected' ? 'Connected' : 'Disconnected';
                document.getElementById('cache-keys').textContent = 
                    data.cache.ta_cache_keys || 0;
            }
            
            // Recent requests
            updateRecentRequests(data.service.requests.recent || []);
            
            // Popular tickers
            updatePopularTickers(data.service.tickers.top_requested || []);
        }
        
        function updateRecentRequests(requests) {
            const container = document.getElementById('recent-requests');
            if (!requests.length) {
                container.innerHTML = '<div class="activity-item">No recent requests</div>';
                return;
            }
            
            container.innerHTML = requests.reverse().map(req => `
                <div class="activity-item">
                    <div>
                        <strong>${req.endpoint}</strong>
                        ${req.ticker ? `<span class="ticker-badge">${req.ticker}</span>` : ''}
                        ${req.error ? '<span class="error-badge">ERROR</span>' : ''}
                    </div>
                    <div>
                        ${req.cache_status ? `<span class="cache-${req.cache_status}">${req.cache_status.toUpperCase()}</span> |` : ''}
                        ${req.response_time_ms}ms |
                        ${new Date(req.timestamp).toLocaleTimeString()}
                    </div>
                </div>
            `).join('');
        }
        
        function updatePopularTickers(tickers) {
            const container = document.getElementById('popular-tickers');
            if (!tickers.length) {
                container.innerHTML = '<div class="ticker-item">No data yet</div>';
                return;
            }
            
            container.innerHTML = tickers.map(([ticker, count]) => `
                <div class="ticker-item">
                    <div class="ticker-symbol">${ticker}</div>
                    <div class="ticker-count">${count} requests</div>
                </div>
            `).join('');
        }
        
        // Initial load
        fetchMetrics();
        
        // Auto-refresh every 5 seconds
        refreshInterval = setInterval(fetchMetrics, 5000);
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        });
    </script>
</body>
</html>
    '''
