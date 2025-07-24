from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
import traceback
import logging
import numpy as np
import re
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import time
import requests
import os
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    CACHE_TTL_SECONDS = 300  # 5 minutes cache
    MAX_CACHE_SIZE = 128
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
# Cache Implementation (Complete)
# ---------------------------

class DataCache:
    def __init__(self, ttl_seconds: int = 300, max_size: int = 128):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache = {}
        self.timestamps = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, ticker: str, interval: str, period: str) -> str:
        """Generate cache key from parameters."""
        key_string = f"{ticker.upper()}_{interval}_{period}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, ticker: str, interval: str, period: str) -> Optional[Tuple[pd.DataFrame, str]]:
        """Get cached data if valid."""
        key = self._generate_key(ticker, interval, period)
        
        if key not in self.cache:
            self.miss_count += 1
            return None
            
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            del self.cache[key]
            del self.timestamps[key]
            self.miss_count += 1
            return None
            
        self.hit_count += 1
        logger.info(f"Cache hit for {ticker}")
        data, source = self.cache[key]
        return data.copy(), source
    
    def set(self, ticker: str, interval: str, period: str, data: pd.DataFrame, source: str):
        """Cache data with TTL."""
        key = self._generate_key(ticker, interval, period)
        
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = (data.copy(), source)
        self.timestamps[key] = time.time()
        logger.info(f"Cached data for {ticker} (source: {source})")
    
    def clear(self):
        """Clear all cache data."""
        self.cache.clear()
        self.timestamps.clear()

# Global cache instance
data_cache = DataCache(config.CACHE_TTL_SECONDS, config.MAX_CACHE_SIZE)

# ---------------------------
# Validation Functions (Complete)
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
# Sample Data Generation (Fallback)
# ---------------------------

def generate_realistic_sample_data(ticker: str, period: str = '1mo') -> pd.DataFrame:
    """Generate realistic sample data when real data fails."""
    stock_profiles = {
        'AAPL': {'base_price': 191.0, 'volatility': 0.018, 'volume_base': 80000000},
        'MSFT': {'base_price': 428.0, 'volatility': 0.016, 'volume_base': 60000000},
        'GOOGL': {'base_price': 170.0, 'volatility': 0.020, 'volume_base': 45000000},
        'TSLA': {'base_price': 248.0, 'volatility': 0.035, 'volume_base': 120000000},
        'NVDA': {'base_price': 135.0, 'volatility': 0.028, 'volume_base': 90000000}
    }
    
    profile = stock_profiles.get(ticker.upper(), {
        'base_price': 100.0, 'volatility': 0.020, 'volume_base': 50000000
    })
    
    period_map = {'1d': 1, '5d': 5, '1mo': 21, '3mo': 63, '6mo': 126, '1y': 252}
    periods = period_map.get(period, 21)
    
    end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=periods)
    dates = pd.bdate_range(start=start_date, end=end_date, freq='D')[:periods]
    
    np.random.seed(42 + hash(ticker) % 1000)
    
    base_price = profile['base_price']
    volatility = profile['volatility']
    
    returns = np.random.normal(0.0005, volatility, periods)
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 0.01))
    
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        intraday_vol = close_price * volatility * 0.7
        open_price = close_price + np.random.normal(0, intraday_vol * 0.3)
        
        high_addon = abs(np.random.exponential(intraday_vol * 0.5))
        low_subtract = abs(np.random.exponential(intraday_vol * 0.5))
        
        high = max(open_price, close_price) + high_addon
        low = min(open_price, close_price) - low_subtract
        
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        open_price = max(min(open_price, high), low)
        
        base_volume = profile['volume_base']
        volume = int(base_volume * np.random.uniform(0.7, 1.3))
        
        data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

# ---------------------------
# Main Data Fetching Function
# ---------------------------

def fetch_ohlcv_robust(ticker: str, interval: str = '1d', period: str = '1mo') -> Tuple[pd.DataFrame, str]:
    """Production data fetching with EODHD + fallback."""
    
    # Check cache first
    cached_result = data_cache.get(ticker, interval, period)
    if cached_result is not None:
        data, source = cached_result
        return data, source
    
    try:
        # Method 1: Try EODHD (real market data)
        logger.info(f"Attempting EODHD fetch for {ticker}")
        eodhd_data, eodhd_success = eodhd.get_historical_data(ticker, period)
        
        if eodhd_success and not eodhd_data.empty:
            logger.info(f"EODHD SUCCESS: Got {len(eodhd_data)} rows for {ticker}")
            
            data_valid, validation_msg = validate_ohlcv_data(eodhd_data)
            if data_valid:
                data_cache.set(ticker, interval, period, eodhd_data, "eodhd")
                return eodhd_data, "eodhd"
        
        # Method 2: Fallback to sample data
        logger.warning(f"EODHD failed for {ticker}, using sample data")
        sample_data = generate_realistic_sample_data(ticker, period)
        data_cache.set(ticker, interval, period, sample_data, "sample")
        return sample_data, "sample"
        
    except Exception as e:
        logger.error(f"Critical error in fetch_ohlcv_robust: {str(e)}")
        sample_data = generate_realistic_sample_data(ticker, period)
        return sample_data, "sample"

# ---------------------------
# Complete Technical Analysis Functions
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
        
        # Support/Resistance Signals - THIS IS THE KEY PART YOU ASKED ABOUT
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
    """Complete technical analysis for a ticker with EODHD integration."""
    
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
    
    # Fetch data
    df, source = fetch_ohlcv_robust(ticker, interval, period)
    if df.empty:
        return {"error": f"No data available for {ticker} with interval {interval} and period {period}"}
    
    # Validate data quality
    data_valid, validation_msg = validate_ohlcv_data(df)
    if not data_valid:
        return {"error": validation_msg}
    
    try:
        # Calculate gaps BEFORE any data modification
        gaps = detect_price_gaps(df)
        
        # Find support/resistance levels - THIS IS WHAT YOU ASKED ABOUT
        pivot_levels = find_pivot_levels(df)
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(df)
        
        # Get latest price data
        latest = df.iloc[-1]
        
        # Build response
        response_data = {
            "ticker": ticker,
            "interval": interval,
            "period": period,
            "data_source": source,
            "last_updated": datetime.now().isoformat(),
            "data_points": len(df),
            "last_price": round(float(latest['Close']), 2),
            "price_change": {
                "amount": round(float(latest['Close'] - df['Close'].iloc[-2]), 2) if len(df) > 1 else 0,
                "percent": round(float((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100), 2) if len(df) > 1 else 0
            },
            "volume": int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
            "support_levels": pivot_levels["support_levels"],      # HERE IS YOUR SUPPORT
            "resistance_levels": pivot_levels["resistance_levels"], # HERE IS YOUR RESISTANCE
            "gaps": gaps,
            "technical_indicators": indicators
        }
        
        # Generate trading signals (includes support/resistance signals)
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
        
        # Add disclaimer if using sample data
        if source == "sample":
            response_data["disclaimer"] = "Using sample data for demonstration. Real market data unavailable."
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}

# ---------------------------
# API Endpoints
# ---------------------------

@app.route("/", methods=["GET"])
def root():
    """Health check endpoint."""
    return jsonify({
        "service": "Technical Analysis Microservice - Production",
        "status": "healthy",
        "version": "2.3.0-eodhd-complete",
        "timestamp": datetime.now().isoformat(),
        "data_sources": {
            "primary": "EODHD (Real Market Data)",
            "fallback": "Sample Data"
        },
        "features": [
            "RSI", "MACD", "EMA", "Bollinger Bands", "VWAP", "ATR",
            "Support/Resistance Levels",  # YOUR QUESTION ANSWERED HERE
            "Gap Analysis", "Trading Signals"
        ],
        "cache_size": len(data_cache.cache),
        "endpoints": {
            "analysis": "/analysis?ticker=SYMBOL&interval=1d&period=1mo",
            "signals": "/signals?ticker=SYMBOL",
            "test_eodhd": "/test-eodhd?ticker=SYMBOL",
            "cache": "/cache/stats",
            "health": "/health"
        }
    })

@app.route("/analysis", methods=["GET"])
def analysis_endpoint():
    """Main technical analysis endpoint with complete feature set."""
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
            "last_price": result["last_price"],
            "support_levels": result["support_levels"],       # SUPPORT INCLUDED
            "resistance_levels": result["resistance_levels"], # RESISTANCE INCLUDED
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
            "features_included": [
                "Historical OHLCV",
                "Support/Resistance Detection",
                "Technical Indicators",
                "Gap Analysis",
                "Trading Signals"
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

@app.route("/cache/stats", methods=["GET"])
def cache_stats():
    """Cache statistics endpoint."""
    return jsonify({
        "cache_size": len(data_cache.cache),
        "max_cache_size": data_cache.max_size,
        "ttl_seconds": data_cache.ttl_seconds,
        "hit_count": data_cache.hit_count,
        "miss_count": data_cache.miss_count,
        "hit_rate": round(data_cache.hit_count / (data_cache.hit_count + data_cache.miss_count) * 100, 2) if (data_cache.hit_count + data_cache.miss_count) > 0 else 0
    })

@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear cache endpoint."""
    try:
        data_cache.clear()
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Detailed health check."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.3.0-eodhd-complete",
        "features": {
            "support_resistance": True,  # YOUR ANSWER
            "technical_indicators": True,
            "gap_analysis": True,
            "trading_signals": True,
            "caching": True,
            "real_data": True
        },
        "data_source": "EODHD + Sample fallback"
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
    logger.info("Starting Complete Technical Analysis Microservice with EODHD v2.3.0")
    app.run(host="0.0.0.0", port=port, debug=False)
