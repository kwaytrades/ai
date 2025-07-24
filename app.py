from flask import Flask, request, jsonify
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import traceback
import logging
import re
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import time
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
# Cache Implementation
# ---------------------------

class DataCache:
    def __init__(self, ttl_seconds: int = 300, max_size: int = 128):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache = {}
        self.timestamps = {}
    
    def _generate_key(self, ticker: str, interval: str, period: str) -> str:
        """Generate cache key from parameters."""
        key_string = f"{ticker.upper()}_{interval}_{period}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
        """Get cached data if valid."""
        key = self._generate_key(ticker, interval, period)
        
        if key not in self.cache:
            return None
            
        # Check if cache entry is still valid
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            del self.cache[key]
            del self.timestamps[key]
            return None
            
        logger.info(f"Cache hit for {ticker}")
        return self.cache[key].copy()
    
    def set(self, ticker: str, interval: str, period: str, data: pd.DataFrame):
        """Cache data with TTL."""
        key = self._generate_key(ticker, interval, period)
        
        # Implement simple LRU by removing oldest entries
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = data.copy()
        self.timestamps[key] = time.time()
        logger.info(f"Cached data for {ticker}")

# Global cache instance
data_cache = DataCache(config.CACHE_TTL_SECONDS, config.MAX_CACHE_SIZE)

# ---------------------------
# Validation Functions
# ---------------------------

def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """Validate ticker symbol format."""
    if not ticker or not isinstance(ticker, str):
        return False, "Ticker must be a non-empty string"
    
    ticker = ticker.strip().upper()
    
    # Basic ticker validation (alphanumeric, dots, hyphens)
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
        
    # Check required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for basic data integrity
    invalid_high_low = (df['High'] < df['Low']).any()
    if invalid_high_low:
        return False, "Invalid data: High < Low detected"
        
    invalid_close = ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).any()
    if invalid_close:
        return False, "Invalid data: Close outside High/Low range"
        
    negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).any().any()
    if negative_prices:
        return False, "Invalid data: Negative prices detected"
    
    # Check for sufficient data points
    if len(df) < config.MIN_DATA_POINTS:
        return False, f"Insufficient data points. Got {len(df)}, minimum required: {config.MIN_DATA_POINTS}"
        
    return True, "Data validation passed"

# ---------------------------
# Data Fetching Functions
# ---------------------------

def fetch_ohlcv(ticker: str, interval: str = '1d', period: str = '1mo') -> pd.DataFrame:
    """Fetch OHLCV data with caching."""
    # Check cache first
    cached_data = data_cache.get(ticker, interval, period)
    if cached_data is not None:
        return cached_data
    
    try:
        logger.info(f"Fetching data for {ticker} with interval={interval}, period={period}")
        
        # Fetch from Yahoo Finance
        data = yf.download(
            ticker, 
            period=period, 
            interval=interval, 
            progress=False, 
            auto_adjust=True,
            prepost=False,  # Exclude pre/post market data for cleaner analysis
            threads=True
        )
        
        if data.empty:
            logger.warning(f"No data returned for ticker {ticker}")
            return pd.DataFrame()
        
        # Convert columns to numeric and handle any data type issues
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows with NaN values in essential columns
        data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        # Cache the data
        data_cache.set(ticker, interval, period, data)
        
        logger.info(f"Successfully fetched {len(data)} data points for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

# ---------------------------
# Technical Analysis Functions
# ---------------------------

def calculate_daily_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price that resets daily."""
    if df.empty or df['Volume'].sum() == 0:
        return pd.Series(index=df.index, dtype=float)
    
    df_copy = df.copy()
    df_copy['date'] = df_copy.index.date
    
    def daily_vwap(group):
        volume_sum = group['Volume'].cumsum()
        price_volume_sum = (group['Close'] * group['Volume']).cumsum()
        return price_volume_sum / volume_sum
    
    try:
        vwap_series = df_copy.groupby('date', group_keys=False).apply(daily_vwap)
        return vwap_series
    except Exception as e:
        logger.error(f"Error calculating VWAP: {str(e)}")
        return pd.Series(index=df.index, dtype=float)

def find_pivot_levels(df: pd.DataFrame, window: int = None, num_levels: int = None) -> Dict[str, List[float]]:
    """Find support and resistance levels using pivot point analysis."""
    if window is None:
        window = config.PIVOT_WINDOW
    if num_levels is None:
        num_levels = config.MAX_SUPPORT_RESISTANCE_LEVELS
        
    if len(df) < window * 2 + 1:
        return {"support_levels": [], "resistance_levels": []}
    
    try:
        # Find pivot highs (resistance levels)
        pivot_high_conditions = []
        for i in range(1, window + 1):
            pivot_high_conditions.append(df['High'].shift(i) < df['High'])
            pivot_high_conditions.append(df['High'].shift(-i) < df['High'])
        
        pivot_high_mask = pd.concat(pivot_high_conditions, axis=1).all(axis=1)
        pivot_highs = df.loc[pivot_high_mask, 'High'].dropna()
        
        # Find pivot lows (support levels)
        pivot_low_conditions = []
        for i in range(1, window + 1):
            pivot_low_conditions.append(df['Low'].shift(i) > df['Low'])
            pivot_low_conditions.append(df['Low'].shift(-i) > df['Low'])
        
        pivot_low_mask = pd.concat(pivot_low_conditions, axis=1).all(axis=1)
        pivot_lows = df.loc[pivot_low_mask, 'Low'].dropna()
        
        # Get most significant levels (closest to current price for relevance)
        current_price = df['Close'].iloc[-1]
        
        # Sort resistance levels by proximity to current price
        resistance_levels = sorted(
            pivot_highs[pivot_highs > current_price].nsmallest(num_levels),
            reverse=True
        )
        
        # Sort support levels by proximity to current price  
        support_levels = sorted(
            pivot_lows[pivot_lows < current_price].nlargest(num_levels)
        )
        
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
        
        # Filter for significant gaps
        significant_mask = abs(gap_percent) > threshold_percent
        significant_gaps = df[significant_mask].copy()
        
        if significant_gaps.empty:
            return []
            
        significant_gaps['gap_percent'] = gap_percent[significant_mask]
        
        gaps = []
        for index, row in significant_gaps.iterrows():
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
        
        # Sort by date (most recent first)
        gaps.sort(key=lambda x: x['datetime'], reverse=True)
        return gaps
        
    except Exception as e:
        logger.error(f"Error detecting gaps: {str(e)}")
        return []

def safe_get_indicator_value(df: pd.DataFrame, column_pattern: str, fallback_cols: List[str] = None) -> Optional[float]:
    """Safely extract indicator values with pattern matching."""
    if df.empty:
        return None
        
    # Try pattern matching first
    matching_cols = [col for col in df.columns if column_pattern.lower() in col.lower()]
    
    if matching_cols:
        try:
            value = df.iloc[-1][matching_cols[0]]
            return round(float(value), 4) if pd.notna(value) else None
        except (IndexError, ValueError, TypeError):
            pass
    
    # Try fallback columns
    if fallback_cols:
        for col in fallback_cols:
            if col in df.columns:
                try:
                    value = df.iloc[-1][col]
                    return round(float(value), 4) if pd.notna(value) else None
                except (IndexError, ValueError, TypeError):
                    continue
    
    return None

def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all technical indicators with robust error handling."""
    if df.empty:
        return {}
    
    indicators = {}
    
    try:
        # RSI
        if len(df) >= config.RSI_PERIOD:
            rsi_series = ta.rsi(df['Close'], length=config.RSI_PERIOD)
            rsi_value = safe_get_indicator_value(
                pd.DataFrame({'RSI': rsi_series}), 'rsi'
            )
            if rsi_value is not None:
                indicators['rsi'] = {
                    'value': rsi_value,
                    'signal': 'overbought' if rsi_value > 70 else 'oversold' if rsi_value < 30 else 'neutral'
                }
        
        # MACD
        if len(df) >= config.MACD_SLOW:
            macd_data = ta.macd(
                df['Close'], 
                fast=config.MACD_FAST, 
                slow=config.MACD_SLOW, 
                signal=config.MACD_SIGNAL
            )
            
            if macd_data is not None and not macd_data.empty:
                macd_val = safe_get_indicator_value(macd_data, 'macd', ['MACD'])
                signal_val = safe_get_indicator_value(macd_data, 'signal', ['Signal'])
                hist_val = safe_get_indicator_value(macd_data, 'hist', ['Histogram'])
                
                if all(v is not None for v in [macd_val, signal_val, hist_val]):
                    indicators['macd'] = {
                        'macd': macd_val,
                        'signal': signal_val,
                        'histogram': hist_val,
                        'trend': 'bullish' if macd_val > signal_val else 'bearish'
                    }
        
        # EMAs
        ema_data = {}
        for period in config.EMA_PERIODS:
            if len(df) >= period:
                ema_series = ta.ema(df['Close'], length=period)
                ema_value = safe_get_indicator_value(
                    pd.DataFrame({f'EMA_{period}': ema_series}), f'ema_{period}'
                )
                if ema_value is not None:
                    ema_data[f'ema_{period}'] = ema_value
        
        if ema_data:
            indicators['ema'] = ema_data
        
        # Bollinger Bands
        if len(df) >= config.BBANDS_PERIOD:
            bbands = ta.bbands(df['Close'], length=config.BBANDS_PERIOD, std=config.BBANDS_STD)
            
            if bbands is not None and not bbands.empty:
                upper = safe_get_indicator_value(bbands, 'upper', ['BBU'])
                middle = safe_get_indicator_value(bbands, 'middle', ['BBM'])
                lower = safe_get_indicator_value(bbands, 'lower', ['BBL'])
                
                if all(v is not None for v in [upper, middle, lower]):
                    current_price = df['Close'].iloc[-1]
                    bb_position = ((current_price - lower) / (upper - lower)) * 100
                    
                    indicators['bollinger_bands'] = {
                        'upper': upper,
                        'middle': middle,
                        'lower': lower,
                        'bb_position': round(bb_position, 2),
                        'signal': 'overbought' if bb_position > 80 else 'oversold' if bb_position < 20 else 'neutral'
                    }
        
        # ATR
        if len(df) >= config.ATR_PERIOD:
            atr_series = ta.atr(df['High'], df['Low'], df['Close'], length=config.ATR_PERIOD)
            atr_value = safe_get_indicator_value(
                pd.DataFrame({'ATR': atr_series}), 'atr'
            )
            if atr_value is not None:
                indicators['atr'] = {
                    'value': atr_value,
                    'volatility': 'high' if atr_value > df['Close'].iloc[-1] * 0.02 else 'low'
                }
        
        # VWAP
        vwap_series = calculate_daily_vwap(df)
        if not vwap_series.empty:
            vwap_value = vwap_series.iloc[-1]
            if pd.notna(vwap_value):
                current_price = df['Close'].iloc[-1]
                indicators['vwap'] = {
                    'value': round(float(vwap_value), 4),
                    'signal': 'bullish' if current_price > vwap_value else 'bearish'
                }
        
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
        
        for gap in recent_gaps[:2]:  # Only recent gaps
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
    """Complete technical analysis for a ticker."""
    
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
    df = fetch_ohlcv(ticker, interval, period)
    if df.empty:
        return {"error": f"No data available for {ticker} with interval {interval} and period {period}"}
    
    # Validate data quality
    data_valid, validation_msg = validate_ohlcv_data(df)
    if not data_valid:
        return {"error": validation_msg}
    
    try:
        # Calculate gaps before any data modification
        gaps = detect_price_gaps(df)
        
        # Find support/resistance levels
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
            "technical_indicators": indicators
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
        return {"error": f"Analysis failed: {str(e)}"}

# ---------------------------
# API Endpoints
# ---------------------------

@app.route("/", methods=["GET"])
def root():
    """Health check endpoint."""
    return jsonify({
        "service": "Technical Analysis Microservice",
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(data_cache.cache),
        "endpoints": {
            "analysis": "/analysis?ticker=SYMBOL&interval=1d&period=1mo",
            "signals": "/signals?ticker=SYMBOL",
            "health": "/health",
            "cache": "/cache/stats"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Detailed health check."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache": {
            "size": len(data_cache.cache),
            "max_size": data_cache.max_size,
            "ttl_seconds": data_cache.ttl_seconds
        },
        "config": {
            "min_data_points": config.MIN_DATA_POINTS,
            "gap_threshold": config.GAP_THRESHOLD_PERCENT,
            "pivot_window": config.PIVOT_WINDOW
        }
    })

@app.route("/analysis", methods=["GET"])
def analysis_endpoint():
    """Main technical analysis endpoint."""
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
            "last_price": result["last_price"],
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
    """Cache statistics endpoint."""
    return jsonify({
        "cache_size": len(data_cache.cache),
        "max_cache_size": data_cache.max_size,
        "ttl_seconds": data_cache.ttl_seconds,
        "cached_tickers": list(data_cache.cache.keys())[:10]  # Show first 10
    })

@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear cache endpoint."""
    try:
        data_cache.cache.clear()
        data_cache.timestamps.clear()
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    logger.info("Starting Technical Analysis Microservice v2.0.0")
    app.run(host="0.0.0.0", port=5000, debug=False)

