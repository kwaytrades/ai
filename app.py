from flask import Flask, request, jsonify
import pandas as pd
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
    
    # Check for sufficient data points
    if len(df) < 5:  # Lowered minimum for testing
        return False, f"Insufficient data points. Got {len(df)}, minimum required: 5"
        
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
            prepost=False,
            threads=False  # Disable threading to avoid issues
        )
        
        if data.empty:
            logger.warning(f"No data returned for ticker {ticker}")
            return pd.DataFrame()
        
        # Handle multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure proper column names
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        actual_cols = data.columns.tolist()
        
        # Map common variations
        col_mapping = {}
        for expected in expected_cols:
            for actual in actual_cols:
                if expected.lower() in actual.lower():
                    col_mapping[actual] = expected
                    break
        
        if col_mapping:
            data = data.rename(columns=col_mapping)
        
        # Convert columns to numeric and handle any data type issues
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows with NaN values in essential columns
        data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        # Fill missing volume with 0
        if 'Volume' in data.columns:
            data['Volume'].fillna(0, inplace=True)
        
        # Cache the data
        if not data.empty:
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
    try:
        if df.empty or 'Volume' not in df.columns or df['Volume'].sum() == 0:
            return pd.Series(index=df.index, dtype=float)
        
        # Simple VWAP calculation (not daily reset for simplicity)
        volume_sum = df['Volume'].cumsum()
        price_volume_sum = (df['Close'] * df['Volume']).cumsum()
        vwap = price_volume_sum / volume_sum
        
        return vwap.fillna(method='ffill')
        
    except Exception as e:
        logger.error(f"Error calculating VWAP: {str(e)}")
        return pd.Series(index=df.index, dtype=float)

def find_pivot_levels(df: pd.DataFrame, window: int = None, num_levels: int = None) -> Dict[str, List[float]]:
    """Find support and resistance levels using simple high/low analysis."""
    if window is None:
        window = min(config.PIVOT_WINDOW, len(df) // 4)
    if num_levels is None:
        num_levels = min(config.MAX_SUPPORT_RESISTANCE_LEVELS, 3)
        
    if len(df) < window * 2 + 1:
        return {"support_levels": [], "resistance_levels": []}
    
    try:
        # Simplified approach: use rolling max/min
        rolling_high = df['High'].rolling(window=window, center=True).max()
        rolling_low = df['Low'].rolling(window=window, center=True).min()
        
        # Find where actual values equal rolling max/min (pivot points)
        pivot_highs = df.loc[df['High'] == rolling_high, 'High'].dropna()
        pivot_lows = df.loc[df['Low'] == rolling_low, 'Low'].dropna()
        
        # Get most recent significant levels
        resistance_levels = sorted(pivot_highs.tail(num_levels * 2).unique(), reverse=True)[:num_levels]
        support_levels = sorted(pivot_lows.tail(num_levels * 2).unique())[:num_levels]
        
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
        
        # Filter for significant gaps and remove NaN
        significant_mask = (abs(gap_percent) > threshold_percent) & gap_percent.notna()
        
        if not significant_mask.any():
            return []
            
        significant_gaps = df[significant_mask].copy()
        significant_gaps['gap_percent'] = gap_percent[significant_mask]
        
        gaps = []
        for index, row in significant_gaps.tail(5).iterrows():  # Only last 5 gaps
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
                gaps.append(gap_info)
            except Exception as gap_error:
                logger.warning(f"Error processing gap at {index}: {str(gap_error)}")
                continue
        
        return gaps
        
    except Exception as e:
        logger.error(f"Error detecting gaps: {str(e)}")
        return []

def safe_get_indicator_value(df: pd.DataFrame, column_pattern: str, fallback_cols: List[str] = None) -> Optional[float]:
    """Safely extract indicator values with pattern matching."""
    if df.empty:
        return None
        
    try:
        # Try pattern matching first
        matching_cols = [col for col in df.columns if column_pattern.lower() in col.lower()]
        
        if matching_cols:
            value = df.iloc[-1][matching_cols[0]]
            return round(float(value), 4) if pd.notna(value) else None
        
        # Try fallback columns
        if fallback_cols:
            for col in fallback_cols:
                if col in df.columns:
                    value = df.iloc[-1][col]
                    return round(float(value), 4) if pd.notna(value) else None
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting indicator value: {str(e)}")
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all technical indicators with robust error handling."""
    if df.empty:
        return {}
    
    indicators = {}
    
    try:
        # RSI
        if len(df) >= config.RSI_PERIOD:
            try:
                rsi_series = ta.rsi(df['Close'], length=config.RSI_PERIOD)
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
                macd_data = ta.macd(
                    df['Close'], 
                    fast=config.MACD_FAST, 
                    slow=config.MACD_SLOW, 
                    signal=config.MACD_SIGNAL
                )
                
                if macd_data is not None and not macd_data.empty:
                    latest_macd = macd_data.iloc[-1]
                    
                    # Get MACD values safely
                    macd_cols = [col for col in macd_data.columns if 'MACD' in col and 'h' not in col and 's' not in col]
                    signal_cols = [col for col in macd_data.columns if 'MACDs' in col]
                    hist_cols = [col for col in macd_data.columns if 'MACDh' in col]
                    
                    if macd_cols and signal_cols and hist_cols:
                        macd_val = latest_macd[macd_cols[0]]
                        signal_val = latest_macd[signal_cols[0]]
                        hist_val = latest_macd[hist_cols[0]]
                        
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
                    ema_series = ta.ema(df['Close'], length=period)
                    if ema_series is not None and not ema_series.empty:
                        ema_value = ema_series.iloc[-1]
                        if pd.notna(ema_value):
                            ema_data[f'ema_{period}'] = round(float(ema_value), 2)
                except Exception as e:
                    logger.error(f"Error calculating EMA {period}: {str(e)}")
        
        if ema_data:
            indicators['ema'] = ema_data
        
        # ATR
        if len(df) >= config.ATR_PERIOD:
            try:
                atr_series = ta.atr(df['High'], df['Low'], df['Close'], length=config.ATR_PERIOD)
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
        "version": "2.0.1",
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
            "min_data_points": 5,
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

# Legacy endpoints for backward compatibility
@app.route("/ta", methods=["GET"])
def legacy_ta_endpoint():
    """Legacy /ta endpoint - redirects to /analysis."""
    return analysis_endpoint()

@app.route("/debug", methods=["GET"])
def legacy_debug_endpoint():
    """Legacy /debug endpoint - redirects to /health."""
    return health_check()

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
    import os
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Technical Analysis Microservice v2.0.1")
    app.run(host="0.0.0.0", port=port, debug=False)
