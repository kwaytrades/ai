from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
import traceback
import logging
import numpy as np
import re
from datetime import datetime, timedelta
import os
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def try_yfinance_with_headers(ticker, interval='1d', period='1mo'):
    """Try yfinance with different configurations."""
    configs_to_try = [
        # Config 1: Basic
        {"auto_adjust": True, "prepost": False, "threads": False},
        
        # Config 2: With different session
        {"auto_adjust": False, "prepost": False, "threads": False},
        
        # Config 3: Minimal
        {}
    ]
    
    for i, config in enumerate(configs_to_try):
        try:
            logger.info(f"Trying yfinance config {i+1}: {config}")
            
            # Create a session with headers
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            # Try with session
            ticker_obj = yf.Ticker(ticker, session=session)
            data = ticker_obj.history(period=period, interval=interval, **config)
            
            if not data.empty:
                logger.info(f"Success with config {i+1}! Got {len(data)} rows")
                return data
            else:
                logger.warning(f"Config {i+1} returned empty data")
                
        except Exception as e:
            logger.error(f"Config {i+1} failed: {str(e)}")
            continue
    
    return pd.DataFrame()

def get_sample_data(ticker, period='1mo'):
    """Generate realistic sample data when real data fails."""
    logger.info(f"Generating sample data for {ticker}")
    
    # Base prices for different stocks
    base_prices = {
        'AAPL': 191.0,
        'MSFT': 428.0, 
        'GOOGL': 170.0,
        'TSLA': 248.0,
        'NVDA': 135.0,
        'AMZN': 175.0,
        'META': 520.0,
        'NFLX': 490.0
    }
    
    base_price = base_prices.get(ticker.upper(), 100.0)
    
    # Generate dates
    if period == '1d':
        periods = 1
    elif period == '5d':
        periods = 5
    elif period == '1mo':
        periods = 21
    elif period == '3mo':
        periods = 63
    else:
        periods = 21
    
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
    
    # Generate realistic price movements
    np.random.seed(42)  # For consistent sample data
    
    # Create price series with some volatility
    returns = np.random.normal(0.001, 0.02, periods)  # ~0.1% daily return, 2% volatility
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close
        volatility = close * 0.015  # 1.5% intraday volatility
        
        high = close + np.random.uniform(0, volatility)
        low = close - np.random.uniform(0, volatility)
        open_price = low + np.random.uniform(0, high - low)
        
        # Ensure OHLC rules
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = int(np.random.uniform(50000000, 150000000))  # Realistic volume
        
        data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2), 
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    logger.info(f"Generated sample data: {len(df)} rows for {ticker}")
    return df

def fetch_data_robust(ticker, interval='1d', period='1mo'):
    """Robust data fetching with multiple fallbacks."""
    try:
        # Method 1: Try yfinance with different configs
        data = try_yfinance_with_headers(ticker, interval, period)
        
        if not data.empty:
            logger.info("Successfully fetched real data from yfinance")
            
            # Clean the data
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Ensure proper column names
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            actual_cols = data.columns.tolist()
            
            col_mapping = {}
            for expected in expected_cols:
                for actual in actual_cols:
                    if expected.lower() in actual.lower():
                        col_mapping[actual] = expected
                        break
            
            if col_mapping:
                data = data.rename(columns=col_mapping)
            
            # Clean data types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            
            if 'Volume' in data.columns:
                data['Volume'].fillna(0, inplace=True)
            
            if not data.empty:
                return data, "real"
        
        # Method 2: If yfinance fails, use sample data
        logger.warning(f"yfinance failed for {ticker}, using sample data")
        sample_data = get_sample_data(ticker, period)
        return sample_data, "sample"
        
    except Exception as e:
        logger.error(f"All data fetching methods failed for {ticker}: {str(e)}")
        # Last resort: minimal sample data
        sample_data = get_sample_data(ticker, period)
        return sample_data, "sample"

# Technical Analysis Functions (same as before)
def calculate_rsi(prices, period=14):
    """Calculate RSI manually."""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return pd.Series(index=prices.index, dtype=float)

def calculate_ema(prices, period):
    """Calculate EMA manually."""
    try:
        return prices.ewm(span=period).mean()
    except:
        return pd.Series(index=prices.index, dtype=float)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD manually."""
    try:
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    except:
        return None

def calculate_vwap(df):
    """Calculate VWAP manually."""
    try:
        if 'Volume' not in df.columns or df['Volume'].sum() == 0:
            return pd.Series(index=df.index, dtype=float)
        
        cumulative_volume = df['Volume'].cumsum()
        cumulative_price_volume = (df['Close'] * df['Volume']).cumsum()
        vwap = cumulative_price_volume / cumulative_volume
        return vwap
    except:
        return pd.Series(index=df.index, dtype=float)

def find_support_resistance(df, window=5, num_levels=3):
    """Find support and resistance levels."""
    try:
        highs = df['High'].rolling(window=window, center=True).max()
        lows = df['Low'].rolling(window=window, center=True).min()
        
        pivot_highs = df.loc[df['High'] == highs, 'High'].dropna()
        pivot_lows = df.loc[df['Low'] == lows, 'Low'].dropna()
        
        resistance_levels = sorted(pivot_highs.tail(num_levels * 2).unique(), reverse=True)[:num_levels]
        support_levels = sorted(pivot_lows.tail(num_levels * 2).unique())[:num_levels]
        
        return {
            "support_levels": [round(float(s), 2) for s in support_levels],
            "resistance_levels": [round(float(r), 2) for r in resistance_levels]
        }
    except:
        return {"support_levels": [], "resistance_levels": []}

def analyze_stock(ticker, interval='1d', period='1mo'):
    """Perform complete stock analysis with robust data fetching."""
    
    # Fetch data with fallbacks
    df, data_source = fetch_data_robust(ticker, interval, period)
    
    if df.empty:
        return {"error": f"Unable to fetch any data for {ticker}"}
    
    try:
        # Calculate indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['EMA_20'] = calculate_ema(df['Close'], 20)
        df['EMA_50'] = calculate_ema(df['Close'], 50)
        df['VWAP'] = calculate_vwap(df)
        
        macd_data = calculate_macd(df['Close'])
        sr_levels = find_support_resistance(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Build response
        result = {
            "ticker": ticker.upper(),
            "interval": interval,
            "period": period,
            "data_source": data_source,  # "real" or "sample"
            "last_updated": datetime.now().isoformat(),
            "data_points": len(df),
            "last_price": round(float(latest['Close']), 2),
            "volume": int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
            "price_change": {
                "amount": round(float(latest['Close'] - df['Close'].iloc[-2]), 2) if len(df) > 1 else 0,
                "percent": round(float((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100), 2) if len(df) > 1 else 0
            },
            "technical_indicators": {
                "rsi": {
                    "value": round(float(latest['RSI']), 2) if pd.notna(latest['RSI']) else None,
                    "signal": "overbought" if pd.notna(latest['RSI']) and latest['RSI'] > 70 else 
                             "oversold" if pd.notna(latest['RSI']) and latest['RSI'] < 30 else "neutral"
                },
                "ema": {
                    "ema_20": round(float(latest['EMA_20']), 2) if pd.notna(latest['EMA_20']) else None,
                    "ema_50": round(float(latest['EMA_50']), 2) if pd.notna(latest['EMA_50']) else None
                },
                "vwap": {
                    "value": round(float(latest['VWAP']), 2) if pd.notna(latest['VWAP']) else None,
                    "signal": "bullish" if pd.notna(latest['VWAP']) and latest['Close'] > latest['VWAP'] else "bearish"
                }
            },
            "support_levels": sr_levels["support_levels"],
            "resistance_levels": sr_levels["resistance_levels"]
        }
        
        # Add MACD if available
        if macd_data and not macd_data['macd'].empty:
            latest_macd = macd_data['macd'].iloc[-1]
            latest_signal = macd_data['signal'].iloc[-1]
            latest_hist = macd_data['histogram'].iloc[-1]
            
            if pd.notna(latest_macd) and pd.notna(latest_signal):
                result["technical_indicators"]["macd"] = {
                    "macd": round(float(latest_macd), 4),
                    "signal": round(float(latest_signal), 4),
                    "histogram": round(float(latest_hist), 4) if pd.notna(latest_hist) else None,
                    "trend": "bullish" if latest_macd > latest_signal else "bearish"
                }
        
        # Generate signals
        signals = []
        
        # RSI signals
        rsi_data = result["technical_indicators"].get("rsi", {})
        if rsi_data.get("value"):
            rsi_val = rsi_data["value"]
            if rsi_val > 70:
                signals.append({
                    "type": "warning",
                    "indicator": "RSI",
                    "message": f"Overbought condition (RSI: {rsi_val})"
                })
            elif rsi_val < 30:
                signals.append({
                    "type": "opportunity",
                    "indicator": "RSI",
                    "message": f"Oversold condition (RSI: {rsi_val})"
                })
        
        # MACD signals
        macd_data = result["technical_indicators"].get("macd", {})
        if macd_data.get("trend"):
            trend = macd_data["trend"]
            signals.append({
                "type": "bullish" if trend == "bullish" else "bearish",
                "indicator": "MACD",
                "message": f"MACD trend is {trend}"
            })
        
        # VWAP signals
        vwap_data = result["technical_indicators"].get("vwap", {})
        if vwap_data.get("signal"):
            signals.append({
                "type": "info",
                "indicator": "VWAP",
                "message": f"Price is {'above' if vwap_data['signal'] == 'bullish' else 'below'} VWAP"
            })
        
        if not signals:
            signals.append({
                "type": "neutral",
                "indicator": "Overall",
                "message": "No strong signals detected"
            })
        
        result["signals"] = signals
        
        # Add disclaimer if using sample data
        if data_source == "sample":
            result["disclaimer"] = "Using sample data for demonstration. Real market data unavailable due to API restrictions."
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}

# API Endpoints
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "Robust Technical Analysis Microservice",
        "status": "healthy",
        "version": "2.2.0-robust",
        "timestamp": datetime.now().isoformat(),
        "features": ["Auto-fallback to sample data", "Multiple data source attempts", "Full technical analysis"],
        "endpoints": {
            "analysis": "/analysis?ticker=SYMBOL&interval=1d&period=1mo",
            "signals": "/signals?ticker=SYMBOL",
            "test": "/test-data?ticker=SYMBOL"
        }
    })

@app.route("/analysis", methods=["GET"])
def analysis_endpoint():
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    interval = request.args.get("interval", "1d")
    period = request.args.get("period", "1mo")
    
    try:
        result = analyze_stock(ticker, interval, period)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route("/signals", methods=["GET"])
def signals_endpoint():
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    interval = request.args.get("interval", "1d")
    period = request.args.get("period", "1mo")
    
    try:
        result = analyze_stock(ticker, interval, period)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "ticker": result["ticker"],
            "data_source": result["data_source"],
            "last_price": result["last_price"],
            "signals": result["signals"],
            "timestamp": result["last_updated"]
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route("/test-data", methods=["GET"])
def test_data_endpoint():
    """Test data fetching capabilities."""
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    
    try:
        # Try real data
        real_data = try_yfinance_with_headers(ticker, "1d", "1mo")
        
        # Generate sample data  
        sample_data = get_sample_data(ticker, "1mo")
        
        return jsonify({
            "ticker": ticker,
            "real_data_available": not real_data.empty,
            "real_data_shape": list(real_data.shape) if not real_data.empty else [0, 0],
            "sample_data_shape": list(sample_data.shape),
            "real_data_columns": real_data.columns.tolist() if not real_data.empty else [],
            "sample_data_columns": sample_data.columns.tolist(),
            "recommendation": "real" if not real_data.empty else "sample"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# Legacy endpoints
@app.route("/ta", methods=["GET"])
def legacy_ta():
    return analysis_endpoint()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_sources": ["yfinance (with fallback)", "sample data generator"],
        "capabilities": ["RSI", "MACD", "EMA", "VWAP", "Support/Resistance"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Robust Technical Analysis Microservice v2.2.0")
    app.run(host="0.0.0.0", port=port, debug=False)
