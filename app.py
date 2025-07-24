from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
import traceback
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def debug_fetch_data(ticker, interval='1d', period='1mo'):
    """Debug version to see exactly what's happening."""
    try:
        logger.info(f"=== DEBUGGING FETCH FOR {ticker} ===")
        
        # Step 1: Try basic yfinance download
        logger.info(f"Step 1: Attempting yf.download({ticker}, period={period}, interval={interval})")
        
        data = yf.download(
            ticker, 
            period=period, 
            interval=interval, 
            progress=False
        )
        
        logger.info(f"Step 2: Raw data shape: {data.shape}")
        logger.info(f"Step 3: Raw data columns: {data.columns.tolist()}")
        logger.info(f"Step 4: Raw data empty? {data.empty}")
        
        if not data.empty:
            logger.info(f"Step 5: Raw data head:\n{data.head()}")
            logger.info(f"Step 6: Raw data types:\n{data.dtypes}")
        
        # Step 7: Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            logger.info("Step 7: Found MultiIndex columns, dropping level 1")
            original_cols = data.columns.tolist()
            data.columns = data.columns.droplevel(1)
            logger.info(f"Step 7a: Original columns: {original_cols}")
            logger.info(f"Step 7b: New columns: {data.columns.tolist()}")
        
        # Step 8: Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        logger.info(f"Step 8: Missing required columns: {missing_cols}")
        
        # Step 9: Clean data
        logger.info(f"Step 9: Before cleaning - shape: {data.shape}")
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                before_convert = data[col].dtype
                data[col] = pd.to_numeric(data[col], errors='coerce')
                after_convert = data[col].dtype
                logger.info(f"Step 9a: {col} type: {before_convert} -> {after_convert}")
        
        # Drop NaN rows
        before_dropna = len(data)
        data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        after_dropna = len(data)
        logger.info(f"Step 9b: Rows after dropna: {before_dropna} -> {after_dropna}")
        
        # Fill volume
        if 'Volume' in data.columns:
            data['Volume'].fillna(0, inplace=True)
            logger.info("Step 9c: Filled Volume NaN with 0")
        
        logger.info(f"Step 10: Final data shape: {data.shape}")
        logger.info(f"Step 11: Final data empty? {data.empty}")
        
        if not data.empty:
            logger.info(f"Step 12: Final data head:\n{data.head()}")
            logger.info(f"Step 13: Final data tail:\n{data.tail()}")
        
        return data
        
    except Exception as e:
        logger.error(f"ERROR in debug_fetch_data: {str(e)}")
        logger.error(f"ERROR traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def simple_analyze(ticker, interval='1d', period='1mo'):
    """Simple analysis with debug output."""
    logger.info(f"Starting analysis for {ticker}")
    
    # Fetch data with debugging
    df = debug_fetch_data(ticker, interval, period)
    
    if df.empty:
        logger.error(f"Data is empty for {ticker}")
        return {"error": f"No data available for {ticker} with interval {interval} and period {period}"}
    
    try:
        latest = df.iloc[-1]
        
        result = {
            "ticker": ticker.upper(),
            "interval": interval,
            "period": period,
            "last_updated": datetime.now().isoformat(),
            "data_points": len(df),
            "last_price": round(float(latest['Close']), 2),
            "volume": int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
            "debug_info": {
                "data_shape": list(df.shape),
                "columns": df.columns.tolist(),
                "first_date": str(df.index[0]),
                "last_date": str(df.index[-1]),
                "sample_prices": {
                    "first_close": float(df['Close'].iloc[0]),
                    "last_close": float(df['Close'].iloc[-1]),
                    "high": float(df['High'].max()),
                    "low": float(df['Low'].min())
                }
            },
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in simple_analyze: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}

# API Endpoints
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "Technical Analysis Debug Service",
        "status": "healthy",
        "version": "debug-1.0",
        "timestamp": datetime.now().isoformat(),
        "message": "Debug version to diagnose data fetching issues"
    })

@app.route("/debug", methods=["GET"])
def debug_endpoint():
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    interval = request.args.get("interval", "1d")
    period = request.args.get("period", "1mo")
    
    logger.info(f"DEBUG REQUEST: ticker={ticker}, interval={interval}, period={period}")
    
    try:
        result = simple_analyze(ticker, interval, period)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route("/analysis", methods=["GET"])
def analysis_endpoint():
    return debug_endpoint()

@app.route("/test-yfinance", methods=["GET"])
def test_yfinance():
    """Test yfinance directly."""
    ticker = request.args.get("ticker", "AAPL")
    
    try:
        logger.info(f"Testing yfinance for {ticker}")
        
        # Test 1: Basic download
        data1 = yf.download(ticker, period="5d", interval="1d", progress=False)
        
        # Test 2: With auto_adjust
        data2 = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
        
        # Test 3: Different period
        data3 = yf.download(ticker, period="1mo", interval="1d", progress=False)
        
        return jsonify({
            "ticker": ticker,
            "test1_basic": {
                "shape": list(data1.shape) if not data1.empty else [0, 0],
                "empty": data1.empty,
                "columns": data1.columns.tolist() if not data1.empty else []
            },
            "test2_auto_adjust": {
                "shape": list(data2.shape) if not data2.empty else [0, 0],
                "empty": data2.empty,
                "columns": data2.columns.tolist() if not data2.empty else []
            },
            "test3_1mo": {
                "shape": list(data3.shape) if not data3.empty else [0, 0],
                "empty": data3.empty,
                "columns": data3.columns.tolist() if not data3.empty else []
            },
            "yfinance_version": yf.__version__ if hasattr(yf, '__version__') else "unknown"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Debug Technical Analysis Service")
    app.run(host="0.0.0.0", port=port, debug=False)
