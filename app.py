from flask import Flask, request, jsonify
import pandas as pd
import pandas_ta as ta
import yfinance as yf

app = Flask(__name__)

# ---------------------------
# Helper Functions
# ---------------------------

def fetch_ohlcv(ticker, interval='1h', period='1mo'):
    """Fetch OHLCV data from Yahoo Finance."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        return data
    except Exception as e:
        return pd.DataFrame()

def calculate_vwap(df):
    """Calculate VWAP."""
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

def detect_gaps(df):
    """Detect price gaps (gap-ups and gap-downs)."""
    gaps = []
    prev_close = None
    for i in range(len(df)):
        if prev_close is not None:
            gap_percent = (df['Open'].iloc[i] - prev_close) / prev_close * 100
            if abs(gap_percent) > 1.0:
                gaps.append({
                    "date": str(df.index[i].date()),
                    "type": "gap_up" if gap_percent > 0 else "gap_down",
                    "gap_percent": round(gap_percent, 2)
                })
        prev_close = df['Close'].iloc[i]
    return gaps

def find_support_resistance(df, num_levels=2):
    """Find major support and resistance using pivot highs/lows."""
    highs = df['High'].rolling(window=5, center=True).max()
    lows = df['Low'].rolling(window=5, center=True).min()

    resistance_levels = sorted(highs.dropna().squeeze().unique())[-num_levels:]
    support_levels = sorted(lows.dropna().squeeze().unique())[:num_levels]

    return {
        "support_levels": [round(float(s), 2) for s in support_levels],
        "resistance_levels": [round(float(r), 2) for r in resistance_levels]
    }

def calculate_indicators(ticker, interval='1h', period='1mo'):
    """Calculate all technical indicators with proper validation."""
    df = fetch_ohlcv(ticker, interval, period)

    if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        return {"error": f"No valid OHLCV data found for {ticker} with interval {interval} and period {period}."}

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    if len(df) < 50:
        return {"error": f"Not enough data points for {ticker} to compute indicators."}

    close_series = df['Close']
    high_series = df['High']
    low_series = df['Low']

    try:
        df['RSI'] = ta.rsi(close_series, length=14)
        macd = ta.macd(close_series, fast=12, slow=26, signal=9)
        df['EMA_20'] = ta.ema(close_series, length=20)
        df['EMA_50'] = ta.ema(close_series, length=50)
        df['EMA_200'] = ta.ema(close_series, length=200)
        bbands = ta.bbands(close_series, length=20, std=2)
        df['ATR'] = ta.atr(high_series, low_series, close_series, length=14)
        df['VWAP'] = calculate_vwap(df)
    except Exception as e:
        return {"error": f"Indicator calculation failed: {str(e)}"}

    latest = df.iloc[-1]
    s_r = find_support_resistance(df)

    return {
        "ticker": ticker.upper(),
        "interval": interval,
        "last_price": round(float(latest['Close']), 2),
        "rsi": round(float(latest['RSI']), 2),
        "macd": {
            "macd": round(float(macd.iloc[-1]['MACD_12_26_9']), 2),
            "signal": round(float(macd.iloc[-1]['MACDs_12_26_9']), 2),
            "hist": round(float(macd.iloc[-1]['MACDh_12_26_9']), 2),
        },
        "ema": {
            "ema_20": round(float(latest['EMA_20']), 2),
            "ema_50": round(float(latest['EMA_50']), 2),
            "ema_200": round(float(latest['EMA_200']), 2)
        },
        "bollinger": {
            "upper": round(float(bbands.iloc[-1]['BBU_20_2.0']), 2),
            "middle": round(float(bbands.iloc[-1]['BBM_20_2.0']), 2),
            "lower": round(float(bbands.iloc[-1]['BBL_20_2.0']), 2)
        },
        "atr": round(float(latest['ATR']), 2),
        "vwap": round(float(latest['VWAP']), 2),
        "support": s_r["support_levels"],
        "resistance": s_r["resistance_levels"],
        "gaps": detect_gaps(df)
    }

def generate_signals(data):
    """Generate human-readable trade signals."""
    if "error" in data:
        return [data["error"]]
        
    signals = []

    if data['rsi'] > 70:
        signals.append(f"{data['ticker']} is overbought (RSI {data['rsi']})")
    elif data['rsi'] < 30:
        signals.append(f"{data['ticker']} is oversold (RSI {data['rsi']})")

    if data['last_price'] > data['vwap']:
        signals.append("Price is above VWAP (bullish intraday trend).")
    else:
        signals.append("Price is below VWAP (bearish intraday trend).")

    if data['last_price'] > data['ema']['ema_50']:
        signals.append("Price is above EMA-50, bullish momentum.")
    else:
        signals.append("Price is below EMA-50, caution.")

    signals.append(f"Support near {data['support']}, resistance at {data['resistance']}.")

    return signals

# ---------------------------
# API Endpoints
# ---------------------------

@app.route("/ta", methods=["GET"])
def ta_endpoint():
    ticker = request.args.get("ticker", "AAPL")
    interval = request.args.get("interval", "1d")
    period = request.args.get("period", "1mo")
    
    try:
        data = calculate_indicators(ticker, interval, period)
        if "error" in data:
            return jsonify(data), 400
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": "An unexpected server error occurred.", "details": str(e)}), 500

@app.route("/signals", methods=["GET"])
def signals_endpoint():
    ticker = request.args.get("ticker", "AAPL")
    interval = request.args.get("interval", "1d")
    period = request.args.get("period", "1mo")

    try:
        data = calculate_indicators(ticker, interval, period)
        signals = generate_signals(data)
        
        if "error" in data:
            return jsonify({"error": signals[0]}), 400

        return jsonify({
            "ticker": ticker.upper(),
            "signals": signals
        })
    except Exception as e:
        return jsonify({"error": "An unexpected server error occurred.", "details": str(e)}), 500

@app.route("/debug", methods=["GET"])
def debug_endpoint():
    ticker = request.args.get("ticker", "AAPL")
    interval = request.args.get("interval", "1d")
    period = request.args.get("period", "1mo")
    df = fetch_ohlcv(ticker, interval, period)
    return jsonify(df.tail(5).reset_index().to_dict(orient="records"))

@app.route("/health", methods=["GET"])
def health_check():
    try:
        df = fetch_ohlcv("AAPL", "1d", "5d")
        if df.empty:
            return jsonify({"status": "error", "message": "Unable to fetch test data from yfinance."}), 500
        last_price = round(float(df['Close'].iloc[-1]), 2)
        return jsonify({"status": "ok", "message": "Service is healthy.", "last_price": last_price})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "TA Microservice is running!"})

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
