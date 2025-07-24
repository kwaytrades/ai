from flask import Flask, request, jsonify
import pandas as pd
import pandas_ta as ta
import yfinance as yf

app = Flask(__name__)

# ---------------------------
# Helper Functions
# ---------------------------

def fetch_ohlcv(ticker, interval='1h', period='1mo'):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        return data
    except Exception:
        return pd.DataFrame()

def calculate_vwap(df):
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

def detect_gaps(df):
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
    highs = df['High'].rolling(window=5, center=True).max()
    lows = df['Low'].rolling(window=5, center=True).min()

    resistance_levels = sorted(highs.dropna().unique())[-num_levels:]
    support_levels = sorted(lows.dropna().unique())[:num_levels]

    return {
        "support_levels": [float(round(s, 2)) for s in support_levels],
        "resistance_levels": [float(round(r, 2)) for r in resistance_levels]
    }

def calculate_indicators(ticker, interval='1h', period='1mo'):
    df = fetch_ohlcv(ticker, interval, period)

    if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        return {"error": f"No valid OHLCV data found for {ticker} with interval {interval} and period {period}."}

    # Convert columns to numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    if df.empty:
        return {"error": f"Data for {ticker} is empty after cleaning."}

    close_series = df['Close']
    high_series = df['High']
    low_series = df['Low']

    # Indicators
    rsi = ta.rsi(close_series, length=14)
    if isinstance(rsi, (float, int)):
        rsi = pd.Series([rsi], index=[df.index[-1]])
    df['RSI'] = rsi

    macd = ta.macd(close_series, fast=12, slow=26, signal=9)
    if isinstance(macd, pd.Series):
        macd = macd.to_frame().T

    df['EMA_20'] = ta.ema(close_series, length=20)
    df['EMA_50'] = ta.ema(close_series, length=50)
    df['EMA_200'] = ta.ema(close_series, length=200)

    bbands = ta.bbands(close_series, length=20, std=2)
    if isinstance(bbands, pd.Series):
        bbands = bbands.to_frame().T

    df['ATR'] = ta.atr(high_series, low_series, close_series, length=14)
    df['VWAP'] = calculate_vwap(df)

    latest = df.iloc[-1]
    s_r = find_support_resistance(df)

    return {
        "ticker": ticker.upper(),
        "interval": interval,
        "last_price": float(round(latest['Close'], 2)),
        "rsi": float(round(latest.get('RSI', 0), 2)) if 'RSI' in latest else None,
        "macd": {
            "macd": float(round(macd.iloc[-1]['MACD_12_26_9'], 2)) if macd is not None else None,
            "signal": float(round(macd.iloc[-1]['MACDs_12_26_9'], 2)) if macd is not None else None,
            "hist": float(round(macd.iloc[-1]['MACDh_12_26_9'], 2)) if macd is not None else None
        } if macd is not None else None,
        "ema": {
            "ema_20": float(round(latest.get('EMA_20', 0), 2)),
            "ema_50": float(round(latest.get('EMA_50', 0), 2)),
            "ema_200": float(round(latest.get('EMA_200', 0), 2))
        },
        "bollinger": {
            "upper": float(round(bbands.iloc[-1]['BBU_20_2.0'], 2)) if bbands is not None else None,
            "middle": float(round(bbands.iloc[-1]['BBM_20_2.0'], 2)) if bbands is not None else None,
            "lower": float(round(bbands.iloc[-1]['BBL_20_2.0'], 2)) if bbands is not None else None
        } if bbands is not None else None,
        "atr": float(round(latest['ATR'], 2)),
        "vwap": float(round(latest['VWAP'], 2)),
        "support": s_r["support_levels"],
        "resistance": s_r["resistance_levels"],
        "gaps": detect_gaps(df)
    }

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
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": "An unexpected server error occurred.", "details": str(e)}), 500

@app.route("/debug", methods=["GET"])
def debug_endpoint():
    ticker = request.args.get("ticker", "AAPL")
    interval = request.args.get("interval", "1d")
    period = request.args.get("period", "1mo")
    df = fetch_ohlcv(ticker, interval, period)
    try:
        return jsonify({
            "columns": list(df.columns),
            "sample_data": df.tail(5).reset_index().to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_endpoint():
    return jsonify({"status": "ok", "message": "TA Microservice is healthy."})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "TA Microservice is running!"})

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
