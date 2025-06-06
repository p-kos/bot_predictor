import os
import numpy as np
import yfinance as yf
from flask import Flask, jsonify, request
from binance.client import Client
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load keys from .env
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

app = Flask(__name__)
client = Client(api_key, api_secret)

LOOKBACK = 60

# Train LSTM model for a given symbol
def train_model(symbol):
    ticker = symbol.replace("USDT", "-USD")
    df = yf.download(ticker, period="1y", interval="1d")[['Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model, scaler, scaled_data

# Generate trading signal
def generate_signal(current, predicted, threshold=0.005):
    change = (predicted - current) / current
    if change > threshold:
        return "BUY"
    elif change < -threshold:
        return "SELL"
    else:
        return "HOLD"

@app.route("/predict")
def predict():
    symbol = request.args.get("symbol", default="BTCUSDT")
    try:
        model, scaler, data = train_model(symbol)

        price_info = client.get_symbol_ticker(symbol=symbol)
        current_price = float(price_info["price"])

        # Ensure last_60 is shaped (LOOKBACK, 1)
        last_60 = data[-LOOKBACK:]
        if last_60.shape != (LOOKBACK, 1):
            last_60 = last_60.reshape((LOOKBACK, 1))
        X_input = np.reshape(last_60, (1, LOOKBACK, 1))
        predicted_scaled = model.predict(X_input)
        # Ensure predicted_scaled is 2D for inverse_transform
        if predicted_scaled.ndim == 1:
            predicted_scaled = predicted_scaled.reshape(-1, 1)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

        signal = generate_signal(current_price, predicted_price)

        return jsonify({
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "signal": signal
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
