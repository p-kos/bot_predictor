import yfinance as yf
import numpy as np
import pandas as pd
import os
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Parámetros
LOOKBACK = 60
EPOCHS = 5
SYMBOL = "BTCUSDT"

# Inicializa Binance con tus claves
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

# # Descargar datos históricos (último año)
# df = yf.download('BTC-USD', period="1y", interval="1d")[['Close']]

# # Escalar datos
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df)

# # Preparar datos para LSTM
# X, y = [], []
# for i in range(LOOKBACK, len(scaled_data)):
#     X.append(scaled_data[i-LOOKBACK:i, 0])
#     y.append(scaled_data[i, 0])
# X, y = np.array(X), np.array(y)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# # Modelo LSTM
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
#     LSTM(50),
#     Dense(1)
# ])
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X, y, epochs=EPOCHS, batch_size=32, verbose=1)

# # Obtener precio actual de Binance
# price_info = client.get_symbol_ticker(symbol=SYMBOL)
# current_price = float(price_info['price'])

# # Predecir el próximo precio
# last_60 = scaled_data[-LOOKBACK:]
# X_test = np.reshape(last_60, (1, LOOKBACK, 1))
# predicted_scaled = model.predict(X_test)
# predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

# # Generar señal
# def generate_signal(current, predicted, threshold=0.005):
#     change = (predicted - current) / current
#     if change > threshold:
#         return "BUY"
#     elif change < -threshold:
#         return "SELL"
#     else:
#         return "HOLD"

# signal = generate_signal(current_price, predicted_price)

# print(f"\nPrecio actual: {current_price:.2f}")
# print(f"Precio predicho: {predicted_price:.2f}")
# print(f"📢 Señal: {signal}")
