# Bot Predictor

This project is a Flask-based API that predicts cryptocurrency prices using an LSTM neural network. It fetches historical price data from Yahoo Finance and current prices from Binance, then generates trading signals (BUY, SELL, HOLD) based on the prediction.

## Features
- LSTM-based price prediction for cryptocurrencies (default: BTCUSDT)
- Fetches historical data from Yahoo Finance
- Fetches real-time prices from Binance
- REST API endpoint for predictions
- CORS enabled for Chrome Extensions and localhost

## Requirements
- Python 3.8+
- pip

## Setup Instructions

1. **Clone the repository**

```sh
git clone git@github.com:p-kos/bot_predictor.git
cd bot_predictor/bot
```

2. **Create a virtual environment**

On Windows:
```sh
python -m venv venv
venv\Scripts\activate
```
On macOS/Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```sh
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the `bot` directory with your Binance API credentials:

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

5. **Run the Flask app**

```sh
python app.py
```

The API will be available at `http://127.0.0.1:5000/predict?symbol=BTCUSDT`.

## Example API Usage

```
GET /predict?symbol=BTCUSDT
```

Response:
```json
{
  "symbol": "BTCUSDT",
  "current_price": "...",
  "predicted_price": "...",
  "signal": "BUY" // or SELL, HOLD
}
```

## Notes
- The LSTM model is trained on-the-fly for each request using the last year of daily data.
- Supported symbols must end with `USDT` (e.g., ETHUSDT, BNBUSDT).

## License
MIT
