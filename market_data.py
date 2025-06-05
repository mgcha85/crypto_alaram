import requests
import pandas as pd

def get_btc_ohlcv(interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    return df

def calculate_moving_average(df, window):
    return df["close"].rolling(window=window).mean().iloc[-1]
