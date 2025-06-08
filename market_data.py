import requests
import pandas as pd

def get_ohlcv(symbol, interval="1m", limit=400+1):
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
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

    # 필요한 형 변환
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')

    return df

def calculate_multi_ma(df, windows):
    ma_dict = {}
    for w in windows:
        if len(df) >= w:
            ma_dict[w] = df["close"].rolling(window=w).mean().iloc[-1]
    return ma_dict

def calculate_vwap(df):
    typical_price = (df["high"].astype(float) + df["low"].astype(float) + df["close"]) / 3
    volume = df["volume"].astype(float)
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap.iloc[-1]
