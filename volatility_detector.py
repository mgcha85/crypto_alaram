import pandas as pd
from datetime import datetime


def detect_volatility_spike(df, threshold_std=3.0, window=60):
    """
    - 종가 기준 수익률을 기반으로 z-score 이상치 감지
    - 상승/하락 방향 `direction` 추가
    - 거래량 비율 `volume_ratio` 추가
    """
    df = df.copy()

    # 수익률 계산
    df["return"] = df["close"].pct_change()
    latest_return = df["return"].iloc[-1]

    # Z-score 계산
    rolling_mean = df["return"].rolling(window=window).mean()
    rolling_std = df["return"].rolling(window=window).std()
    z_scores = (df["return"] - rolling_mean) / rolling_std
    z_score = z_scores.iloc[-1]  # 가장 최근 z-score
    
    # spike 여부 및 방향
    df["z_score"] = z_scores
    df["is_spike"] = z_scores.abs() >= threshold_std
    df["direction"] = df["z_score"].apply(
        lambda z: "up" if z >= threshold_std else ("down" if z <= -threshold_std else None)
    )

    # 거래량 비율 계산
    df["volume"] = df["volume"].astype(float)
    volume_ma = df["volume"].rolling(window=window).mean()
    df["volume_ratio"] = df["volume"] / volume_ma

    if abs(z_score) >= threshold_std:
        direction = "상승 📈" if latest_return > 0 else "하락 📉"
        return {
            "z_score": z_score,
            "return_pct": latest_return * 100,
            "direction": direction,
            "time": df["close_time"].iloc[-1]
        }

    return None


def mark_volatility_spikes(df, threshold_std=3.0, window=60):
    """
    - 종가 기준 수익률을 기반으로 z-score 이상치 감지
    - 상승/하락 방향 `direction` 추가
    - 거래량 비율 `volume_ratio` 추가
    """
    df = df.copy()

    # 수익률 계산
    df["return"] = df["close"].pct_change()

    # Z-score 계산
    rolling_mean = df["return"].rolling(window=window).mean()
    rolling_std = df["return"].rolling(window=window).std()
    z_scores = (df["return"] - rolling_mean) / rolling_std

    # spike 여부 및 방향
    df["z_score"] = z_scores
    df["is_spike"] = z_scores.abs() >= threshold_std
    df["direction"] = df["z_score"].apply(
        lambda z: "up" if z >= threshold_std else ("down" if z <= -threshold_std else None)
    )

    # 거래량 비율 계산
    df["volume"] = df["volume"].astype(float)
    volume_ma = df["volume"].rolling(window=window).mean()
    df["volume_ratio"] = df["volume"] / volume_ma

    return df

def resample_ohlcv(df, interval="5T"):
    """
    1분봉 OHLCV DataFrame을 주어진 interval(예: '5T', '15T', '1H')로 리샘플링합니다.
    시간 기준은 'close_time'이며, 'close_time'은 각 구간의 마지막 시각으로 설정됩니다.
    """
    df = df.copy()
    df.set_index("time", inplace=True)

    df_resampled = pd.DataFrame()
    df_resampled["open"] = df["open"].astype(float).resample(interval).first()
    df_resampled["high"] = df["high"].astype(float).resample(interval).max()
    df_resampled["low"] = df["low"].astype(float).resample(interval).min()
    df_resampled["close"] = df["close"].astype(float).resample(interval).last()
    df_resampled["volume"] = df["volume"].astype(float).resample(interval).sum()

    df_resampled = df_resampled.dropna().reset_index()

    return df_resampled


if __name__ == '__main__':
    from market_data import get_ohlcv

    # df = pd.read_csv("BTCUSDT.csv")
    # df.to_parquet("BTCUSDT.parquet", index=False)

    df = pd.read_parquet("BTCUSDT.parquet")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['time'] += pd.Timedelta("9h")  # UTC+9 시간대 설정
    # df.set_index('time', inplace=True)

    df = resample_ohlcv(df, interval='5T')
    # df = get_ohlcv("BTCUSDT", interval="1m", limit=4000)

    df = mark_volatility_spikes(df, threshold_std=3.0, window=400)
    df = df[(df['is_spike']) & (df["direction"] == 'down')]
    print(df)