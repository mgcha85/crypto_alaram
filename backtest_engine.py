import pandas as pd
from datetime import timedelta, datetime
from backtest_config import SEED, ENTRY_COUNT, TARGET_PROFIT_PCT, ENTRY_STEP_PCT, MA_WINDOWS
from config import FEATURE_COLS
import shortuuid
import xgboost as xgb


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return drawdown.min()


def load_model(fname):
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=2,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.load_model(fname)
    return model

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


def compute_indicators(df, rsi_window=14):
    df = df.copy()

    for ma_window in MA_WINDOWS:
        df["ma"] = df["close"].rolling(ma_window).mean()
        df[f"ma{ma_window}_disparity"] = (df["close"] - df["ma"]) / df["ma"]

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df

def prepare_indicators_by_timeframes(df_5m_raw):
    timeframes = {
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "1d": "1D"
    }

    tf_data = {}

    for label, interval in timeframes.items():
        df_tf = resample_ohlcv(df_5m_raw, interval=interval)
        df_tf = compute_indicators(df_tf)
        tf_data[label] = df_tf

    return tf_data

def run_backtest(df, min_bars_between_entries=3):
    cash = SEED
    position = 0
    avg_price = 0
    entry_step = 0
    entries = []
    trades = []
    buy_info = []
    model = load_model("xgboost_model.json")

    for idx, row in df.iterrows():
        price = row["close"]
        time = row["time"] + timedelta(minutes=5)
        if time < datetime(2024, 1, 1):
            continue  # 2024년 이전 데이터는 제외

        # 최초 매수
        if position == 0 and row["is_spike"] and row["direction"] == "down":
            tf_data = prepare_indicators_by_timeframes(df.loc[:idx])

            # ✅ tf_data에서 마지막 시점 지표값 가져오기
            indicators_at_entry = {}
            for tf_label, tf_df in tf_data.items():
                latest = tf_df.iloc[-1]
                indicators_at_entry[f"rsi_{tf_label}"] = latest.get("rsi")
                for ma_window in MA_WINDOWS:
                    indicators_at_entry[f"disp{ma_window}_{tf_label}"] = latest.get(f"ma{ma_window}_disparity")
            
            data = pd.DataFrame([indicators_at_entry])[FEATURE_COLS]  # 지표값을 DataFrame으로 변환
            pred = model.predict(data.values).argmax()
            if pred == 1:
                continue

            vratio = row["volume_ratio"]
            buy_price = price
            position = SEED / ENTRY_COUNT / buy_price
            avg_price = buy_price
            cash -= position * buy_price
            entry_step = 1
            entries = [(time, buy_price)]
            entry_time = time
            last_entry_index = idx

            max_price = df.loc[idx +  1: idx + min_bars_between_entries + 1, 'high'].max()
            # trade 기록
            tid = shortuuid.uuid()
            trade = {
                "tid": tid,
                "time": time,
                "buy_price": buy_price,
                "entry_steps": entry_step,
                "position": position,
                "max_price": max_price,
                "volume_ratio": vratio,
                **indicators_at_entry  # ⬅️ 지표 포함
            }
            buy_info.append(trade)

            continue

        # 분할 매수 이후 상태
        if position > 0:
            next_entry_price = avg_price * (1 - ENTRY_STEP_PCT * entry_step)
            bars_since_last_entry = idx - last_entry_index
            vratio = row["volume_ratio"]

            if (
                price <= next_entry_price and 
                entry_step < ENTRY_COUNT and 
                bars_since_last_entry >= min_bars_between_entries
            ):
                # ✅ tf_data에서 마지막 시점 지표값 가져오기
                indicators_at_entry = {}
                for tf_label, tf_df in tf_data.items():
                    latest = tf_df.iloc[-1]
                    indicators_at_entry[f"rsi_{tf_label}"] = latest.get("rsi")
                    for ma_window in MA_WINDOWS:
                        indicators_at_entry[f"disp{ma_window}_{tf_label}"] = latest.get(f"ma{ma_window}_disparity")

                data = pd.DataFrame([indicators_at_entry])[FEATURE_COLS]  # 지표값을 DataFrame으로 변환
                # pred = model.predict(data.values).argmax()
                # if pred == 1:
                #     continue

                additional_position = SEED / ENTRY_COUNT / price
                cash -= additional_position * price
                total_value = (position * avg_price) + (additional_position * price)
                position += additional_position
                avg_price = total_value / position
                entry_step += 1
                entries.append((time, price))
                last_entry_index = idx  # 분할매수 인덱스 갱신

                max_price = df.loc[idx + 1: idx + min_bars_between_entries + 1, 'high'].max()

                # trade 기록
                trade = {
                    "tid": tid,
                    "time": time,
                    "buy_price": buy_price,
                    "entry_steps": entry_step,
                    "position": position,
                    "max_price": max_price,
                    "volume_ratio": vratio,
                    **indicators_at_entry  # ⬅️ 지표 포함
                }
                buy_info.append(trade)
                
            elif price >= avg_price * (1 + TARGET_PROFIT_PCT):
                revenue = position * price
                profit = revenue - (SEED - cash)

                # ✅ tf_data에서 마지막 시점 지표값 가져오기
                indicators_at_entry = {}
                for tf_label, tf_df in tf_data.items():
                    latest = tf_df.iloc[-1]
                    indicators_at_entry[f"rsi_{tf_label}"] = latest.get("rsi")
                    for ma_window in MA_WINDOWS:
                        indicators_at_entry[f"disp{ma_window}_{tf_label}"] = latest.get(f"ma{ma_window}_disparity")

                # trade 기록
                trade = {
                    "tid": tid,
                    "entry_time": entry_time,
                    "exit_time": time,
                    "holding_days": time - entry_time,
                    "avg_entry_price": avg_price,
                    "exit_price": price,
                    "entry_steps": entry_step,
                    "position": position,
                    "cash_after": cash + revenue,
                    "profit": profit,
                    "entry_log": entries,
                    **indicators_at_entry  # ⬅️ 지표 포함
                }
                trades.append(trade)
                
                # 포지션 정리
                cash += revenue
                position = 0
                avg_price = 0
                entry_step = 0
                entries = []
                entry_time = None
                last_entry_index = None

    result_df = pd.DataFrame(trades)
    # result_df["mdd"] = None  # 빈 컬럼 초기화
    # df = df.set_index("time")  # 시계열 인덱스 설정

    buy_info_df = pd.DataFrame(buy_info)

    # for i, trade in result_df.iterrows():
    #     start = trade["entry_time"]
    #     end = trade["exit_time"]
    #     entry_price = trade["avg_entry_price"]
    #     size = trade["position"]

    #     # 진입~청산 구간 가격 데이터
    #     price_window = df.loc[start:end]["close"]
    #     if price_window.empty:
    #         result_df.at[i, "mdd"] = None
    #         continue

        # 포지션 가치 시계열
        # equity_curve = price_window * size

        # # 개별 MDD 계산
        # mdd = calculate_max_drawdown(equity_curve)
        # result_df.at[i, "mdd"] = mdd

    return result_df, buy_info_df, cash
