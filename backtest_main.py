from market_data import get_ohlcv
from volatility_detector import mark_volatility_spikes
from backtest_engine import run_backtest, resample_ohlcv
from backtest_config import SEED
import pandas as pd
from datetime import datetime


# 데이터 로딩
df = pd.read_parquet("BTCUSDT.parquet")
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['time'] += pd.Timedelta("9h")
df = resample_ohlcv(df, interval='5min')

# 변동성 스파이크 표시
df = mark_volatility_spikes(df, threshold_std=3.0, window=400)

# 조건 필터링은 백테스트 엔진에서 처리하므로 전체 데이터 전달
result_df, buy_info_df, final_cash = run_backtest(df, min_bars_between_entries=48)
result_df.to_excel("backtest_results.xlsx", index=False)

buy_info_df = buy_info_df.merge(result_df[['tid', 'holding_days', 'entry_steps']], on='tid')
buy_info_df.to_excel("backtest_buy_info.xlsx", index=False)

# print(result_df[["exit_time", "avg_entry_price", "exit_price", "entry_steps", "profit"]])
print(f"\n💰 최종 자산: ${final_cash:,.2f} / 수익률: {(final_cash - SEED) / SEED * 100:.2f}%")
