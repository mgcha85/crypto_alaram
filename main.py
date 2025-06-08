import time
from config import CHECK_INTERVAL, MA_WINDOWS, THRESHOLDS_PERCENT, INTERVALS, SYMBOL
from telegram_notifier import TelegramNotifier
from market_data import get_ohlcv, calculate_multi_ma, calculate_vwap
from dotenv import load_dotenv
import os
from volatility_detector import detect_volatility_spike


load_dotenv()  # .env 파일 로드

# 텔레그램 설정
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def check_and_notify(notifier, interval, current_price, ma_dict, threshold_percent):
    close_ma_pairs = []

    for window, ma_price in ma_dict.items():
        deviation = abs(current_price - ma_price) / ma_price * 100
        if deviation <= threshold_percent:
            close_ma_pairs.append((window, ma_price, deviation))

    if close_ma_pairs:
        message_lines = [
            f"📢 *{SYMBOL} MA 수렴 알림!*\n",
            f"Timeframe: `{interval}`\n",
            f"현재가: `{current_price:,.2f}`\n",
            f"수렴된 MA:"
        ]
        for window, ma_price, deviation in sorted(close_ma_pairs):
            message_lines.append(f"- {window}MA: `{ma_price:,.2f}` (이격도 `{deviation:.3f}%`)")

        message_lines.append(f"\n🔔 수렴 조건: 이격도 ≤ `{threshold_percent}%`")
        notifier.send_message("\n".join(message_lines))

def main():
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    print(f"⏳ {SYMBOL} 다중 MA 알림 봇 실행 중...")
    while True:
        try:
            for interval, threshold in zip(INTERVALS, THRESHOLDS_PERCENT):
                df = get_ohlcv(SYMBOL, interval=interval, limit=max(MA_WINDOWS) + 1)
                
                current_price = df["close"].iloc[-1]
                ma_dict = calculate_multi_ma(df, MA_WINDOWS)
                ma_dict[100] = calculate_vwap(df)
                
                detect_volatility_spike(df, window=400, threshold_std=3.0)

                check_and_notify(notifier, interval, current_price, ma_dict, threshold)
        except Exception as e:
            print(f"[에러 발생] {e}")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
