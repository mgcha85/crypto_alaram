import time
from config import CHECK_INTERVAL, MA_WINDOW, THRESHOLD_PERCENT
from telegram_notifier import TelegramNotifier
from market_data import get_btc_ohlcv, calculate_moving_average
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

# 텔레그램 설정
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def check_and_notify(notifier, current_price, ma_price, threshold_percent):
    deviation = abs(current_price - ma_price) / ma_price * 100
    if deviation <= threshold_percent:
        message = f"📢 *BTCUSDT Alert!*\n\n" \
                  f"현재가: `{current_price:,.2f}`\n" \
                  f"{MA_WINDOW}MA: `{ma_price:,.2f}`\n" \
                  f"이격도: `{deviation:.3f}%` <= `{threshold_percent}%`\n\n" \
                  f"🔔 MA 수렴 조건 만족"
        notifier.send_message(message)

def main():
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    print("⏳ BTC MA 알림 봇 실행 중...")
    while True:
        try:
            df = get_btc_ohlcv(limit=MA_WINDOW+1)
            current_price = df["close"].iloc[-1]
            ma_price = calculate_moving_average(df, MA_WINDOW)
            check_and_notify(notifier, current_price, ma_price, THRESHOLD_PERCENT)
        except Exception as e:
            print(f"[에러 발생] {e}")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
