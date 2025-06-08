# 체크 주기 (초): 60초=1분, 300초=5분 등
CHECK_INTERVAL = 60

# 이평선 기간 설정
MA_WINDOW = 20

# 현재가와 이동평균선의 이격도 임계값 (%)
MA_WINDOWS = [25, 50, 100, 200, 400]  # 다양한 이평선 기간 설정
THRESHOLDS_PERCENT = [0.1, 0.2, 0.2, 0.3, 0.5, 0.5]  # 시간 간격에 맞는 이격도 임계값 설정
INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]  # 다양한 시간 간격 설정
SYMBOL = "BTCUSDT"  # 거래할 심볼 설정
FEATURE_COLS = ['rsi_5m', 'disp25_5m', 'disp50_5m', 'disp100_5m', 'disp200_5m', 'disp400_5m', 'rsi_15m', 'disp25_15m', 'disp50_15m', 'disp100_15m', 'disp200_15m', 'disp400_15m', 'rsi_1h', 'disp25_1h', 'disp50_1h', 'disp100_1h', 'disp200_1h', 'disp400_1h', 'rsi_1d', 'disp25_1d', 'disp50_1d', 'disp100_1d', 'disp200_1d', 'disp400_1d']
