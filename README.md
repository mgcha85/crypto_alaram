# 📉 BTC MA Alert Bot 📲

> **바이낸스 BTCUSDT 가격이 특정 이동평균선(MA)에 수렴하면 텔레그램으로 알림을 보내는 자동 모니터링 봇**


## ✅ 주요 기능

- 바이낸스 API를 통해 실시간 BTCUSDT 시세 모니터링
- 지정된 주기마다 이동평균선(MA) 계산
- 현재가와 MA의 이격도가 임계값 이내면 텔레그램으로 알림 전송
- 모든 설정은 `.env` 파일에서 조절 가능

---

## 📦 설치 및 실행 방법

### 1. 프로젝트 클론

```bash
git clone https://github.com/yourname/btc-ma-alert.git
cd btc-ma-alert
```

### 2. 가상환경 설정 (선택)

```bash
python -m venv venv
source venv/bin/activate  # 또는 venv\Scripts\activate (Windows)
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. `.env` 파일 생성

`.env` 파일을 프로젝트 루트에 생성하고 아래 내용을 입력하세요:

```env
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

> 📌 텔레그램 봇 토큰과 채팅 ID는 [@BotFather](https://t.me/BotFather)와 [getUpdates](https://api.telegram.org/bot<your_token>/getUpdates)로 확인할 수 있습니다.


### 5. 실행

```bash
python main.py
```

---

## ⚙️ 환경 변수 설명

| 변수명                 | 설명                           | 예시          |
| ------------------- | ---------------------------- | ----------- |
| `TELEGRAM_TOKEN`    | 텔레그램 봇 API 토큰                | `123456...` |
| `TELEGRAM_CHAT_ID`  | 메시지를 보낼 채팅 ID                | `12345678`  |
| `CHECK_INTERVAL`    | 가격 체크 주기 (초 단위)              | `60`        |
| `MA_WINDOW`         | 이동평균선 기간 (ex: 20일선)          | `20`        |
| `THRESHOLD_PERCENT` | 이격도 임계값 (%) — 해당 이하일 때 알림 전송 | `0.2`       |

---

## 📌 예시 메시지

```
📢 BTCUSDT Alert!

현재가: 66,000.00
20MA: 65,850.00
이격도: 0.227% <= 0.2%

🔔 MA 수렴 조건 만족
```

---

## 🛠 향후 개발 아이디어

* MA20, MA50, MA200 등 다중 이평선 감시
* 알림 중복 방지 (최근 알림 이력 저장)
* 알림 메시지 UI 개선 (Markdown + 차트)
* FastAPI 또는 텔레그램 명령어로 설정 동적 제어

---

## 📃 라이선스

MIT License

```
