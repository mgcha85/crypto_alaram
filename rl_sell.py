import pandas as pd
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



def extract_post_entry_window(entry_time, df_price, window=48):
    """
    매수 이후 일정 구간(기본 4시간 = 48개 5분봉) 가격과 지표 슬라이스
    """
    start = entry_time
    end = entry_time + pd.Timedelta(minutes=5 * window)
    return df_price.loc[start:end].reset_index()


class SellEnv(gym.Env):
    def __init__(self, df_window, entry_price):
        super(SellEnv, self).__init__()
        self.df = df_window
        self.entry_price = entry_price
        self.current_step = 0
        self.max_steps = len(df_window) - 1

        # 상태: 가격, RSI, 이격도 등
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 = hold, 1 = sell

    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row["close"],
            row.get("rsi", 0),
            row.get("disp25_5m", 0),
            row.get("disp100_5m", 0),
            row.get("volume", 0),
            (row["close"] / self.entry_price) - 1  # 수익률
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = False
        reward = 0

        if action == 1 or self.current_step >= self.max_steps:
            final_price = self.df.iloc[self.current_step]["close"]
            reward = (final_price / self.entry_price) - 1
            done = True

        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape)
        return obs, reward, done, {}



if __name__ == "__main__":
    df_buy = pd.read_excel("backtest_buy_info.xlsx")
    df_price = pd.read_parquet("BTCUSDT.parquet")
    df_price["time"] = pd.to_datetime(df_price["time"])
    df_price.set_index("time", inplace=True)


    # 매수 샘플 1개에 대해 환경 생성
    entry_row = df_buy.iloc[0]
    entry_time = pd.to_datetime(entry_row["time"])
    entry_price = entry_row["buy_price"]

    df_window = extract_post_entry_window(entry_time, df_price, window=48)
    env = SellEnv(df_window, entry_price)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    print(f"📈 최종 수익률: {reward:.2%}")
    print(f"📌 매도 시점: {df_window.iloc[env.current_step]['time']}")
