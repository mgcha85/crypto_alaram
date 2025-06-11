import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
import torch
from backtest_engine import prepare_indicators_by_timeframes
from backtest_config import MA_WINDOWS
from config import FEATURE_COLS

# CUDA 확인
print("✅ CUDA available:", torch.cuda.is_available())
print("🖥️ Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


class MultiSellEnv(gym.Env):
    """
    매수 타점마다 샘플을 무작위 선택하여 학습하는 강화학습 환경
    """
    def __init__(self, df_buy, df_price, feature_cols, window=48):
        super(MultiSellEnv, self).__init__()
        self.df_buy = df_buy
        self.df_price = df_price
        self.window = window
        self.samples = self._build_samples()
        self.current_episode = None
        self.df = None
        self.entry_price = None
        self.current_step = 0
        self.feature_cols = feature_cols + ["relative_return"]

        # 관측값: close, rsi, disp25, disp100, volume, 수익률
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feature_cols),), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 = hold, 1 = sell

    def _build_samples(self):
        samples = []
        for _, row in self.df_buy.iterrows():
            entry_time = pd.to_datetime(row["time"])
            entry_price = row["buy_price"]
            start = entry_time
            end = entry_time + pd.Timedelta(minutes=5 * self.window)
            df_window = self.df_price.loc[start:end].reset_index()
            if len(df_window) >= 10:
                samples.append((df_window, entry_price))
        return samples

    def reset(self):
        self.current_episode = np.random.randint(len(self.samples))
        self.df, self.entry_price = self.samples[self.current_episode]
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = [row.get(f, 0) for f in self.feature_cols if f != "relative_return"]
        obs.append((row["close"] / self.entry_price) - 1)  # 마지막에 수익률 추가
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = False
        reward = 0

        if action == 1 or self.current_step >= len(self.df) - 1:
            final_price = self.df.iloc[self.current_step]["close"]
            reward = (final_price / self.entry_price) - 1
            done = True

        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape)
        return obs, reward, done, {}

# 학습 및 추론
df_buy = pd.read_excel("backtest_buy_info.xlsx")
df_price = pd.read_parquet("BTCUSDT.parquet")
df_price['time'] = pd.to_datetime(df_price['time'], errors='coerce')
df_price['time'] += pd.Timedelta("9h")
df_price["volume"] = df_price["volume"].astype(float)
volume_ma = df_price["volume"].rolling(window=100).mean()
df_price["volume_ratio"] = df_price["volume"] / volume_ma

tf_data = prepare_indicators_by_timeframes(df_price)
df_price.set_index("time", inplace=True)

for key, tf_df in tf_data.items():
    feature_cols = [
        col for col in tf_df.columns 
        if col.startswith("rsi") or col.endswith("disparity")
    ]
    df_price = df_price.join(tf_df[feature_cols].add_prefix(key + "_"), how="left")

# df_price.dropna(inplace=True)

env = MultiSellEnv(df_buy, df_price, feature_cols, window=240)
model = PPO("MlpPolicy", env, verbose=1, device="cuda", learning_rate=1e-5, ent_coef=0.01, n_steps=128, batch_size=64)

model.learn(total_timesteps=20000)

# 학습된 모델 저장
model.save("ppo_sell_model")

# 모든 매수 타점에 대한 추론 결과 저장
results = []
for _, row in df_buy.iterrows():
    entry_time = pd.to_datetime(row["time"])
    entry_price = row["buy_price"]
    start = entry_time
    end = entry_time + pd.Timedelta(minutes=5 * 48)
    df_window = df_price.loc[start:end].reset_index()
    if len(df_window) < 10:
        continue

    env = MultiSellEnv(df_buy.iloc[[_]], df_price, window=48)
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

    results.append({
        "entry_time": entry_time,
        "exit_time": df_window.iloc[env.current_step]["time"],
        "return": reward
    })

df_results = pd.DataFrame(results)
df_results.to_excel("sell_results.xlsx", index=False)
