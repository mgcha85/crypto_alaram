import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
import seaborn as sns


result_df = pd.read_excel("backtest_buy_info.xlsx")
# 사용 가능한 지표 feature만 선택
feature_cols = [col for col in result_df.columns if col.startswith(("rsi", "disp"))]
print(feature_cols)

result_df["holding_days"] = (result_df["holding_days"] >= 1).astype(int)  # 5일 이상 보유 여부를 이진 분류로 변환
X = result_df[feature_cols].dropna()
y = result_df.loc[X.index, "holding_days"]  # 또는 "holding_seconds"

df_pos = result_df[result_df["holding_days"] >= 1]
df_neg = result_df[result_df["holding_days"] < 1]

# 히스토그램 그리기
# for feature_col in feature_cols:
#     plt.figure(figsize=(8, 5))
#     sns.histplot(df_pos[feature_col], color="green", label="holding_days > 1", alpha=0.5, bins=30)
#     sns.histplot(df_neg[feature_col], color="red", label="holding_days <= 1", alpha=0.5, bins=30)
#     plt.title(f"Distribution of {feature_col} by holding_days")
#     plt.show()
#     plt.close()

# 훈련/검증 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# XGBoost 분류기 초기화
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=2,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
# 모델 학습
model.fit(X_train, y_train, sample_weight=sample_weights)
model.save_model("xgboost_model.json")

# 예측 수행
y_pred = model.predict(X_test).argmax(axis=1)

# 정확도 및 분류 리포트 출력
print("정확도:", accuracy_score(y_test, y_pred))
print("분류 리포트:\n", classification_report(y_test, y_pred, target_names=["짧음", "김"]))

import matplotlib.pyplot as plt

xgb.plot_importance(model, importance_type="gain", height=0.5)
plt.title("XGBoost 특징 중요도 (Gain 기준)")
plt.tight_layout()
plt.show()
