"""
베이스라인 2: Simple RNN 기반 피크전력(15분) 예측 (KAMP 가이드북 재현)
- 단계1 라이브러리/데이터 불러오기
- 단계2 데이터종류및개수확인
- 단계3 데이터정제(전처리)
- 단계4 훈련/테스트 데이터 분리
- 단계5 Simple RNN 모델 구축 및 훈련
- 단계6 분석 시각화

가이드북은 "15분 Peak Consumption"에 대해 t-1~t-168(1주일치 시간별) lag를
피처로 만들고 MinMax 정규화 후 RNN에 넣어 예측했다. 정확한 레이어 구성은
가이드북에 공개돼 있지 않아, 여기서는 통상적인 Simple RNN 구조로 재현한다.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)

DATA_PATH = "./data/okm_augumented_2021.csv"
N_LAGS = 168          # 1주일치 시간별 lag (가이드북과 동일)
TEST_HOURS = 336      # 9/1~9/14 (2주), 가이드북과 동일한 테스트 구간

# 단계1: 데이터 불러오기 -------------------------------------------------
df = pd.read_csv(DATA_PATH)
series = df["15분"].astype(float).reset_index(drop=True)
print(f"[정보] 원계열 길이: {len(series)}")

# 단계2/3: lag 피처 생성 (결측 발생 구간은 제거) --------------------------
lagged = pd.DataFrame({"y": series})
for lag in range(1, N_LAGS + 1):
    lagged[f"lag_{lag}"] = series.shift(lag)
lagged = lagged.dropna().reset_index(drop=True)
print(f"[정보] lag 피처 구성 후 shape: {lagged.shape}  (경고 없이 168개 lag 생성)")

# 단계4: 훈련/테스트 분리 (시간순, 뒤 336개를 테스트로) --------------------
train_df = lagged.iloc[:-TEST_HOURS]
test_df = lagged.iloc[-TEST_HOURS:]

feat_cols = [c for c in lagged.columns if c != "y"]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(train_df[feat_cols])
X_test = scaler_X.transform(test_df[feat_cols])
y_train = scaler_y.fit_transform(train_df[["y"]])
y_test = scaler_y.transform(test_df[["y"]])

# RNN 입력 형태: (샘플, 168 타임스텝, 1 피처)
X_train_seq = X_train.reshape(-1, N_LAGS, 1)
X_test_seq = X_test.reshape(-1, N_LAGS, 1)

print(f"[정보] train={X_train_seq.shape}, test={X_test_seq.shape}")

# 단계5: Simple RNN 모델 구축 및 훈련 -------------------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(N_LAGS, 1)),
    keras.layers.SimpleRNN(64, activation="tanh"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
history = model.fit(
    X_train_seq, y_train,
    epochs=60, batch_size=128,
    verbose=2, callbacks=[early_stop],
)

# 예측 및 역정규화 --------------------------------------------------------
pred_scaled = model.predict(X_test_seq, verbose=0)
pred = scaler_y.inverse_transform(pred_scaled).flatten()
actual = test_df["y"].values

test_mse = mean_squared_error(actual, pred)
print("\n=== 결과 ===")
print(f"test MSE (원단위): {test_mse:.4f}")
print(f"test RMSE (원단위): {np.sqrt(test_mse):.4f}")

result_df = pd.DataFrame({"actual_15min": actual, "forecast": pred})
result_df.to_csv("rnn_forecast_result.csv", index=False)
with open("rnn_metrics.txt", "w") as f:
    f.write(f"test_mse={test_mse:.4f}\ntest_rmse={np.sqrt(test_mse):.4f}\n")

# 시각화 ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(11, 4))
plt.plot(actual, label="실측 (15분 소비량)", linewidth=1.2)
plt.plot(pred, label="RNN 예측", linewidth=1.2, alpha=0.85)
plt.title("Simple RNN 피크전력(15분) 예측 — 2021-09-01~09-14 테스트 구간")
plt.xlabel("시간 (시간 단위, 336=2주)")
plt.ylabel("15분 피크소비량")
plt.legend()
plt.tight_layout()
plt.savefig("rnn_forecast_plot.png", dpi=130)
print("[정보] rnn_forecast_plot.png 저장 완료")
