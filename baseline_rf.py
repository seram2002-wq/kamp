"""
베이스라인 1: 랜덤포레스트 회귀 (KAMP 가이드북 재현)
- 단계1 라이브러리/데이터 불러오기
- 단계2 데이터종류및개수확인
- 단계3 데이터정제(전처리)
- 단계4 랜덤포레스트 모델학습 및 결과분석

목표: 가이드북이 보고한 train MSE 185.98 / test MSE 183.06 과 같은 자릿수의
결과가 재현되는지 확인하고, 이후 XGBoost/LightGBM과 비교할 기준점을 만든다.

주의: 가이드북은 정확한 피처/타겟/분할 방식을 코드로 전부 공개하지 않았다.
여기서는 15분/30분/45분/60분 서브컬럼(평균과 상관계수 0.99+, 사실상 평균의
구성요소)을 피처에서 제외해 데이터 누수를 막고, 시간 순서를 지키기 위해
RNN 재현과 동일하게 9월 1~14일(336시간)을 테스트셋으로 고정했다.
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

os.makedirs("results", exist_ok=True)  # 결과물을 results/ 폴더에 정리 저장

DATA_PATH = "./data/okm_augumented_2021.csv"

# 단계1: 데이터 불러오기 -------------------------------------------------
df = pd.read_csv(DATA_PATH)

# 단계2: 데이터 종류 및 개수 확인 ----------------------------------------
print(f"[정보] 데이터 shape: {df.shape}")

# 단계3: 데이터 정제(전처리) ----------------------------------------------
# 결측치: 풍속(3) 강수량(1)은 시계열 보간, 공장인원(17)은 전부 생산량=0
# 구간이므로 0으로 채운다 (앞선 데이터진단 단계 결과에 근거).
df["풍속"] = df["풍속"].interpolate()
df["강수량"] = df["강수량"].interpolate().fillna(0)
df["공장인원"] = df["공장인원"].fillna(0)

# 이상치 보정: "시간" 컬럼이 0~23 범위를 벗어난 48개 행(2021-07-13, 07-15 두 날짜)
# -> 확인 결과 두 날짜 모두 24행이 그대로 존재하고 앞뒤 날짜와 행 순서(인덱스)도
# 이어져 있어, 데이터가 밀리거나 빠진 게 아니라 "시간" 값 자체만 깨진 것으로 판단.
# 두 날짜 모두 생산량이 24시간 내내 0이라 어느 시점에 어떤 값이었는지 다툴 여지가
# 없으므로, 각 날짜 안에서 행이 등장한 순서를 그대로 0~23시로 재구성한다(행 삭제
# 없이 보정 -> RNN의 168시간 lag 구간에 결측 구간이 생기지 않도록 함).
anomaly_dates = df.loc[~df["시간"].between(0, 23), "날짜"].unique()
for d in anomaly_dates:
    idx = df.index[df["날짜"] == d]
    df.loc[idx, "시간"] = range(len(idx))
remaining_violations = (~df["시간"].between(0, 23)).sum()
print(f"[정보] '시간' 이상치 보정 완료: 대상 날짜 {list(anomaly_dates)} -> 보정 후 범위 위반 {remaining_violations}건")

TARGET = "평균"
LEAK_COLS = ["15분", "30분", "45분", "60분", TARGET, "날짜"]
feature_cols = [c for c in df.columns if c not in LEAK_COLS]
print(f"[정보] 사용 피처({len(feature_cols)}개): {feature_cols}")

X = df[feature_cols]
y = df[TARGET]

# 시간순 분할: 9/1~9/14(336시간)을 테스트로, 나머지를 학습으로 사용
test_mask = df["날짜"] >= 20210901
X_train, X_test = X[~test_mask], X[test_mask]
y_train, y_test = y[~test_mask], y[test_mask]
print(f"[정보] train={len(X_train)}행 / test={len(X_test)}행")

# 단계4: 랜덤포레스트 모델학습 및 결과분석 --------------------------------
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
forest_reg.fit(X_train, y_train)

train_mse = mean_squared_error(y_train, forest_reg.predict(X_train))
test_mse = mean_squared_error(y_test, forest_reg.predict(X_test))

print("\n=== 결과 ===")
print(f"train MSE: {train_mse:.4f}   (가이드북 참고값: 185.98)")
print(f"test  MSE: {test_mse:.4f}   (가이드북 참고값: 183.06)")

importances = pd.Series(forest_reg.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n[변수중요도]")
print(importances.to_string())

importances.to_csv("results/rf_feature_importance.csv")
with open("results/rf_metrics.txt", "w") as f:
    f.write(f"train_mse={train_mse:.4f}\ntest_mse={test_mse:.4f}\n")