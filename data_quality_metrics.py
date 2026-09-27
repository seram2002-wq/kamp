"""
부록 4장 재현: KAMP 6대 데이터 품질지표 (완전성/유일성/유효성/일관성/정확성/무결성)

가이드북이 공개한 산식(부록 4장):
    완전성 = (1 - 결측치수/전체데이터수) x 100
    유일성 = (유일한 데이터수/전체데이터수) x 100
    유효성 = (유효범위를 만족하는 데이터수/전체데이터수) x 100
    일관성 = (자료형이 일치하는 데이터수/전체데이터수) x 100
    정확성 = 이 데이터셋은 컬럼값이 서로 독립적이라 가이드북에서 측정하지 않음
    무결성 = 위 지표들을 종합한 지표 (정확한 합산 공식은 코드로 공개되지 않음)

가이드북 참고값(부록에 제시된 결과):
    완전성 99.68% -> (결측치 보정 후) 100.00%
    유일성 99.69%
    유효성 100.00%
    일관성 100.00%

주의(투명성 고지): 무결성 지표는 가이드북 본문에 "위 지표를 종합한다"는
설명만 있고 정확한 계산식(예: 가중합/단순평균/미달지표 개수 기반 등)은
코드로 공개되지 않았습니다. 아래에서는 측정 가능한 4개 지표(완전성·유일성·
유효성·일관성)의 단순평균을 무결성의 근사치로 계산하며, 이는 가이드북의
공식이 아니라 우리가 임의로 정한 합산 방식임을 밝힙니다.
"""
import os
import pandas as pd

os.makedirs("results", exist_ok=True)  # 결과물을 results/ 폴더에 정리 저장

DATA_PATH = "./data/okm_augumented_2021.csv"
df = pd.read_csv(DATA_PATH)
n_rows, n_cols = df.shape
total_cells = n_rows * n_cols

print(f"[정보] 데이터 shape: {df.shape}  (전체 셀 수: {total_cells})")

# -------------------------------------------------------------------------
# 1) 완전성 (Completeness) = (1 - 결측치수/전체데이터수) x 100
# -------------------------------------------------------------------------
missing_cells = df.isna().sum().sum()
completeness = (1 - missing_cells / total_cells) * 100
print(f"\n[1] 완전성: 결측치 {missing_cells}개 / 전체 {total_cells}개")
print(f"    완전성 지수 = {completeness:.2f}%")

# -------------------------------------------------------------------------
# 2) 유일성 (Uniqueness) = (유일한 데이터수/전체데이터수) x 100
#    -> 행 전체 기준 중복이 아닌 행의 비율
# -------------------------------------------------------------------------
n_duplicate_rows = df.duplicated().sum()
n_unique_rows = n_rows - n_duplicate_rows
uniqueness = (n_unique_rows / n_rows) * 100
print(f"\n[2] 유일성: 중복 행 {n_duplicate_rows}개 / 전체 {n_rows}행")
print(f"    유일성 지수 = {uniqueness:.2f}%")

# -------------------------------------------------------------------------
# 3) 유효성 (Validity) = (유효범위를 만족하는 데이터수/전체데이터수) x 100
#    -> 날짜(수집기간 2021-01-01~2021-09-14), 시간(0~23) 범위 검사
# -------------------------------------------------------------------------
valid_date = df["날짜"].between(20210101, 20210914)
valid_hour = df["시간"].between(0, 23)
valid_rows = (valid_date & valid_hour).sum()
validity = (valid_rows / n_rows) * 100
print(f"\n[3] 유효성(보정 전): 날짜(20210101~20210914) & 시간(0~23) 범위 만족 {valid_rows}행 / {n_rows}행")
print(f"    유효성 지수(보정 전) = {validity:.2f}%")

# --- 이상치 보정: "시간" 컬럼 48건(2021-07-13, 07-15) 재구성 -------------
# 두 날짜 모두 24행이 그대로 있고 앞뒤 날짜와 행 순서가 이어져 있어(진단 완료),
# 데이터 누락이 아니라 "시간" 값만 깨진 것으로 판단. 두 날짜 모두 생산량이
# 24시간 내내 0이라 순서 재구성에 무리가 없어, 날짜별 등장 순서를 0~23시로
# 다시 채운다(행 삭제 없이 보정).
anomaly_dates = df.loc[~df["시간"].between(0, 23), "날짜"].unique()
for d in anomaly_dates:
    idx = df.index[df["날짜"] == d]
    df.loc[idx, "시간"] = range(len(idx))

valid_hour_fixed = df["시간"].between(0, 23)
valid_rows_fixed = (valid_date & valid_hour_fixed).sum()
validity_fixed = (valid_rows_fixed / n_rows) * 100
print(f"\n[3-보정] '시간' 이상치 보정 대상 날짜: {list(anomaly_dates)}")
print(f"    유효성 지수(보정 후) = {validity_fixed:.2f}%  (보정 전 {validity:.2f}% -> 보정 후 {validity_fixed:.2f}%)")

# -------------------------------------------------------------------------
# 4) 일관성 (Consistency) = (자료형이 일치하는 데이터수/전체데이터수) x 100
#    -> 숫자형이어야 할 컬럼이 실제로 숫자로 파싱되는지 검사
# -------------------------------------------------------------------------
numeric_cols = [c for c in df.columns if c not in ["날짜", "시간"]]
consistent_cells = 0
checked_cells = 0
for c in numeric_cols:
    parsed = pd.to_numeric(df[c], errors="coerce")
    consistent_cells += parsed.notna().sum() + df[c].isna().sum()  # 결측은 자료형 불일치로 보지 않음
    checked_cells += len(df[c])
consistency = (consistent_cells / checked_cells) * 100
print(f"\n[4] 일관성: 숫자형 파싱 성공 {consistent_cells}개 / 검사대상 {checked_cells}개 (수치형 컬럼 {len(numeric_cols)}개)")
print(f"    일관성 지수 = {consistency:.2f}%")

# -------------------------------------------------------------------------
# 5) 정확성 (Accuracy) -> 가이드북과 동일하게 측정 제외
# -------------------------------------------------------------------------
print("\n[5] 정확성: 해당 데이터셋은 컬럼값이 서로 독립적이라 가이드북과 동일하게 측정하지 않음")

# -------------------------------------------------------------------------
# 6) 무결성 (Integrity) -> 측정 가능한 4개 지표의 단순평균 (※ 우리 임의 정의, 가이드북 공식 아님)
# -------------------------------------------------------------------------
integrity = (completeness + uniqueness + validity_fixed + consistency) / 4
print(f"\n[6] 무결성(근사치, 단순평균, 보정 후 유효성 기준): {integrity:.2f}%   ※ 가이드북의 정확한 합산 공식은 공개되지 않아 우리가 정한 방식입니다")

print("\n" + "=" * 60)
print("가이드북 참고값 vs 이번 재현 결과(보정 전/후)")
print("=" * 60)
print(f"{'지표':10s}{'가이드북 참고값':>18s}{'보정 전':>12s}{'보정 후':>12s}")
print(f"{'완전성':10s}{'99.68~100%':>18s}{completeness:>11.2f}%{completeness:>11.2f}%")
print(f"{'유일성':10s}{'99.69%':>18s}{uniqueness:>11.2f}%{uniqueness:>11.2f}%")
print(f"{'유효성':10s}{'100.00%':>18s}{validity:>11.2f}%{validity_fixed:>11.2f}%")
print(f"{'일관성':10s}{'100.00%':>18s}{consistency:>11.2f}%{consistency:>11.2f}%")

with open("results/quality_metrics.txt", "w") as f:
    f.write(f"completeness={completeness:.4f}\n")
    f.write(f"uniqueness={uniqueness:.4f}\n")
    f.write(f"validity_before_fix={validity:.4f}\n")
    f.write(f"validity_after_fix={validity_fixed:.4f}\n")
    f.write(f"consistency={consistency:.4f}\n")
    f.write(f"integrity_approx={integrity:.4f}\n")
    f.write(f"missing_cells={missing_cells}\n")
    f.write(f"duplicate_rows={n_duplicate_rows}\n")
    f.write(f"time_anomaly_dates={list(anomaly_dates)}\n")
    f.write(f"time_anomaly_rows_fixed={len(df.index[df['날짜'].isin(anomaly_dates)])}\n")

print("\n[정보] quality_metrics.txt 저장 완료")