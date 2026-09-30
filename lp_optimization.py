"""
분석 C: 선형계획(LP) 기반 인력배치 최적화 (KAMP 가이드북 재현 + 확장)

가이드북 공개 내용:
    목적함수  Cost = P_human + P_electric   (최소화)
    제약조건  1 <= P_electric <= 2
              1 <= P_human    <= 75
              11 <= 2*P_electric + P_human <= 95
    공개된 예시 출력: "생산량이 0.17183770899999998일 때, 낮에 공장직원 5.0을
                     투입시켜야한다", 최종 해 610.4238440032458

주의(투명성 고지):
    가이드북은 "생산량"이 이 LP에 정확히 어떻게 연결되는지(목적함수 계수인지,
    제약조건의 우변인지)와 610.42라는 최종값이 단일 해인지 여러 시점을 합산한
    값인지를 코드로 공개하지 않았습니다. 그래서 100% 동일 재현은 불가능하고,
    아래 두 단계로 나눠 정직하게 재현/확장합니다.

    [Part A] 가이드북에 문자 그대로 공개된 제약조건만으로 "생산량 무관 정적 LP"를
              1회 풀어, 가이드북이 공개한 구조 자체가 어떤 해를 내는지 확인합니다.
    [Part B] 가이드북에는 없는 "우리의 확장"입니다. 목적함수의 계수를 실제
              데이터의 시간당 인건비/전기요금(계절)로 바꿔서, 전체 데이터셋의
              모든 시간(행)에 대해 LP를 반복 실행 -> 시간대별 최적 인원배치를
              구하고 합산합니다. 가이드북 수치(610.42, 5.0명)와의 일치를
              주장하지 않습니다 — 어디까지나 같은 문제구조를 실데이터에 적용해본
              참고용 확장입니다.

구현 노트(Windows 사용자용):
    이전 버전은 pulp + CBC(cbc.exe 외부 실행파일)를 사용했는데, 이 방식은
    행마다 별도의 외부 프로세스를 새로 띄우기 때문에 Windows + 보안 프로그램
    환경에서 극도로 느려지거나(또는 다시 차단당할) 위험이 있었습니다. 그래서
    이번 버전은 scipy.optimize.linprog로 교체했습니다 — 외부 실행파일을 전혀
    부르지 않고 파이썬 프로세스 안에서 직접 계산하므로 훨씬 빠르고 안전합니다
    (scipy는 scikit-learn의 의존 라이브러리라 이미 정상 동작이 확인된 상태).
    결과값은 동일합니다.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog

os.makedirs("results", exist_ok=True)  # 결과물을 results/ 폴더에 정리 저장

DATA_PATH = "./data/okm_augumented_2021.csv"
df = pd.read_csv(DATA_PATH)

# 공통 제약조건: 1<=P_electric<=2, 1<=P_human<=75, 11<=2*P_electric+P_human<=95
# linprog는 "<=" 부등식만 받으므로 2*Pe+Ph>=11 은 -2*Pe-Ph<=-11 로 변환
A_ub = [[-2, -1], [2, 1]]
b_ub = [-11, 95]
bounds = [(1, 2), (1, 75)]  # (P_electric, P_human)

# =========================================================================
# Part A. 가이드북 그대로: 생산량과 무관한 정적 LP 1회 풀이
# =========================================================================
print("=" * 70)
print("[Part A] 가이드북 공개 그대로의 정적 LP (생산량 미반영)")
print("=" * 70)

c_static = [1, 1]  # Cost = P_electric + P_human
res_static = linprog(c_static, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

P_electric_static, P_human_static = res_static.x
print(f"status: {'Optimal' if res_static.success else res_static.message}")
print(f"P_electric = {P_electric_static}")
print(f"P_human    = {P_human_static}")
print(f"Cost(최소) = {res_static.fun}")
print(
    "\n[해석] 가이드북이 공개한 3개 부등식만으로는 목적함수가 P_electric을 "
    "상한(2)까지, P_human을 하한 제약이 만족되는 최소값까지 밀어붙이는 "
    "'경계해'로 수렴합니다. 가이드북 예시(공장직원 5.0명, 최종값 610.42)와는 "
    "다른 값인데, 이는 가이드북이 '생산량'을 이 LP에 연결하는 정확한 방식을 "
    "코드로 공개하지 않았기 때문입니다(목적함수 계수 또는 제약조건 우변이 "
    "생산량에 따라 매 시점 달라졌을 가능성이 높습니다)."
)

# =========================================================================
# Part B. 우리의 확장: 실제 인건비/전기요금(계절)을 목적함수 계수로 사용해
#          데이터셋의 모든 시간(행)에 대해 반복 실행 (scipy, 외부 프로세스 없음)
# =========================================================================
print("\n" + "=" * 70)
print("[Part B] 확장 재현 — 실제 인건비/전기요금(계절)을 비용계수로 사용")
print("(※ 가이드북에 없는 우리의 해석 — 참고용)")
print("=" * 70)

wages = df["인건비"].to_numpy()
elec_rates = df["전기요금(계절)"].to_numpy()
n = len(df)

P_electric_arr = np.empty(n)
P_human_arr = np.empty(n)
cost_arr = np.empty(n)

for i in range(n):
    res = linprog(
        [elec_rates[i], wages[i]],  # [P_electric 계수, P_human 계수]
        A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
    )
    P_electric_arr[i] = res.x[0]
    P_human_arr[i] = res.x[1]
    cost_arr[i] = res.fun

res_df = pd.DataFrame(
    {
        "index": df.index,
        "생산량": df["생산량"].to_numpy(),
        "인건비": wages,
        "전기요금(계절)": elec_rates,
        "추천_P_electric": P_electric_arr,
        "추천_P_human": P_human_arr,
        "최소비용": cost_arr,
    }
)
res_df.to_csv("results/lp_optimization_result.csv", index=False)

total_cost = res_df["최소비용"].sum()
mean_human = res_df["추천_P_human"].mean()
mean_electric = res_df["추천_P_electric"].mean()

print(f"전체 {len(res_df)}개 시간(행)에 대한 LP 반복 실행 완료")
print(f"평균 추천 공장인원(P_human)     : {mean_human:.2f}")
print(f"평균 추천 전기설비가동(P_electric): {mean_electric:.2f}")
print(f"전체 시간 합산 최소비용 총계     : {total_cost:.4f}")

# 가이드북 예시(생산량 정규화값 0.17183770899999998)와 가장 가까운 행 참고 출력
prod_min, prod_max = df["생산량"].min(), df["생산량"].max()
res_df["생산량_정규화"] = (res_df["생산량"] - prod_min) / (prod_max - prod_min)
closest = res_df.iloc[(res_df["생산량_정규화"] - 0.17183770899999998).abs().argsort()[:1]]
print("\n[참고] 가이드북 예시 생산량(정규화 0.1718...)에 가장 가까운 실제 행:")
print(closest.to_string(index=False))

with open("results/lp_metrics.txt", "w") as f:
    f.write(f"static_cost={res_static.fun:.4f}\n")
    f.write(f"static_P_human={P_human_static}\n")
    f.write(f"static_P_electric={P_electric_static}\n")
    f.write(f"batch_total_cost={total_cost:.4f}\n")
    f.write(f"batch_mean_P_human={mean_human:.4f}\n")
    f.write(f"batch_mean_P_electric={mean_electric:.4f}\n")

print("\n[정보] lp_optimization_result.csv / lp_metrics.txt 저장 완료")