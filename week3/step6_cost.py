# Step 2 · 3주차 실습 · 단계 6 : 계산 비용 (제출물 ⑥ 학습 시간 · 추론 시간 표)
# 목적: 정확도 말고 다른 축(학습 시간 · 추론 시간 · 모델 크기)을 잰다.
#
# 측정 방법
#   - 학습 시간: hold-out 학습셋 전체로 fit 하는 시간 (슬라이드 14쪽). 3번 재고 중앙값을 쓴다 (첫 번째 값도 함께 기록).
#   - 추론 시간 (슬라이드 방식): predict(표본 1개) 를 100번 반복한 평균 ms/표본.
#   - 묶음 추론: 테스트셋 전체를 한 번에 predict 한 시간 / 표본 수 (참고).
#   - 구성 A는 표본 = 윈도우 1개라, 시행 하나를 판정하려면 윈도우 19개를 예측하고 다수결해야 한다 → "시행 1개 판정" 열.
#   - 모델 크기: pickle 로 저장했을 때 바이트 수 (MB = 10^6 바이트, scaler 제외).
#   - 특징 추출: 2초 구간 하나에서 윈도우 19개 + 25차원 특징을 만드는 시간 (본 강의자료 43쪽 "특징 추출 시간은 별도 평가").
# 다른 프로그램이 CPU를 쓰지 않을 때 실행해야 한다 (step4_cv.py 와 동시에 돌리지 말 것).
#
# 실행 방법: python step6_cost.py
# 결과: cost.md

import os, sys, time, pickle
import numpy as np
from common3 import (BASE_DIR, WEEK2_DIR, COMPS, COMP_NAME, MODELS, load_holdout, make_models, represent,
                     vote, env_info, load_segments)
from sklearn.preprocessing import StandardScaler

(Ta, ya, ga, sa), (Tb, yb, gb, sb) = load_holdout()
REPEAT = 3
res, rf1_ms = {}, {}
for c in COMPS:
    sc = StandardScaler().fit(represent(Ta, c))
    Xtr, Xte = sc.transform(represent(Ta, c)), sc.transform(represent(Tb, c))
    ytr = np.repeat(ya, Ta.shape[1]) if c == 'A' else ya
    for k in MODELS:
        times = []
        for r in range(REPEAT):
            m = make_models()[k]
            t0 = time.perf_counter(); m.fit(Xtr, ytr); times.append(time.perf_counter() - t0)
        # 슬라이드 방식: 표본 1개 predict 100번
        t0 = time.perf_counter()
        for _ in range(100): m.predict(Xte[:1])
        infer_ms = (time.perf_counter() - t0) * 10
        # 묶음 추론
        t0 = time.perf_counter(); m.predict(Xte); batch_ms = (time.perf_counter() - t0) * 1000 / len(Xte)
        # 시행 1개 판정 (구성 A만 의미 있음: 윈도우 19개 + 다수결)
        if c == 'A':
            t0 = time.perf_counter()
            for i in range(100): vote(m.predict(Xte[i * 19:(i + 1) * 19]), 19)
            trial_ms = (time.perf_counter() - t0) * 10
        else:
            trial_ms = infer_ms
        size = len(pickle.dumps(m)) / 1e6
        # RF는 n_jobs=-1 (논문 설정) 이면 predict 를 부를 때마다 CPU 8개에 일을 나눠 주는 준비 시간이 든다.
        # 표본 1개씩 예측할 때 그 시간이 얼마인지 보려고, 학습이 끝난 같은 모델을 n_jobs=1 로 바꿔 한 번 더 잰다.
        if k == 'RF':
            m.set_params(n_jobs=1)
            t0 = time.perf_counter()
            for _ in range(100): m.predict(Xte[:1])
            rf1_ms[c] = (time.perf_counter() - t0) * 10
            m.set_params(n_jobs=-1)
        res[(c, k)] = (times[0], float(np.median(times)), min(times), max(times), infer_ms, batch_ms, trial_ms, size)
        print(f'{c} {k}: 학습 {np.median(times):.3f}s (첫 측정 {times[0]:.3f}s), 추론 {infer_ms:.3f} ms/표본, '
              f'묶음 {batch_ms:.4f} ms/표본, 시행 판정 {trial_ms:.3f} ms, 크기 {size:.2f} MB', flush=True)

# 특징 추출 시간 (2초 구간 1개 -> 19 x 25)
sys.path.insert(0, WEEK2_DIR); sys.path.insert(0, os.path.join(BASE_DIR, '..', 'week1'))
from step3_window import make_windows
from step4_features import td_features
S = load_segments()
t0 = time.perf_counter()
for seg in S[:200]:
    np.stack([td_features(w) for w in make_windows(seg)])
feat_ms = (time.perf_counter() - t0) * 1000 / 200
print(f'특징 추출: {feat_ms:.3f} ms / 시행 (윈도우 19개)')

PAPER = {'RF': (0.313, 0.036, 4.31), 'SVM': (0.334, 0.382, 4.36), 'XGB': (19.846, 0.031, 3.59)}   # 논문 Table 5 (43쪽)
with open(os.path.join(BASE_DIR, 'cost.md'), 'w', encoding='utf-8') as f:
    f.write('# 계산 비용 (hold-out 학습셋 1,536시행으로 학습)\n\n')
    f.write(f'측정 환경: {env_info()} (GPU 미사용)\n\n')
    f.write(f'| 표본 구성 | 모델 | 학습 표본 수 | 학습 시간 (초, {REPEAT}회 중앙값) | 학습 시간 범위 | 추론 (ms/표본, 1개씩 100회) | 묶음 추론 (ms/표본) | 시행 1개 판정 (ms) | 모델 크기 (MB) |\n'
            '|---|---|---|---|---|---|---|---|---|\n')
    for c in COMPS:
        n = len(represent(Ta, c))
        for k in MODELS:
            t1, tm, tmin, tmax, inf, bat, tri, size = res[(c, k)]
            f.write(f'| {COMP_NAME[c]} | {k} | {n} | {tm:.3f} | {tmin:.3f}~{tmax:.3f} | {inf:.3f} | {bat:.4f} | {tri:.3f} | {size:.2f} |\n')
    f.write('\n"시행 1개 판정": 구성 A는 윈도우 19개를 예측한 뒤 다수결하는 시간, B · C는 표본 1개 = 시행 1개라 추론 시간과 같다.\n')
    f.write('\nRF를 학습한 뒤 n_jobs=1 로만 바꿔 표본 1개씩 100회 예측: '
            + ', '.join(f'구성 {c} {v:.3f} ms' for c, v in rf1_ms.items())
            + '. n_jobs=-1 과의 차이가 표본 1개마다 CPU 여러 개에 일을 나눠 주는 준비 시간이고, '
            'n_jobs=1 에서도 남는 시간은 트리 100개를 하나씩 호출하는 비용이다. 묶음 추론에서는 이 고정 비용이 표본 수로 나뉘어 작아진다.\n')
    f.write(f'\n특징 추출 (2초 구간 1개 → 윈도우 19개 x 25차원, 2주차 td_features): **{feat_ms:.3f} ms / 시행** (200시행 평균)\n')
    f.write('\n## 논문 Table 5 (본 강의자료 43쪽, 측정 환경 다름 — 직접 비교 불가)\n\n| 모델 | 학습 시간 (초) | 샘플당 추론 (ms) | 모델 크기 (MB) |\n|---|---|---|---|\n')
    for k in ['RF', 'SVM', 'XGB']:
        f.write(f'| {k} | {PAPER[k][0]} | {PAPER[k][1]} | {PAPER[k][2]} |\n')
print('저장: cost.md')
