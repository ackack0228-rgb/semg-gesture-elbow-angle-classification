# Step 2 · 3주차 심화 4 : 계층적 분류 (본 강의자료 54쪽, 난이도 ★★★ — 수업용 확장)
# 목적: 손동작과 팔꿈치 각도를 두 단계로 나눠 예측하고 8클래스 단일 분류(본문 5-fold)와 비교한다.
#   1단계  손동작 4종 분류기 (학습 폴드 전체)
#   2단계  손동작마다 따로 학습한 각도 2종 분류기 (그 손동작의 학습 시행만) — 1단계가 고른 손동작의 분류기를 쓴다
#   최종 클래스 = 각도 x 4 + 손동작  (common3 의 클래스 번호 규칙)
# 구성 A는 단계마다 윈도우 19개 예측을 다수결한다.
# XGBoost 각도 분류기는 클래스가 2개라 objective 를 binary:logistic 으로 바꾼다 (나머지 설정은 같다).
#
# 실행 방법: python ext4_hierarchical.py   (약 8분)
# 결과: ext4_hierarchical.md

import os, time
import numpy as np
from scipy.stats import ttest_rel
from common3 import (BASE_DIR, COMPS, COMP_NAME, MODELS, load_all_trials, cv_splits, make_models, fit_predict)

T, y, g, s = load_all_trials()
folds = cv_splits(y, s)
cv = np.load(os.path.join(BASE_DIR, 'cv_results.npz'))


def model(k, binary=False):
    m = make_models()[k]
    if binary and k == 'XGB':
        m.set_params(objective='binary:logistic')
    return m


res = {}
t0 = time.time()
for c in COMPS:
    for k in MODELS:
        acc, acc_g, acc_a_given_g = [], [], []
        for tr, va in folds:
            pg, _, _, _ = fit_predict(c, model(k), T[tr], y[tr] % 4, T[va])          # 1단계: 손동작
            pa = np.zeros(len(va), dtype=int)
            for gi in range(4):
                sel_tr = tr[y[tr] % 4 == gi]
                sel_va = np.where(pg == gi)[0]
                if len(sel_va) == 0: continue
                pa[sel_va], _, _, _ = fit_predict(c, model(k, True), T[sel_tr], y[sel_tr] // 4, T[va][sel_va])
            p = pa * 4 + pg
            acc.append(np.mean(p == y[va])); acc_g.append(np.mean(pg == y[va] % 4))
            ok = pg == y[va] % 4
            acc_a_given_g.append(np.mean(pa[ok] == y[va][ok] // 4))
        res[(c, k)] = (np.array(acc), np.array(acc_g), np.array(acc_a_given_g))
        print(f'{c}_{k}: 계층 {np.mean(acc) * 100:.2f}%  (단일 {cv[f"acc_{c}_{k}"].mean() * 100:.2f}%)', flush=True)
print(f'소요 {time.time() - t0:.0f}초')

with open(os.path.join(BASE_DIR, 'ext4_hierarchical.md'), 'w', encoding='utf-8') as f:
    f.write('# 심화 4. 계층적 분류 (손동작 → 각도) vs 8클래스 단일 분류 (5-fold, 시행 단위 정확도 %)\n\n')
    f.write('| 표본 구성 | 모델 | 단일 8클래스 | 계층적 8클래스 | 차이 (%p) | p | 1단계 손동작 정확도 | 2단계 각도 정확도 (손동작을 맞힌 시행) |\n|---|---|---|---|---|---|---|---|\n')
    for c in COMPS:
        for k in MODELS:
            a, ag, aa = res[(c, k)]
            b = cv[f'acc_{c}_{k}']
            f.write(f'| {COMP_NAME[c]} | {k} | {b.mean() * 100:.2f} ± {b.std() * 100:.2f} | {a.mean() * 100:.2f} ± {a.std() * 100:.2f} | '
                    f'{(a.mean() - b.mean()) * 100:+.2f} | {ttest_rel(a, b).pvalue:.4f} | {ag.mean() * 100:.2f} | {aa.mean() * 100:.2f} |\n')
print(open(os.path.join(BASE_DIR, 'ext4_hierarchical.md'), encoding='utf-8').read())
