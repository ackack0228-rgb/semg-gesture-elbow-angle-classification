# Step 2 · 3주차 실습 · 단계 5 : 통계 검정 (제출물 ④ 세 쌍의 p값과 해석 문장)
# 목적: 폴드별 정확도로 대응표본 t-검정을 해서 세 모델의 차이가 우연인지 판단한다.
#
# 슬라이드 13쪽 코드 그대로 (ttest_rel, 쌍 = combinations(models, 2), α = 0.05) 를 표본 구성마다 돌린다.
# 함께 적는 참고값
#   - 평균 차이와 그 95% 신뢰구간 (t 분포, 자유도 4)
#   - 5개 폴드 중 앞 모델이 더 높았던 폴드 수
#   - 보정 t-검정 (Nadeau & Bengio, 2003): 교차검증의 학습 폴드는 서로 75%씩 겹쳐 폴드 정확도가 독립이 아니다.
#     보통의 t-검정은 이 겹침을 무시해 분산을 작게 잡으므로, 분산에 (1/k + 검증 수/학습 수) 를 곱해 보정한다.
#     슬라이드 방식이 아니므로 판정은 슬라이드 방식(ttest_rel)으로 하고 이 값은 참고로만 쓴다.
#
# 실행 방법: python step5_ttest.py   (step4_cv.py 를 먼저 실행)
# 결과: ttest.md

import os
import numpy as np
from itertools import combinations
from scipy.stats import ttest_rel, t as tdist
from common3 import BASE_DIR, COMPS, COMP_NAME, MODELS

d = np.load(os.path.join(BASE_DIR, 'cv_results.npz'))
K, N_TE, N_TR = 5, 384, 1536
PAPER = {('RF', 'SVM'): 0.1193, ('RF', 'XGB'): 0.1609, ('SVM', 'XGB'): 0.3739}   # 논문 (본 강의자료 41쪽)


def paper_p(a, b):
    return PAPER.get((a, b), PAPER.get((b, a)))


rows = []
for c in COMPS:
    fold_acc = {k: d[f'acc_{c}_{k}'] for k in MODELS}
    print(f'[{COMP_NAME[c]}]')
    for a, b in combinations(fold_acc.keys(), 2):
        t, p = ttest_rel(fold_acc[a], fold_acc[b])
        if not np.isfinite(p): print(a, b, '검정 불가'); continue
        verdict = '유의함' if p < 0.05 else '유의하지 않음'
        print(f'{a} vs {b}: p = {p:.4f}  ({verdict})')
        diff = (fold_acc[a] - fold_acc[b]) * 100
        m, sd = diff.mean(), diff.std(ddof=1)
        half = tdist.ppf(0.975, K - 1) * sd / np.sqrt(K)
        t_c = m / np.sqrt((1 / K + N_TE / N_TR) * sd ** 2) if sd > 0 else np.nan
        p_c = 2 * tdist.sf(abs(t_c), K - 1) if np.isfinite(t_c) else np.nan
        rows.append((c, a, b, t, p, verdict, m, m - half, m + half, int((diff > 0).sum()), int((diff == 0).sum()), p_c))

# 표본 구성 사이 비교 (같은 모델, 같은 폴드) — 표본 구성 선택이 결과를 얼마나 바꾸는지
rows_c = []
for k in MODELS:
    for a, b in combinations(COMPS, 2):
        t, p = ttest_rel(d[f'acc_{a}_{k}'], d[f'acc_{b}_{k}'])
        rows_c.append((k, a, b, (d[f'acc_{a}_{k}'] - d[f'acc_{b}_{k}']).mean() * 100, p))

with open(os.path.join(BASE_DIR, 'ttest.md'), 'w', encoding='utf-8') as f:
    f.write('# 대응표본 t-검정 (5-fold 폴드별 정확도, α = 0.05)\n\n')
    f.write('| 표본 구성 | 비교 | 평균 차이 (%p) | 차이의 95% 신뢰구간 | 앞 모델이 높은 폴드 | t | p (ttest_rel) | 판정 | 참고: 보정 t-검정 p | 논문 p |\n'
            '|---|---|---|---|---|---|---|---|---|---|\n')
    for c, a, b, t, p, v, m, lo, hi, wins, ties, p_c in rows:
        f.write(f'| {COMP_NAME[c]} | {a} vs {b} | {m:+.2f} | {lo:+.2f} ~ {hi:+.2f} | {wins}/5'
                + (f' (동률 {ties})' if ties else '') + f' | {t:.3f} | **{p:.4f}** | {v} | {p_c:.4f} | {paper_p(a, b):.4f} |\n')
    f.write('\n평균 차이 = 앞 모델 − 뒤 모델. 신뢰구간이 0을 포함하면 p ≥ 0.05와 같은 뜻이다.\n')
    f.write('보정 t-검정: Nadeau & Bengio (2003)의 corrected resampled t-test, 분산 x (1/5 + 384/1536). 슬라이드 방식이 아니어서 참고로만 적었다.\n')
    f.write('\n## 참고: 표본 구성 사이 비교 (같은 모델, 같은 5개 폴드)\n\n| 모델 | 비교 | 평균 차이 (%p) | p (ttest_rel) |\n|---|---|---|---|\n')
    for k, a, b, m, p in rows_c:
        f.write(f'| {k} | {a} vs {b} | {m:+.2f} | {p:.4f} |\n')
print(open(os.path.join(BASE_DIR, 'ttest.md'), encoding='utf-8').read())
