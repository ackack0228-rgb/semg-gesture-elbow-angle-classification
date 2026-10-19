# Step 2 · 3주차 심화 2 : 채널 어블레이션 (본 강의자료 54쪽, 난이도 ★★ — 수업용 확장)
# 목적: 상완 2채널(Biceps · Triceps)을 뺐을 때 각도 분류가 얼마나 떨어지는지 잰다.
#       팔꿈치 각도는 상완 근육이 버티는 자세라 상완 채널에 정보가 있을 것이라는 가설을 확인한다.
#       반대로 상완 2채널만 남긴 경우도 함께 본다.
#
# 특징 순서 (2주차 feature_order.md): 특징 f(0 RMS ~ 4 SSC)의 채널 c(0 ECR, 1 ED, 2 FDS, 3 Biceps, 4 Triceps) = f*5 + c
# 평가: 본문과 같은 5-fold (같은 폴드 · 같은 설정), 시행 단위 정확도
#
# 실행 방법: python ext2_channels.py   (약 10분)
# 결과: ext2_channels.md

import os, time
import numpy as np
from scipy.stats import ttest_rel
from common3 import (BASE_DIR, COMPS, COMP_NAME, MODELS, load_all_trials, cv_splits, make_models, fit_predict)

T, y, g, s = load_all_trials()
folds = cv_splits(y, s)
SETS = {'전체 5채널': [0, 1, 2, 3, 4], '전완 3채널 (상완 제외)': [0, 1, 2], '상완 2채널만': [3, 4]}

res = {}
t0 = time.time()
for name, chs in SETS.items():
    cols = [f * 5 + c for f in range(5) for c in chs]
    Tc = T[:, :, cols]
    for c in COMPS:
        for k in MODELS:
            a8, ag, aa = [], [], []
            for tr, va in folds:
                pt, _, _, _ = fit_predict(c, make_models()[k], Tc[tr], y[tr], Tc[va])
                a8.append(np.mean(pt == y[va])); ag.append(np.mean(pt % 4 == y[va] % 4)); aa.append(np.mean(pt // 4 == y[va] // 4))
            res[(name, c, k)] = (np.array(a8), np.array(ag), np.array(aa))
            print(f'{name} {c}_{k}: 8클래스 {np.mean(a8) * 100:.2f}  동작 {np.mean(ag) * 100:.2f}  각도 {np.mean(aa) * 100:.2f}', flush=True)
print(f'소요 {time.time() - t0:.0f}초')

base = '전체 5채널'
with open(os.path.join(BASE_DIR, 'ext2_channels.md'), 'w', encoding='utf-8') as f:
    f.write('# 심화 2. 채널 어블레이션 (5-fold, 시행 단위 정확도 %, 평균 ± 표준편차)\n\n')
    f.write('"동작만" = 손동작 4종만 채점 (우연 25%), "각도만" = 팔꿈치 각도 2종만 채점 (우연 50%). '
            'p = 전체 5채널과의 대응표본 t-검정 (같은 5개 폴드).\n\n')
    f.write('| 채널 | 표본 구성 | 모델 | 8클래스 | 동작만 | 각도만 | 각도만: 5채널 대비 (%p) | p (각도만) |\n|---|---|---|---|---|---|---|---|\n')
    for name in SETS:
        for c in COMPS:
            for k in MODELS:
                a8, ag, aa = res[(name, c, k)]
                b = res[(base, c, k)][2]
                p = ttest_rel(aa, b).pvalue if name != base else np.nan
                f.write(f'| {name} | {COMP_NAME[c]} | {k} | {a8.mean() * 100:.2f} ± {a8.std() * 100:.2f} | '
                        f'{ag.mean() * 100:.2f} | {aa.mean() * 100:.2f} | '
                        + ('—' if name == base else f'{(aa.mean() - b.mean()) * 100:+.2f}') + ' | '
                        + ('—' if name == base else f'{p:.4f}') + ' |\n')
print(open(os.path.join(BASE_DIR, 'ext2_channels.md'), encoding='utf-8').read())
