# Step 2 · 3주차 심화 1 : LOSO 검증 (본 강의자료 54쪽, 난이도 ★★★ — 논문의 향후 과제)
# 목적: 한 피험자를 통째로 빼고 7명으로 학습한 뒤 빠진 1명을 평가한다 (8번 반복).
#       시행 단위 5-fold (같은 사람 · 같은 기록 파일의 다른 시행이 학습에 들어감) 와 달리
#       "처음 보는 사람"에 대한 성능이다.
#
# 실행 방법: python ext1_loso.py   (약 10분)
# 결과: ext1_loso.md, ext1_loso.npz

import os, time
import numpy as np
from common3 import (BASE_DIR, COMPS, COMP_NAME, MODELS, load_all_trials, make_models, fit_predict,
                     error_types)

T, y, g, s = load_all_trials()
subs = sorted(set(s))
acc = {f'{c}_{k}': [] for c in COMPS for k in MODELS}
oof = {f'{c}_{k}': np.full(len(y), -1) for c in COMPS for k in MODELS}
t0 = time.time()
for u in subs:
    tr, te = np.where(s != u)[0], np.where(s == u)[0]
    for c in COMPS:
        for k, m in make_models().items():
            pt, _, _, _ = fit_predict(c, m, T[tr], y[tr], T[te])
            acc[f'{c}_{k}'].append(np.mean(pt == y[te])); oof[f'{c}_{k}'][te] = pt
            print(f'평가 피험자 {u} {c}_{k}: {acc[f"{c}_{k}"][-1] * 100:.1f}%', flush=True)
print(f'소요 {time.time() - t0:.0f}초')
np.savez(os.path.join(BASE_DIR, 'ext1_loso.npz'), y=y, s=s, **{f'oof_{k}': v for k, v in oof.items()},
         **{f'acc_{k}': np.array(v) for k, v in acc.items()})

cv = np.load(os.path.join(BASE_DIR, 'cv_results.npz'))
with open(os.path.join(BASE_DIR, 'ext1_loso.md'), 'w', encoding='utf-8') as f:
    f.write('# 심화 1. LOSO 검증 (7명 학습 / 1명 평가 x 8회, 평가 피험자마다 240시행)\n\n')
    f.write('표준화는 매번 학습 피험자 7명의 표본으로만 fit. 모델 설정은 본문과 같다.\n\n')
    f.write('| 표본 구성 | 모델 | ' + ' | '.join(subs) + ' | LOSO 평균 ± 표준편차 | 5-fold (시행 단위) | 차이 (%p) |\n|---|---|'
            + '---|' * (len(subs) + 3) + '\n')
    for c in COMPS:
        for k in MODELS:
            v = np.array(acc[f'{c}_{k}']) * 100
            cvm = cv[f'acc_{c}_{k}'].mean() * 100
            f.write(f'| {COMP_NAME[c]} | {k} | ' + ' | '.join(f'{x:.1f}' for x in v)
                    + f' | **{v.mean():.2f} ± {v.std():.2f}** | {cvm:.2f} | {v.mean() - cvm:+.2f} |\n')
    f.write('\n## 동작만 · 각도만 맞힌 비율 (%, 8명 전체 1,920시행)\n\n'
            '| 표본 구성 | 모델 | 8클래스 | 동작만 (우연 25%) | 각도만 (우연 50%) | 오류 중 같은 동작 · 다른 각도 |\n|---|---|---|---|---|---|\n')
    for c in COMPS:
        for k in MODELS:
            p = oof[f'{c}_{k}']
            n, ga, an, both = error_types(y, p)
            f.write(f'| {COMP_NAME[c]} | {k} | {np.mean(p == y) * 100:.2f} | {np.mean(p % 4 == y % 4) * 100:.2f} | '
                    f'{np.mean(p // 4 == y // 4) * 100:.2f} | {ga}/{n} ({ga / max(n, 1) * 100:.0f}%) |\n')
print(open(os.path.join(BASE_DIR, 'ext1_loso.md'), encoding='utf-8').read())
