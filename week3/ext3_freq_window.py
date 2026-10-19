# Step 2 · 3주차 심화 3 : 주파수 특징 추가 · 윈도우 길이 실험 (본 강의자료 54쪽, 둘 다 난이도 ★★)
#   (1) 주파수 특징: 시간영역 25차원에 MNF · MDF (5채널씩 10차원) 를 더한 35차원 — 각도 오분류가 줄어드는지
#       (논문의 향후 과제 "주파수 · 시간–주파수 특징 추가", 본 강의자료 48쪽)
#   (2) 윈도우 길이: 100 / 200 / 300 ms (모두 50% 오버랩) — 성능과 지연의 절충 (수업용 확장)
# 평가: 본문과 같은 5-fold, 시행 단위 정확도. 구성 A는 윈도우 단위 정확도도 적는다
#       (윈도우 하나로 바로 판정한다면 판정 지연 = 윈도우 길이).
#
# 실행 방법: python ext3_freq_window.py   (약 30분, 대부분 100 ms 윈도우의 구성 A SVM)
# 결과: ext3_freq_window.md

import os, time
import numpy as np
from scipy.stats import ttest_rel
from common3 import (BASE_DIR, COMPS, COMP_NAME, MODELS, load_all_trials, load_segments, segment_features,
                     cv_splits, make_models, fit_predict, error_types)

T0, y, g, s = load_all_trials()
folds = cv_splits(y, s)
S = load_segments()
print('2초 구간:', S.shape)

# 확인: 캐시한 구간에서 다시 만든 200 ms 특징 = 2주차 특징 (같은 함수 · 같은 순서)
T200 = segment_features(S)
print('200 ms 특징 = 2주차 step5_features:', np.allclose(T200, T0))

CONFIGS = {
    '시간영역 25차원 · 200 ms': lambda: T200,
    '시간영역+MNF·MDF 35차원 · 200 ms': lambda: segment_features(S, extra=True),
    '시간영역 25차원 · 100 ms': lambda: segment_features(S, 100, 50),
    '시간영역 25차원 · 300 ms': lambda: segment_features(S, 300, 150),
}
res = {}
t_all = time.time()
for name, make in CONFIGS.items():
    T = make()
    print(f'\n[{name}] 시행 배열 {T.shape}', flush=True)
    for c in COMPS:
        for k in MODELS:
            a8, aa, aw, errs = [], [], [], [0, 0]
            for tr, va in folds:
                t0 = time.time()
                pt, pw, _, ft = fit_predict(c, make_models()[k], T[tr], y[tr], T[va])
                a8.append(np.mean(pt == y[va])); aa.append(np.mean(pt // 4 == y[va] // 4))
                n, ga, _, _ = error_types(y[va], pt); errs[0] += n; errs[1] += ga
                if pw is not None:
                    aw.append(np.mean(pw == np.repeat(y[va], T.shape[1])))
            res[(name, c, k)] = (np.array(a8), np.array(aa), np.array(aw), errs, T.shape)
            print(f'  {c}_{k}: {np.mean(a8) * 100:.2f}%  각도 {np.mean(aa) * 100:.2f}%'
                  + (f'  윈도우 {np.mean(aw) * 100:.2f}%' if aw else ''), flush=True)
print(f'소요 {time.time() - t_all:.0f}초')

base = '시간영역 25차원 · 200 ms'
with open(os.path.join(BASE_DIR, 'ext3_freq_window.md'), 'w', encoding='utf-8') as f:
    f.write('# 심화 3. 주파수 특징 추가 · 윈도우 길이 (5-fold, 시행 단위 정확도 %)\n\n')
    f.write('MNF = 평균 주파수, MDF = 중앙 주파수 (윈도우 하나의 |FFT|² 스펙트럼, 채널마다). p = 기준(시간영역 25차원 · 200 ms)과의 대응표본 t-검정.\n\n')
    for title, names in [('## (1) 주파수 특징 추가', [base, '시간영역+MNF·MDF 35차원 · 200 ms']),
                         ('## (2) 윈도우 길이 (50% 오버랩)', ['시간영역 25차원 · 100 ms', base, '시간영역 25차원 · 300 ms'])]:
        f.write(f'\n{title}\n\n| 설정 | 시행당 윈도우 수 · 차원 | 표본 구성 | 모델 | 8클래스 평균 ± 표준편차 | 기준 대비 (%p) | p | 각도만 | 오류 중 같은 동작 · 다른 각도 | 구성 A 윈도우 단위 |\n'
                '|---|---|---|---|---|---|---|---|---|---|\n')
        for name in names:
            for c in COMPS:
                for k in MODELS:
                    a8, aa, aw, errs, shp = res[(name, c, k)]
                    b = res[(base, c, k)][0]
                    same = name == base
                    f.write(f'| {name} | {shp[1]} x {shp[2]} | {COMP_NAME[c]} | {k} | {a8.mean() * 100:.2f} ± {a8.std() * 100:.2f} | '
                            + ('—' if same else f'{(a8.mean() - b.mean()) * 100:+.2f}') + ' | '
                            + ('—' if same else f'{ttest_rel(a8, b).pvalue:.4f}') + ' | '
                            f'{aa.mean() * 100:.2f} | {errs[1]}/{errs[0]} | '
                            + (f'{aw.mean() * 100:.2f}' if len(aw) else '—') + ' |\n')
    f.write('\n300 ms (증분 150 ms) 는 2초 구간에 윈도우 12개가 들어가고 마지막 50 ms는 쓰이지 않는다.\n')
print(open(os.path.join(BASE_DIR, 'ext3_freq_window.md'), encoding='utf-8').read())
