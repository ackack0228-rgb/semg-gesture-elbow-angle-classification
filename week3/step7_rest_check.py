# Step 2 · 3주차 실습 · 한계 점검 (제출물 ⑦ 한계 서술의 근거)
# 질문: 손동작을 하지 않는 휴식 구간만 보여 줘도 분류기가 클래스를 맞히는가?
#   이 데이터는 피험자 x 클래스마다 30시행을 한 파일에 이어서 기록했다 (1주차 확인). 시행 단위로 나누면
#   같은 파일(같은 기록 구간)의 다른 시행이 학습과 평가에 함께 들어간다. 휴식 구간에서도 클래스가 맞는다면
#   분류기는 동작이 아니라 "어느 기록 파일인지"(전극 상태 · 잡음 · 기저 수준 등)를 함께 배우고 있다는 뜻이다.
#
# 방법: 분석 구간(mark + 0.5~2.5초) 대신 휴식 구간(mark + 4~5초, 1초 → 200 ms 윈도우 9개)으로 똑같이 특징을 만들고,
#       본문과 같은 5-fold 폴드 · 같은 모델 · 같은 세 표본 구성으로 평가한다. 피험자를 통째로 빼는 LOSO도 함께 본다
#       (LOSO에서는 평가 파일이 학습에 전혀 없으므로 파일 식별로는 맞힐 수 없다).
#
# 실행 방법: python step7_rest_check.py   (약 3분, 처음 한 번은 CSV 읽기 포함)
# 결과: rest_check.md

import os, time
import numpy as np
from common3 import (BASE_DIR, COMPS, COMP_NAME, MODELS, load_all_trials, load_segments, load_rest_segments,
                     segment_features, cv_splits, make_models, fit_predict)

T, y, g, s = load_all_trials()
R = load_rest_segments()
TR = segment_features(R)                     # (1920, 9, 25)
print('휴식 구간 특징:', TR.shape)

# 휴식 구간이 정말 휴식인지: 채널별 RMS를 분석 구간과 비교
S = load_segments()
rms_ratio = np.sqrt((R ** 2).mean(axis=(0, 1))) / np.sqrt((S ** 2).mean(axis=(0, 1)))
print('휴식 / 분석 구간 RMS 비 (채널별):', rms_ratio.round(3))

folds = cv_splits(y, s)
subs = sorted(set(s))
res = {}
t0 = time.time()
for c in COMPS:
    for k in MODELS:
        p_cv = np.full(len(y), -1)
        for tr, va in folds:
            p_cv[va] = fit_predict(c, make_models()[k], TR[tr], y[tr], TR[va])[0]
        p_lo = np.full(len(y), -1)
        for u in subs:
            tr, te = np.where(s != u)[0], np.where(s == u)[0]
            p_lo[te] = fit_predict(c, make_models()[k], TR[tr], y[tr], TR[te])[0]
        res[(c, k)] = (p_cv, p_lo)
        print(f'{c}_{k}: 5-fold {np.mean(p_cv == y) * 100:.2f}%  LOSO {np.mean(p_lo == y) * 100:.2f}%', flush=True)
print(f'소요 {time.time() - t0:.0f}초')

cv = np.load(os.path.join(BASE_DIR, 'cv_results.npz'))
with open(os.path.join(BASE_DIR, 'rest_check.md'), 'w', encoding='utf-8') as f:
    f.write('# 한계 점검: 휴식 구간(mark + 4~5초)만으로 분류 (1,920시행)\n\n')
    f.write('휴식 구간의 RMS는 분석 구간(mark + 0.5~2.5초)의 '
            + ', '.join(f'{n} {r:.2f}' for n, r in zip(['ECR', 'ED', 'FDS', 'Biceps', 'Triceps'], rms_ratio))
            + ' 배 (채널별, 전체 시행).\n\n')
    f.write('우연 수준: 8클래스 12.5%, 동작만 25%, 각도만 50%.\n\n')
    f.write('| 표본 구성 | 모델 | 분석 구간 5-fold (본문) | 휴식 구간 5-fold 8클래스 | 동작만 | 각도만 | 휴식 구간 LOSO 8클래스 | 동작만 | 각도만 |\n'
            '|---|---|---|---|---|---|---|---|---|\n')
    for c in COMPS:
        for k in MODELS:
            p_cv, p_lo = res[(c, k)]
            f.write(f'| {COMP_NAME[c]} | {k} | {cv[f"acc_{c}_{k}"].mean() * 100:.2f} | '
                    f'**{np.mean(p_cv == y) * 100:.2f}** | {np.mean(p_cv % 4 == y % 4) * 100:.2f} | {np.mean(p_cv // 4 == y // 4) * 100:.2f} | '
                    f'**{np.mean(p_lo == y) * 100:.2f}** | {np.mean(p_lo % 4 == y % 4) * 100:.2f} | {np.mean(p_lo // 4 == y // 4) * 100:.2f} |\n')
    f.write('\n5-fold (시행 단위)는 평가 시행과 같은 파일의 다른 시행이 학습에 들어가고, LOSO는 평가 피험자의 파일이 학습에 전혀 없다.\n')
print(open(os.path.join(BASE_DIR, 'rest_check.md'), encoding='utf-8').read())
