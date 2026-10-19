# Step 2 · 3주차 실습 · 단계 4 : 5-fold 교차검증 (제출물 ③ 폴드별 값 + 평균 ± 표준편차)
# 목적: 세 모델 x 세 표본 구성의 5-fold 평균 정확도와 표준편차를 구한다.
#
# 슬라이드 12쪽과 같은 원칙
#   - 시행 단위로 나눈다 (한 시행의 윈도우 19개는 항상 같은 폴드)  — StratifiedKFold(5, shuffle=True, random_state=42)
#   - 특징은 미리 만들어 두고 인덱스로 나눈다 (2주차 step5_features.npz, 표준화 전) — 슬라이드 16쪽 "교차검증이 너무 느림"
#   - 각 학습 폴드에서만 scaler 를 fit 한다 (common3.fit)
#   - 모델은 폴드마다 새로 만든다 (슬라이드의 clone(proto) 와 같은 효과)
# 층화 기준은 피험자 x 클래스 (2주차 hold-out 과 같은 기준, 클래스 비율도 자동으로 유지된다).
# 표준편차는 슬라이드 코드와 같이 np.std (모집단 표준편차, ddof=0). 표본 표준편차(ddof=1)도 함께 적는다.
#
# 실행 방법: python step4_cv.py   (약 6분, 대부분 구성 A의 SVM)
# 결과: cv_results.npz (단계 5 입력), cv_results.md

import os, time
import numpy as np
from common3 import (BASE_DIR, COMPS, COMP_NAME, MODELS, SHORT, load_all_trials, cv_splits, make_models,
                     fit_predict)

T, y, g, s = load_all_trials()
folds = cv_splits(y, s)
print('전체 시행:', len(y), ' 폴드별 검증 시행 수:', [len(v) for _, v in folds])
# 누수 점검: 폴드끼리 검증 시행이 겹치지 않고, 합치면 전체가 된다
allv = np.concatenate([v for _, v in folds])
print('검증 시행 중복 없음 · 전체 포함:', len(set(allv)) == len(allv) == len(y))

acc = {f'{c}_{k}': [] for c in COMPS for k in MODELS}
acc_w = {k: [] for k in MODELS}                       # 구성 A의 윈도우 단위
fit_t = {f'{c}_{k}': [] for c in COMPS for k in MODELS}
oof = {f'{c}_{k}': np.full(len(y), -1) for c in COMPS for k in MODELS}
t_all = time.time()
for fi, (tr_i, va_i) in enumerate(folds):
    for c in COMPS:
        for k, m in make_models().items():
            pt, pw, _, t = fit_predict(c, m, T[tr_i], y[tr_i], T[va_i])
            key = f'{c}_{k}'
            acc[key].append(np.mean(pt == y[va_i])); fit_t[key].append(t); oof[key][va_i] = pt
            if pw is not None:
                acc_w[k].append(np.mean(pw == np.repeat(y[va_i], T.shape[1])))
            print(f'fold {fi + 1} {key}: {acc[key][-1] * 100:.2f}%  (학습 {t:.1f}초)', flush=True)
print(f'전체 소요 {time.time() - t_all:.0f}초\n')

for c in COMPS:
    for k in MODELS:
        v = np.array(acc[f'{c}_{k}'])
        print(f'{COMP_NAME[c]} {k}: {np.mean(v) * 100:.2f} ± {np.std(v) * 100:.2f} %')
for k in MODELS:
    print(f'A 윈도우 단위 {k}: {np.mean(acc_w[k]) * 100:.2f} ± {np.std(acc_w[k]) * 100:.2f} %')

np.savez(os.path.join(BASE_DIR, 'cv_results.npz'), y=y, g=g, s=s,
         **{f'acc_{k}': np.array(v) for k, v in acc.items()},
         **{f'accw_{k}': np.array(v) for k, v in acc_w.items()},
         **{f'fit_{k}': np.array(v) for k, v in fit_t.items()},
         **{f'oof_{k}': v for k, v in oof.items()})

PAPER = {'RF': (96.72, 0.58), 'XGB': (96.15, 0.56), 'SVM': (95.83, 0.79)}   # 논문 Table 3 (본 강의자료 40쪽)
SUBJ_PAPER = {'RF': '94.2~100', 'SVM': '87.5~100', 'XGB': '94.6~100'}       # 논문 Table 4 범위 (42쪽)
with open(os.path.join(BASE_DIR, 'cv_results.md'), 'w', encoding='utf-8') as f:
    f.write('# 5-fold 층화 교차검증 (시행 단위, 1,920시행, 폴드마다 검증 384시행)\n\n')
    f.write('분할: StratifiedKFold(5, shuffle=True, random_state=42), 층 = 피험자 x 클래스. 표준화는 폴드마다 학습 폴드로만 fit.\n\n')
    f.write('## 폴드별 정확도 (%, 시행 단위)\n\n| 표본 구성 | 모델 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | 평균 ± 표준편차 (ddof=0) | 표본 표준편차 (ddof=1) | 논문 Table 3 |\n'
            '|---|---|---|---|---|---|---|---|---|---|\n')
    for c in COMPS:
        for k in MODELS:
            v = np.array(acc[f'{c}_{k}']) * 100
            f.write(f'| {COMP_NAME[c]} | {k} | ' + ' | '.join(f'{x:.2f}' for x in v)
                    + f' | **{v.mean():.2f} ± {v.std():.2f}** | {v.std(ddof=1):.2f} | {PAPER[k][0]:.2f} ± {PAPER[k][1]:.2f} |\n')
    f.write('\n## 구성 A의 윈도우 단위 정확도 (%, 폴드마다 검증 7,296윈도우)\n\n| 모델 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | 평균 ± 표준편차 |\n|---|---|---|---|---|---|---|\n')
    for k in MODELS:
        v = np.array(acc_w[k]) * 100
        f.write(f'| {k} | ' + ' | '.join(f'{x:.2f}' for x in v) + f' | {v.mean():.2f} ± {v.std():.2f} |\n')
    f.write('\n## 표본 구성별 순위 (평균 정확도 높은 순)\n\n')
    for c in COMPS:
        order = sorted(MODELS, key=lambda k: -np.mean(acc[f'{c}_{k}']))
        f.write(f'- {COMP_NAME[c]}: ' + ' > '.join(f'{k} {np.mean(acc[f"{c}_{k}"]) * 100:.2f}' for k in order) + '\n')
    f.write('- 논문 Table 3: RF 96.72 > XGB 96.15 > SVM 95.83\n')

    f.write('\n## 피험자별 정확도 (%, 5-fold 검증 예측을 피험자별로 모음, 피험자마다 240시행) — 논문 Table 4에 해당\n\n')
    subs = sorted(set(s))
    f.write('| 표본 구성 | 모델 | ' + ' | '.join(subs) + ' | 최소~최대 | 논문 범위 |\n|---|---|' + '---|' * (len(subs) + 2) + '\n')
    for c in COMPS:
        for k in MODELS:
            p = oof[f'{c}_{k}']
            v = [np.mean(p[s == u] == y[s == u]) * 100 for u in subs]
            f.write(f'| {COMP_NAME[c]} | {k} | ' + ' | '.join(f'{x:.1f}' for x in v)
                    + f' | {min(v):.1f}~{max(v):.1f} | {SUBJ_PAPER[k]} |\n')
    f.write('\n피험자 이름은 데이터 파일의 a~h 표기 그대로다 (논문의 GYJ 등 이니셜과의 대응은 알 수 없다).\n')

    f.write('\n## 폴드별 학습 시간 평균 (초, 참고 — 정식 측정은 단계 6)\n\n| 표본 구성 | SVM | RF | XGB |\n|---|---|---|---|\n')
    for c in COMPS:
        f.write(f'| {COMP_NAME[c]} | ' + ' | '.join(f'{np.mean(fit_t[f"{c}_{k}"]):.2f}' for k in MODELS) + ' |\n')
print('저장: cv_results.npz, cv_results.md')
