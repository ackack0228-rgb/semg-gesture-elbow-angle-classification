# Step 2 · 3주차 실습 · 단계 3 : 혼동행렬 (제출물 ⑤ confusion8.png + 오류 구조 해석)
# 목적: 어떤 클래스끼리 헷갈리는지 찾고, 오류가 "같은 동작의 다른 각도"인지 "다른 동작"인지 나눠 센다.
#
# 실행 방법: python step3_confusion.py   (step2_holdout.py 를 먼저 실행)
# 결과: confusion8.png         본문 표본 구성 C의 RF · SVM · XGBoost (논문 Fig. 5와 같은 배치) — 제출 그림
#       confusion8_rf_abc.png  RF (슬라이드 코드의 모델) 를 표본 구성 A · B · C 로 나란히
#       confusion8_all.png     9개 모델 전부 (행 = 표본 구성, 열 = 모델)
#       confusion_svm.png / confusion_rf.png / confusion_xgb.png   본문 구성 C의 모델별 그림 (README 3절)
#       confusion_analysis.md  (끝에 구성 C의 클래스별 Precision / Recall / F1 표)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from common3 import BASE_DIR, COMPS, COMP_NAME, MODELS, SHORT, CLASSES, error_types, draw_confusion, save_single_confusion

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

d = np.load(os.path.join(BASE_DIR, 'holdout_pred.npz'))
y = d['yte']
L = SHORT                          # 실제 인코딩 순서 (0 = 30_fist ... 7 = 60_supination), 논문 Fig. 5 표기
cms = {f'{c}_{k}': confusion_matrix(y, d[f'{c}_{k}'], labels=range(8)) for c in COMPS for k in MODELS}


def draw(ax, cm, title):
    return draw_confusion(ax, cm, title)


def row(keys, titles, fname, suptitle):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, key, t in zip(axes, keys, titles):
        cm = cms[key]
        im = draw(ax, cm, f'{t}\n정확도 {np.trace(cm)}/{cm.sum()} = {np.trace(cm) / cm.sum() * 100:.2f}%')
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(suptitle, fontsize=11)
    plt.savefig(os.path.join(BASE_DIR, fname), dpi=120, bbox_inches='tight')
    plt.close()


# 제출 그림: 본문 표본 구성 C의 세 모델 (논문 Fig. 5 (a) RF (b) SVM (c) XGBoost 와 같은 배치)
row(['C_RF', 'C_SVM', 'C_XGB'], ['(a) RF', '(b) SVM', '(c) XGBoost'], 'confusion8.png',
    f'hold-out 혼동행렬 · {COMP_NAME["C"]} (테스트 384시행) — 빨간 점선 = 같은 동작 · 다른 각도')
# 비교 그림: 슬라이드 코드와 같은 RF를 세 표본 구성으로
row(['A_RF', 'B_RF', 'C_RF'], [f'RF · {COMP_NAME[c]}' for c in COMPS], 'confusion8_rf_abc.png',
    'hold-out 혼동행렬 · RF의 표본 구성별 비교 (테스트 384시행) — 빨간 점선 = 같은 동작 · 다른 각도')

fig, axes = plt.subplots(3, 3, figsize=(16, 15))
for r, c in enumerate(COMPS):
    for q, k in enumerate(MODELS):
        cm = cms[f'{c}_{k}']
        draw(axes[r, q], cm, f'{k} · {COMP_NAME[c]}  ({np.trace(cm)}/{cm.sum()} = {np.trace(cm) / cm.sum() * 100:.2f}%)')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion8_all.png'), dpi=100)
plt.close()

# README용 모델별 그림 (본문 구성 C) — 같은 예측 · 같은 그리기 함수
FULL = {'SVM': 'SVM', 'RF': 'Random Forest', 'XGB': 'XGBoost'}
for k in MODELS:
    cm = cms[f'C_{k}']
    save_single_confusion(os.path.join(BASE_DIR, f'confusion_{k.lower()}.png'), cm,
                          f'{FULL[k]} · {COMP_NAME["C"]}\n정확도 {np.trace(cm)}/{cm.sum()} = {np.trace(cm) / cm.sum() * 100:.2f}%')

# ── 오류 구조 ────────────────────────────────────────────────────────────────────────────────
PAPER_ERR = {'RF': (13, 8), 'SVM': (18, 8), 'XGB': (11, 8)}     # 본 강의자료 47쪽 (전체 오류, 같은 동작 각도 오류)
with open(os.path.join(BASE_DIR, 'confusion_analysis.md'), 'w', encoding='utf-8') as f:
    f.write('# 혼동행렬 오류 구조 (hold-out, 테스트 384시행)\n\n')
    f.write('| 표본 구성 | 모델 | 정답 | 오류 | 같은 동작 · 다른 각도 | 같은 각도 · 다른 동작 | 둘 다 다름 | 논문 Fig. 5 (오류 / 각도 오류) |\n'
            '|---|---|---|---|---|---|---|---|\n')
    for c in COMPS:
        for k in MODELS:
            cm = cms[f'{c}_{k}']
            n, ga, an, both = error_types(y, d[f'{c}_{k}'])
            f.write(f'| {COMP_NAME[c]} | {k} | {np.trace(cm)} | {n} | {ga} ({ga / max(n, 1) * 100:.0f}%) | '
                    f'{an} | {both} | {PAPER_ERR[k][0]} / {PAPER_ERR[k][1]} |\n')

    f.write('\n## 가장 많이 틀린 쌍 (실제 → 예측, 2건 이상)\n\n| 표본 구성 | 모델 | 쌍 |\n|---|---|---|\n')
    for c in COMPS:
        for k in MODELS:
            cm = cms[f'{c}_{k}'].copy(); np.fill_diagonal(cm, 0)
            idx = np.argsort(cm.ravel())[::-1]
            pairs = [f'{L[i // 8]}→{L[i % 8]} {cm.ravel()[i]}' for i in idx if cm.ravel()[i] >= 2]
            f.write(f'| {COMP_NAME[c]} | {k} | {", ".join(pairs) if pairs else "없음"} |\n')

    f.write('\n## 클래스별 재현율 (%) — 48시행 중 맞힌 비율\n\n| 표본 구성 | 모델 | ' + ' | '.join(L) + ' |\n|---|---|' + '---|' * 8 + '\n')
    for c in COMPS:
        for k in MODELS:
            cm = cms[f'{c}_{k}']
            f.write(f'| {COMP_NAME[c]} | {k} | ' + ' | '.join(f'{cm[i, i] / cm[i].sum() * 100:.1f}' for i in range(8)) + ' |\n')

    f.write('\n## 동작별 · 각도별로 묶은 정확도 (%)\n\n'
            '"동작만" = 각도는 틀려도 손동작만 맞으면 정답 (4클래스, 우연 25%), "각도만" = 손동작은 틀려도 각도만 맞으면 정답 (2클래스, 우연 50%)\n\n'
            '| 표본 구성 | 모델 | 8클래스 | 동작만 | 각도만 | fist | open | pronation | supination |\n|---|---|---|---|---|---|---|---|---|\n')
    for c in COMPS:
        for k in MODELS:
            p = d[f'{c}_{k}']
            g_ok = (y % 4) == (p % 4); a_ok = (y // 4) == (p // 4)
            per_g = [np.mean((y == p)[(y % 4) == gi]) * 100 for gi in range(4)]
            f.write(f'| {COMP_NAME[c]} | {k} | {np.mean(y == p) * 100:.2f} | {g_ok.mean() * 100:.2f} | {a_ok.mean() * 100:.2f} | '
                    + ' | '.join(f'{v:.1f}' for v in per_g) + ' |\n')
with open(os.path.join(BASE_DIR, 'confusion_analysis.md'), 'a', encoding='utf-8') as f:
    f.write('\n## 클래스별 Precision / Recall / F1 (%) — 구성 C, hold-out (클래스마다 실제 48시행)\n\n'
            '| 클래스 | ' + ' | '.join(f'{k} P / R / F1' for k in MODELS) + ' | ' + ' | '.join(f'{k} 예측 수' for k in MODELS) + ' |\n'
            '|---|' + '---|' * (2 * len(MODELS)) + '\n')
    prf = {k: precision_recall_fscore_support(y, d[f'C_{k}'], labels=range(8), zero_division=0) for k in MODELS}
    for i, name in enumerate(L):
        f.write(f'| {name} | ' + ' | '.join(f'{prf[k][0][i] * 100:.1f} / {prf[k][1][i] * 100:.1f} / {prf[k][2][i] * 100:.1f}' for k in MODELS)
                + ' | ' + ' | '.join(str(int((d[f'C_{k}'] == i).sum())) for k in MODELS) + ' |\n')
print(open(os.path.join(BASE_DIR, 'confusion_analysis.md'), encoding='utf-8').read())
print('저장: confusion8.png, confusion8_all.png, confusion_{svm,rf,xgb}.png, confusion_analysis.md')
