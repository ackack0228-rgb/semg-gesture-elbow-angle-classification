# Step 2 · 3주차 실습 · 단계 2 : hold-out 평가 (제출물 ② 성능 비교표)
# 목적: 정확도 · 정밀도 · 재현율 · F1을 세 모델 x 세 표본 구성에 대해 한 번에 구한다.
#
# 다중 클래스 평균 방식: macro (클래스별 값의 단순 평균) — 논문은 평균 방식을 밝히지 않았으므로 실습 선택이다.
# 테스트셋이 클래스마다 같은 수(시행 48개 · 윈도우 912개)라 weighted 평균과도 값이 같다.
#
# 실행 방법: python step2_holdout.py   (step1_train.py 를 먼저 실행)
# 결과: holdout_metrics.md, holdout_pred.npz (단계 3에서 사용)

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from common3 import BASE_DIR, COMPS, COMP_NAME, MODELS, load_holdout, load_model, predict

(Ta, ya, ga, sa), (Tb, yb, gb, sb) = load_holdout()

PAPER = {'RF': [0.9688, 0.9695, 0.9688, 0.9687], 'SVM': [0.9635, 0.9641, 0.9635, 0.9635],
         'XGB': [0.9557, 0.9567, 0.9557, 0.9559]}          # 논문 Table 2 (본 강의자료 38쪽)


def four(y, p):
    return [accuracy_score(y, p),
            precision_score(y, p, average='macro', zero_division=0),
            recall_score(y, p, average='macro'),
            f1_score(y, p, average='macro')]


rows, preds = [], {}
for comp in COMPS:
    for k in MODELS:
        s = load_model(f'{comp}_{k}')
        pt, pw, ties = predict(comp, s['scaler'], s['model'], Tb)
        rows.append([comp, k, 'trial'] + four(yb, pt) + [ties])
        preds[f'{comp}_{k}'] = pt
        if pw is not None:                                    # 구성 A: 윈도우 단위 (슬라이드 코드의 직접 출력)
            yw = np.repeat(yb, Tb.shape[1])
            rows.append([comp, k, 'window'] + four(yw, pw) + [0])
            preds[f'{comp}_{k}_window'] = pw

df = pd.DataFrame(rows, columns=['comp', 'model', 'unit', 'acc', 'precision', 'recall', 'f1', 'ties'])
pd.set_option('display.width', 120)
print(df.round(4).to_string(index=False))
np.savez(os.path.join(BASE_DIR, 'holdout_pred.npz'), yte=yb, gte=gb, ste=sb, **preds)

with open(os.path.join(BASE_DIR, 'holdout_metrics.md'), 'w', encoding='utf-8') as f:
    f.write('# hold-out 평가 (학습 1,536시행 / 테스트 384시행, 2주차 분할 그대로)\n\n')
    f.write('평균 방식: macro (실습 선택, 논문 미명시). 테스트셋이 클래스마다 같은 수라 weighted 평균과 같은 값이다.\n\n')
    f.write('## 시행 단위 (테스트 384개) — 논문 Table 2와 같은 단위\n\n')
    f.write('| 표본 구성 | 모델 | Accuracy | Precision | Recall | F1-score | 논문 Accuracy | 차이 (%p) |\n'
            '|---|---|---|---|---|---|---|---|\n')
    for r in df[df.unit == 'trial'].itertuples(index=False):
        f.write(f'| {COMP_NAME[r.comp]} | {r.model} | {r.acc:.4f} | {r.precision:.4f} | {r.recall:.4f} | '
                f'{r.f1:.4f} | {PAPER[r.model][0]:.4f} | {(r.acc - PAPER[r.model][0]) * 100:+.2f} |\n')
    f.write('\n## 구성 A의 윈도우 단위 (테스트 7,296개) — 슬라이드 코드가 바로 출력하는 값\n\n')
    f.write('| 모델 | Accuracy | Precision | Recall | F1-score | 다수결 동점 시행 수 |\n|---|---|---|---|---|---|\n')
    for k in MODELS:
        w = df[(df.unit == 'window') & (df.model == k)].iloc[0]
        t = df[(df.unit == 'trial') & (df.model == k) & (df.comp == 'A')].iloc[0]
        f.write(f'| {k} | {w.acc:.4f} | {w.precision:.4f} | {w.recall:.4f} | {w.f1:.4f} | {t.ties} |\n')
    f.write('\n동점이면 번호가 작은 클래스로 정했다.\n')
    f.write('\n## 논문 Table 2 (본 강의자료 38쪽)\n\n| 모델 | Accuracy | Precision | Recall | F1-score |\n|---|---|---|---|---|\n')
    for k in ['RF', 'SVM', 'XGB']:
        f.write(f'| {k} | ' + ' | '.join(f'{v:.4f}' for v in PAPER[k]) + ' |\n')
print('저장: holdout_metrics.md, holdout_pred.npz')

# ── 점검: 학습 표본의 순서만 바꾸면 RF 결과가 달라지는가 ─────────────────────────────────────
# RF는 트리마다 부트스트랩으로 표본 번호를 뽑으므로, 같은 표본이라도 순서가 바뀌면 뽑히는 표본이 달라진다.
# 학습 시행을 2주차 분할 순서(본문) 대신 시행 번호 순으로 정렬해 같은 RF를 다시 학습한다.
from common3 import make_models, fit_predict
o = np.argsort(ga)
with open(os.path.join(BASE_DIR, 'holdout_metrics.md'), 'a', encoding='utf-8') as f:
    f.write('\n## 점검: 학습 표본 순서만 바꾼 RF (시행 번호 순 정렬, 같은 표본 · 같은 설정)\n\n| 표본 구성 | 본문 (분할 순서) | 시행 번호 순 | 차이 (%p) |\n|---|---|---|---|\n')
    for comp in COMPS:
        pt = fit_predict(comp, make_models()['RF'], Ta[o], ya[o], Tb)[0]
        a0 = df[(df.comp == comp) & (df.model == 'RF') & (df.unit == 'trial')].acc.iloc[0]
        f.write(f'| {COMP_NAME[comp]} | {a0 * 100:.2f} | {np.mean(pt == yb) * 100:.2f} | {(np.mean(pt == yb) - a0) * 100:+.2f} |\n')
        print(f'RF 표본 순서 점검 {comp}: 본문 {a0 * 100:.2f}%  시행 번호 순 {np.mean(pt == yb) * 100:.2f}%')
