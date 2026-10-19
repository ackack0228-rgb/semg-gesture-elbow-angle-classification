# Step 2 · 3주차 심화 5 보충 : 1D-CNN의 hold-out 혼동행렬 (README 3절의 모델별 그림)
# ext5_cnn1d.py 는 정확도와 오류 수만 저장했으므로, 같은 함수(CNN1D, run) · 같은 seed 로 hold-out 학습만 다시 해서
# 예측을 얻는다. seed 와 cuDNN deterministic 설정이 같으므로 ext5_cnn1d.md 의 hold-out 정확도와 같아야 하고,
# 다르면 멈춘다 (같은 모델이라는 확인).
#
# 실행 방법: python ext5_confusion.py   (ext5_cnn1d.py 를 먼저 실행, GPU 약 20초)
# 결과: confusion_cnn1d.png, ext5_confusion.md

import os, re
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from common3 import BASE_DIR, SHORT, load_all_trials, load_segments, error_types, save_single_confusion
from ext5_cnn1d import run

S = load_segments()
T, y, g, s = load_all_trials()
assert len(S) == len(y) and (g == np.arange(len(g))).all()
d2 = np.load(os.path.join(BASE_DIR, '..', 'week2', 'step5_features.npz'))
tr_ids, te_ids = np.unique(d2['gtr']), np.unique(d2['gte'])
pred, _, _, _ = run(S[tr_ids], y[tr_ids], S[te_ids])
yt = y[te_ids]
acc = np.mean(pred == yt) * 100

# ext5_cnn1d.md 의 hold-out 정확도와 같은지 확인
md = open(os.path.join(BASE_DIR, 'ext5_cnn1d.md'), encoding='utf-8').read()
ref = float(re.search(r'\| \*\*1D-CNN \(원신호\)\*\* \| ([\d.]+) \|', md).group(1))
assert abs(acc - ref) < 0.005, f'hold-out 정확도가 ext5_cnn1d.md 와 다릅니다: {acc:.2f} vs {ref:.2f}'

cm = confusion_matrix(yt, pred, labels=range(8))
save_single_confusion(os.path.join(BASE_DIR, 'confusion_cnn1d.png'), cm,
                      f'1D-CNN (2초 원신호 5 x 2,000)\n정확도 {np.trace(cm)}/{cm.sum()} = {acc:.2f}%')

n, ga, an, both = error_types(yt, pred)
P, R, F, _ = precision_recall_fscore_support(yt, pred, labels=range(8), zero_division=0)
off = cm.copy(); np.fill_diagonal(off, 0)
idx = np.argsort(off.ravel())[::-1]
pairs = [f'{SHORT[i // 8]}→{SHORT[i % 8]} {off.ravel()[i]}' for i in idx if off.ravel()[i] >= 2]
with open(os.path.join(BASE_DIR, 'ext5_confusion.md'), 'w', encoding='utf-8') as f:
    f.write('# 심화 5 보충. 1D-CNN hold-out 혼동행렬 (테스트 384시행)\n\n')
    f.write(f'정확도 {np.trace(cm)}/{cm.sum()} = {acc:.2f}% (ext5_cnn1d.md 와 같음). '
            f'macro Precision {P.mean() * 100:.2f} / Recall {R.mean() * 100:.2f} / F1 {F.mean() * 100:.2f}\n\n')
    f.write(f'오류 {n}건: 같은 동작 · 다른 각도 {ga}건, 같은 각도 · 다른 동작 {an}건, 둘 다 다름 {both}건\n\n')
    f.write(f'가장 많이 틀린 쌍 (2건 이상): {", ".join(pairs)}\n\n')
    f.write('| 클래스 | Precision | Recall | F1 | 예측 수 |\n|---|---|---|---|---|\n')
    for i, name in enumerate(SHORT):
        f.write(f'| {name} | {P[i] * 100:.1f} | {R[i] * 100:.1f} | {F[i] * 100:.1f} | {int((pred == i).sum())} |\n')
print(open(os.path.join(BASE_DIR, 'ext5_confusion.md'), encoding='utf-8').read())
