# Step 2 · 3주차 심화 5 : 1D-CNN 비교 (본 강의자료 54쪽, 난이도 ★★★ — 논문의 향후 과제 "딥러닝 비교")
# 목적: 손으로 만든 특징 없이 2초 구간 원신호(필터 후, 5채널 x 2,000샘플)를 바로 넣는 1D-CNN을 학습해
#       정확도와 학습 시간을 세 모델과 함께 비교한다.
#
# 조건은 본문과 같게: 같은 hold-out 분할 · 같은 5-fold 폴드 · 시행 단위 채점 · 정규화(채널별 평균 · 표준편차)는 학습 시행으로만.
# 학습 설정 (실습 선택 — 검증셋으로 고르지 않은 고정값): Adam lr 1e-3, weight decay 1e-4, 배치 32, 60 epoch,
#   마지막 epoch 모델로 평가 (테스트셋을 보고 멈출 시점을 고르지 않는다). seed 42.
# 측정 환경이 다르다: CNN은 GPU(RTX 3060 Laptop), 세 모델은 CPU — 학습 시간은 직접 비교할 수 없다.
#
# 실행 방법: python ext5_cnn1d.py   (약 10분)
# 결과: ext5_cnn1d.md

import os, time, random
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_rel
from common3 import BASE_DIR, COMPS, COMP_NAME, MODELS, load_all_trials, load_segments, cv_splits, error_types

SEED, EPOCHS, BATCH, LR, WD = 42, 60, 32, 1e-3, 1e-4
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_all(sd=SEED):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class CNN1D(nn.Module):
    def __init__(self, n_ch=5, n_cls=8):
        super().__init__()

        def block(i, o, k, pool):
            layers = [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()]
            return layers + ([nn.MaxPool1d(pool)] if pool else [])

        self.net = nn.Sequential(*block(n_ch, 32, 7, 4), *block(32, 64, 7, 4), *block(64, 128, 5, 4),
                                 *block(128, 128, 3, 0), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                 nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, x):            # x: (배치, 5, 2000)
        return self.net(x)


def run(Sa, ya, Sb):
    # 학습 시행으로만 채널별 평균 · 표준편차를 구해 정규화한다
    mu, sd = Sa.mean(axis=(0, 1)), Sa.std(axis=(0, 1))
    to_t = lambda S: torch.tensor(((S - mu) / sd).transpose(0, 2, 1), dtype=torch.float32)
    Xa, Xb, Ya = to_t(Sa).to(dev), to_t(Sb).to(dev), torch.tensor(ya).to(dev)
    seed_all()
    net = CNN1D().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator(device='cpu').manual_seed(SEED)
    if dev == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        net.train()
        for idx in torch.randperm(len(Xa), generator=g).split(BATCH):
            opt.zero_grad(); loss = lossf(net(Xa[idx]), Ya[idx]); loss.backward(); opt.step()
    if dev == 'cuda': torch.cuda.synchronize()
    train_t = time.perf_counter() - t0
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xb[i:i + 256]).argmax(1) for i in range(0, len(Xb), 256)]).cpu().numpy()
    return pred, train_t, net, Xb


# 다른 스크립트(ext5_confusion.py)가 CNN1D · run 만 불러 쓸 수 있도록 실행 부분을 main 아래에 둔다
if __name__ == '__main__':
    S = load_segments()                          # (1920, 2000, 5), 시행 번호 순
    T, y, g, s = load_all_trials()
    assert len(S) == len(y) and (g == np.arange(len(g))).all()     # 시행 번호 = S 의 행 번호

    # ── hold-out (2주차 분할) ──
    d2 = np.load(os.path.join(BASE_DIR, '..', 'week2', 'step5_features.npz'))
    tr_ids, te_ids = np.unique(d2['gtr']), np.unique(d2['gte'])
    pred_h, t_h, net, Xb = run(S[tr_ids], y[tr_ids], S[te_ids])
    acc_h = np.mean(pred_h == y[te_ids])
    n_err, ga, an, both = error_types(y[te_ids], pred_h)
    print(f'hold-out: {acc_h * 100:.2f}%  학습 {t_h:.1f}초  오류 {n_err} (같은 동작 다른 각도 {ga})', flush=True)

    # 추론 시간: 표본 1개씩 100번 (GPU, CPU)
    n_par = sum(p.numel() for p in net.parameters())
    inf = {}
    with torch.no_grad():
        for where in (['cuda', 'cpu'] if dev == 'cuda' else ['cpu']):
            m = net.to(where); x1 = Xb[:1].to(where)
            for _ in range(10): m(x1)                       # 예열
            if where == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100): m(x1).argmax(1)
            if where == 'cuda': torch.cuda.synchronize()
            inf[where] = (time.perf_counter() - t0) * 10

    # ── 5-fold (본문과 같은 폴드) ──
    folds = cv_splits(y, s)
    acc_cv, t_cv = [], []
    for fi, (tr, va) in enumerate(folds):
        p, t, _, _ = run(S[tr], y[tr], S[va])
        acc_cv.append(np.mean(p == y[va])); t_cv.append(t)
        print(f'fold {fi + 1}: {acc_cv[-1] * 100:.2f}%  학습 {t:.1f}초', flush=True)
    acc_cv = np.array(acc_cv)

    cv = np.load(os.path.join(BASE_DIR, 'cv_results.npz'))
    ho = np.load(os.path.join(BASE_DIR, 'holdout_pred.npz'))
    with open(os.path.join(BASE_DIR, 'ext5_cnn1d.md'), 'w', encoding='utf-8') as f:
        f.write('# 심화 5. 1D-CNN (2초 원신호 5 x 2,000 입력) vs 세 모델\n\n')
        f.write(f'구조: Conv1d(5→32, k7)–BN–ReLU–MaxPool4 → Conv1d(32→64, k7)–BN–ReLU–MaxPool4 → Conv1d(64→128, k5)–BN–ReLU–MaxPool4 '
                f'→ Conv1d(128→128, k3)–BN–ReLU → 전역 평균 풀링 → Dropout 0.3 → Linear(128→8). 파라미터 {n_par:,}개.\n\n')
        f.write(f'학습: Adam lr {LR}, weight decay {WD}, 배치 {BATCH}, {EPOCHS} epoch 고정 (마지막 epoch 모델로 평가), seed {SEED}, '
                f'{torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"} · PyTorch {torch.__version__}\n\n')
        f.write('| 모델 | hold-out 정확도 | 5-fold 평균 ± 표준편차 | fold 1~5 | vs 1D-CNN p (5-fold) |\n|---|---|---|---|---|\n')
        f.write(f'| **1D-CNN (원신호)** | {acc_h * 100:.2f} | **{acc_cv.mean() * 100:.2f} ± {acc_cv.std() * 100:.2f}** | '
                + ' · '.join(f'{v * 100:.2f}' for v in acc_cv) + ' | — |\n')
        for c in COMPS:
            for k in MODELS:
                v = cv[f'acc_{c}_{k}']
                f.write(f'| {k} · {COMP_NAME[c]} | {np.mean(ho[f"{c}_{k}"] == ho["yte"]) * 100:.2f} | {v.mean() * 100:.2f} ± {v.std() * 100:.2f} | '
                        + ' · '.join(f'{x * 100:.2f}' for x in v) + f' | {ttest_rel(acc_cv, v).pvalue:.4f} |\n')
        f.write(f'\n## 1D-CNN 계산 비용\n\n| 항목 | 값 |\n|---|---|\n')
        f.write(f'| 학습 시간 (hold-out 1,536시행, {EPOCHS} epoch, GPU) | {t_h:.1f} 초 |\n')
        f.write(f'| 학습 시간 (5-fold 평균) | {np.mean(t_cv):.1f} 초 |\n')
        for where, v in inf.items():
            f.write(f'| 추론 (표본 1개 = 시행 1개, {where.upper()}, 100회 평균) | {v:.3f} ms |\n')
        f.write(f'| 파라미터 수 | {n_par:,} |\n| 모델 크기 (float32) | {n_par * 4 / 1e6:.2f} MB |\n')
        f.write(f'\nhold-out 오류 {n_err}건 중 같은 동작 · 다른 각도 {ga}건, 같은 각도 · 다른 동작 {an}건, 둘 다 다름 {both}건.\n')
    print(open(os.path.join(BASE_DIR, 'ext5_cnn1d.md'), encoding='utf-8').read())
