# Step 2 · 2주차 실습 · 단계 6 : 표준화와 저장 (제출물 ⑤ 데이터셋 요약, dataset.npz)
# 목적: 학습셋 통계로만 표준화하고, 3주차에 바로 쓸 수 있게 dataset.npz 로 남긴다.
#
# 실행 방법: python step6_scale_save.py   (step5_dataset.py 를 먼저 실행)
# 결과: dataset.npz, dataset_summary.md
#
# dataset.npz 내용
#   Xtr, Xte   표준화한 특징 (윈도우 수, 25)          ytr, yte   클래스 번호 0~7 (classes 의 인덱스)
#   gtr, gte   윈도우가 나온 시행의 고유 번호          str, ste   피험자 (a~h)
#   classes, feature_names                            scaler_mean, scaler_scale  (학습셋에서 구한 평균 · 표준편차)
# 3주차 5-fold 교차검증에서는 폴드마다 표준화를 다시 맞춰야 하므로 표준화 전 특징(step5_features.npz)을 쓴다.

import os
import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

d = np.load(os.path.join(BASE_DIR, 'step5_features.npz'))
Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
classes, names = d['classes'], d['feature_names']

sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr)     # 학습셋에서 fit
Xte_s = sc.transform(Xte)         # 테스트엔 적용만 (여기에 fit_transform 을 쓰면 정보 누출)

print('값이 변하지 않는(상수) 특징 수:', int((Xtr.std(0) == 0).sum()))
print('학습셋 평균:', Xtr_s.mean(0).round(6))
print('학습셋 표준편차:', Xtr_s.std(0).round(6))
print('테스트셋 평균 범위:', Xte_s.mean(0).min().round(3), '~', Xte_s.mean(0).max().round(3),
      ' 표준편차 범위:', Xte_s.std(0).min().round(3), '~', Xte_s.std(0).max().round(3))

np.savez(os.path.join(BASE_DIR, 'dataset.npz'), Xtr=Xtr_s, ytr=ytr,
         Xte=Xte_s, yte=yte,
         gtr=d['gtr'], gte=d['gte'], str=d['str'], ste=d['ste'],
         classes=classes, feature_names=names, scaler_mean=sc.mean_, scaler_scale=sc.scale_)
print('저장 완료')

# ── 데이터셋 요약 ──────────────────────────────────────────────────────────────
subjects = sorted(set(d['str']))
n_tr, n_te = len(set(d['gtr'])), len(set(d['gte']))
with open(os.path.join(BASE_DIR, 'dataset_summary.md'), 'w', encoding='utf-8') as f:
    f.write('# 데이터셋 요약 (dataset.npz)\n\n')
    f.write('| 항목 | 학습 | 테스트 | 합계 |\n|---|---|---|---|\n')
    f.write(f'| 시행(2초 세그먼트) 수 | {n_tr} | {n_te} | {n_tr + n_te} |\n')
    f.write(f'| 샘플(윈도우) 수 | {len(Xtr)} | {len(Xte)} | {len(Xtr) + len(Xte)} |\n')
    f.write(f'| 배열 모양 | {Xtr_s.shape} | {Xte_s.shape} | |\n')
    f.write(f'| 비율 | {len(Xtr) / (len(Xtr) + len(Xte)) * 100:.1f}% | {len(Xte) / (len(Xtr) + len(Xte)) * 100:.1f}% | |\n')
    f.write(f'\n- 학습·테스트에 함께 들어간 시행: {len(set(d["gtr"]) & set(d["gte"]))}개\n')
    f.write(f'- 샘플 1개 = 200 ms 윈도우 1개의 25차원 특징 벡터, 시행 1개에서 19개\n')
    f.write(f'- 결측치(NaN) · 무한값: {int(np.isnan(Xtr_s).sum() + np.isnan(Xte_s).sum())} · '
            f'{int(np.isinf(Xtr_s).sum() + np.isinf(Xte_s).sum())}\n')

    f.write('\n## 클래스별 분포 (윈도우 수, 괄호 = 시행 수)\n\n| 번호 | 클래스 | 학습 | 테스트 |\n|---|---|---|---|\n')
    for i, c in enumerate(classes):
        a, b = int((ytr == i).sum()), int((yte == i).sum())
        f.write(f'| {i} | {c} | {a} ({a // 19}) | {b} ({b // 19}) |\n')

    f.write('\n## 피험자별 분포 (윈도우 수, 괄호 = 시행 수)\n\n| 피험자 | 학습 | 테스트 |\n|---|---|---|\n')
    for s in subjects:
        a, b = int((d['str'] == s).sum()), int((d['ste'] == s).sum())
        f.write(f'| {s} | {a} ({a // 19}) | {b} ({b // 19}) |\n')

    # 표준화 전 값의 크기가 특징마다 얼마나 다른지 (표준화가 필요한 이유) 와 표준화 후 확인
    f.write('\n## 표준화 전후 (특징 종류별, 5채널 범위)\n\n')
    f.write('| 특징 | 표준화 전 학습셋 평균 (채널 최소~최대) | 표준화 후 학습셋 평균 · 표준편차 | 표준화 후 테스트셋 평균 · 표준편차 (채널 범위) |\n|---|---|---|---|\n')
    for i, name in enumerate(['RMS', 'MAV', 'WL', 'ZC', 'SSC']):
        k = slice(i * 5, i * 5 + 5)
        m = Xtr[:, k].mean(0)
        f.write(f'| {name} | {m.min():.4f} ~ {m.max():.4f} | '
                f'{abs(Xtr_s[:, k].mean(0)).max():.1e} · {Xtr_s[:, k].std(0).min():.4f}~{Xtr_s[:, k].std(0).max():.4f} | '
                f'{Xte_s[:, k].mean(0).min():+.3f}~{Xte_s[:, k].mean(0).max():+.3f} · '
                f'{Xte_s[:, k].std(0).min():.3f}~{Xte_s[:, k].std(0).max():.3f} |\n')
print('요약 저장: dataset_summary.md')
