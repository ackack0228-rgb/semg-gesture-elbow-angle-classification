# Step 2 · 2주차 실습 · 단계 4 : 시간영역 특징 구현 (제출물 ③ td_features 함수)
# 목적: 윈도우 하나 (200, 5) 에서 25차원 특징 벡터를 만든다. 라이브러리 없이 numpy만 쓴다.
#
# 특징 벡터의 순서 (해석할 때 필요하므로 고정):
#   [RMS x 5채널, MAV x 5채널, WL x 5채널, ZC x 5채널, SSC x 5채널],  채널 순서 = ECR, ED, FDS, Biceps, Triceps
#   -> 0~4 RMS, 5~9 MAV, 10~14 WL, 15~19 ZC, 20~24 SSC
#
# 실행 방법: python step4_features.py   (step3_window.py 를 먼저 실행)
# 결과: feature_order.md, step4_f.npy

import os, sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'week1'))
from common import CH_NAMES

FEATURES = ['RMS', 'MAV', 'WL', 'ZC', 'SSC']
FEATURE_NAMES = [f'{f}_{c}' for f in FEATURES for c in CH_NAMES]   # 25개, td_features 출력 순서와 같다


def td_features(w, th=0.01):
    """w: (200, 5) -> (25,)   논문 식 (1)~(5)"""
    rms = np.sqrt(np.mean(w**2, axis=0))                 # (1) 제곱평균의 제곱근 — 신호 파워
    mav = np.mean(np.abs(w), axis=0)                     # (2) 절댓값의 평균
    d   = np.diff(w, axis=0)                             #     인접 샘플 차이 x[i+1] - x[i]
    wl  = np.sum(np.abs(d), axis=0)                      # (3) 인접 샘플 차이의 누적합 — 파형 길이
    zc  = np.sum(((w[:-1]*w[1:]) < 0) &                  # (4) 부호가 바뀌고 차이가 th 이상일 때만
                 (np.abs(d) >= th), axis=0)
    ssc = np.sum((d[:-1]*d[1:]) < 0, axis=0)             # (5) 기울기 부호가 바뀐 횟수 (임계값 없음)
    return np.concatenate([rms, mav, wl, zc, ssc])


if __name__ == '__main__':
    w = np.load(os.path.join(BASE_DIR, 'step3_w.npy'))   # (19, 200, 5)
    f = td_features(w[0])
    print('특징벡터:', f.shape)        # (25,)
    for i, name in enumerate(FEATURES):
        print(f'  {name:3s}', f[i * 5:(i + 1) * 5].round(4))
    print('NaN 개수:', int(np.isnan(f).sum()))

    # SSC: 본 강의자료 28쪽은 논문 식 (5) 그대로 (x[i]-x[i-1])(x[i]-x[i+1]) > 0 으로 쓰고,
    # 실습 슬라이드는 d[i-1]*d[i] < 0 으로 쓴다. 같은 식인지 19개 윈도우 전부에서 확인한다.
    ssc_paper = lambda v: np.sum((v[1:-1] - v[:-2]) * (v[1:-1] - v[2:]) > 0, axis=0)
    print('SSC 두 표기가 같은 값:', all(np.array_equal(td_features(v)[20:], ssc_paper(v)) for v in w))

    # ZC 임계값: th = 0.01(논문) 과 th = 0 을 비교한다 (슬라이드 "th를 0으로 바꿔보고 관찰").
    F = np.stack([td_features(v) for v in w])            # (19, 25)
    F0 = np.stack([td_features(v, th=0) for v in w])
    print('ZC (19개 윈도우 평균)  th=0.01:', F[:, 15:20].mean(0).round(1), ' th=0:', F0[:, 15:20].mean(0).round(1))
    # 임계값이 신호 크기에 비해 어느 정도인지: 인접 샘플 차이의 절댓값 중앙값
    step = np.median(np.abs(np.diff(w, axis=1)), axis=(0, 1))
    print('인접 샘플 차이 |d| 의 중앙값:', step.round(4), ' (th = 0.01)')

    with open(os.path.join(BASE_DIR, 'feature_order.md'), 'w', encoding='utf-8') as fo:
        fo.write('# 25차원 특징 벡터의 순서\n\n')
        fo.write('`td_features` 는 특징 5종을 차례로 이어 붙이고, 각 특징 안에서는 채널 1~5 순서다.\n\n')
        fo.write('| 인덱스 | 특징 | ' + ' | '.join(CH_NAMES) + ' |\n|---|---|' + '---|' * 5 + '\n')
        for i, name in enumerate(FEATURES):
            fo.write(f'| {i * 5}~{i * 5 + 4} | {name} | ' + ' | '.join(str(i * 5 + c) for c in range(5)) + ' |\n')
        fo.write('\n## 예시 세그먼트(a_30 fist, 시행 1)의 19개 윈도우 평균\n\n')
        fo.write('| 특징 | ' + ' | '.join(CH_NAMES) + ' |\n|---|' + '---|' * 5 + '\n')
        for i, name in enumerate(FEATURES):
            fo.write(f'| {name} | ' + ' | '.join(f'{v:.4f}' for v in F[:, i * 5:(i + 1) * 5].mean(0)) + ' |\n')
        fo.write('| ZC (th = 0 일 때) | ' + ' | '.join(f'{v:.4f}' for v in F0[:, 15:20].mean(0)) + ' |\n')
        fo.write('| 인접 샘플 차이의 절댓값 중앙값 | ' + ' | '.join(f'{v:.4f}' for v in step) + ' |\n')

    np.save(os.path.join(BASE_DIR, 'step4_f.npy'), F)
    print('저장 완료: feature_order.md, step4_f.npy')
