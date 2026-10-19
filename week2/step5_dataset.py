# Step 2 · 2주차 실습 · 단계 5 : 데이터셋 만들기 — 누수 막기 (제출물 ④ 분할 코드)
# 목적: 시행 단위로 먼저 학습/테스트를 나눈 뒤, 각 집합 안에서만 윈도우와 특징을 만든다.
#
# 슬라이드 예시는 "파일 1개 = 시행 1개"라서 파일 목록을 나누지만, 이 데이터는 파일 1개에 30시행이 이어져 있다.
# 그래서 나누는 단위를 (파일, 시행 번호)로 바꿨다 — 전체 1,920시행 -> 학습 1,536 / 테스트 384 (논문과 같은 8:2).
# 한 시행에서 나온 19개 윈도우는 서로 50%씩 겹치므로, 윈도우를 먼저 만들고 나누면 거의 같은 조각이
# 학습·테스트 양쪽에 들어간다. 시행끼리는 5초 간격으로 떨어져 있어 겹치지 않는다.
#
# 실행 방법: python step5_dataset.py
# 결과: step5_features.npz (표준화 전 특징 — 다음 단계 입력, 3주차 교차검증에서도 사용)

import os, sys
from functools import lru_cache
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'week1'))
from common import CLASSES, find_files, label_of, subject_of
from step2_segment import cut, load_filtered_trials
from step3_window import make_windows
from step4_features import td_features, FEATURE_NAMES


@lru_cache(maxsize=None)
def file_segments(path):
    # 파일 하나 -> 필터 -> 시행으로 자르기 -> 시행마다 2초 구간.  (시행 수, 2000, 5)
    # 같은 파일을 학습용·테스트용으로 두 번 읽지 않도록 결과를 기억해 둔다.
    return np.stack([cut(tr) for tr in load_filtered_trials(path)])


def build(tlist, trial_id):
    # tlist = [(파일 경로, 시행 번호), ...] 에 들어 있는 시행에서만 윈도우와 특징을 만든다.
    # X: 특징 (윈도우 수, 25),  y: 클래스 번호 0~7
    # g: 그 윈도우가 나온 시행의 고유 번호,  s: 피험자
    X, y, g, s = [], [], [], []
    for p, k in tlist:
        seg = file_segments(p)[k]
        for wnd in make_windows(seg):
            X.append(td_features(wnd)); y.append(CLASSES.index(label_of(p)))
            g.append(trial_id[(p, k)]); s.append(subject_of(p))
    return np.stack(X), np.array(y), np.array(g), np.array(s)


if __name__ == '__main__':
    files = find_files()
    print('파일 읽는 중 ...')
    trials = [(p, k) for p in files for k in range(len(file_segments(p)))]   # 전체 시행 목록
    trial_id = {t: i for i, t in enumerate(trials)}
    labels = [label_of(p) for p, _ in trials]
    # 층화 기준: 클래스 + 피험자 (= 파일). 클래스만 맞추면 피험자별 시행 수가
    # 양쪽에서 들쭉날쭉해질 수 있어, 파일마다 30시행이 24 / 6 으로 나뉘게 했다.
    # 클래스 비율도 자동으로 똑같아진다.
    strata = [f'{subject_of(p)}_{lab}' for (p, _), lab in zip(trials, labels)]
    print('전체 시행:', len(trials), ' 클래스:', len(set(labels)), ' 층:', len(set(strata)))

    # ★★★ 1) 시행 단위로 먼저 분할  ← 순서가 핵심 (윈도우는 아직 만들지 않았다) ★★★
    tr_t, te_t = train_test_split(
        trials, test_size=0.2, stratify=strata,
        random_state=42)

    # ★★★ 2) 각 집합 안에서만 윈도우 · 특징 생성 ★★★
    Xtr, ytr, gtr, str_ = build(tr_t, trial_id)
    Xte, yte, gte, ste = build(te_t, trial_id)
    print(Xtr.shape, Xte.shape)

    # ── 누수 점검 ──────────────────────────────────────────────────────────────
    print('학습 시행:', len(tr_t), ' 테스트 시행:', len(te_t))
    print('학습·테스트에 함께 들어간 시행 수:', len(set(gtr) & set(gte)))            # 0 이어야 한다
    print('윈도우 수 = 시행 수 x 19:', len(Xtr) == len(tr_t) * 19, len(Xte) == len(te_t) * 19)
    print('학습/테스트 샘플 수 비:', round(len(Xtr) / len(Xte), 2))                  # 약 4
    print('NaN:', int(np.isnan(Xtr).sum() + np.isnan(Xte).sum()),
          ' 무한값:', int(np.isinf(Xtr).sum() + np.isinf(Xte).sum()))

    np.savez(os.path.join(BASE_DIR, 'step5_features.npz'),
             Xtr=Xtr, ytr=ytr, gtr=gtr, str=str_, Xte=Xte, yte=yte, gte=gte, ste=ste,
             classes=np.array(CLASSES), feature_names=np.array(FEATURE_NAMES),
             trial_files=np.array([os.path.basename(p) for p, _ in trials]),
             trial_index=np.array([k for _, k in trials]))
    print('저장 완료: step5_features.npz')
