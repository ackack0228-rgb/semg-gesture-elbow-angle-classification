# Step 2 · 2주차 실습 · 단계 2 : 분석 구간(2초 세그먼트) 잘라내기
# 목적: 5초짜리 시행에서 동작이 들어 있는 2초(2,000샘플)를 잘라낸다.
#
# 논문은 "2초 세그먼트"라고만 하고 시작점을 밝히지 않는다. 슬라이드 예시의 1.5초 시작도 실습 가정이고
# "시작 지점은 데이터에 맞게 조정할 것"이라고 되어 있다. 1주차에 확인했듯이 이 데이터의 근활성은 mark 뒤
# 2~3초가 아니라 약 1~2.5초에 나타나므로, 후보 시작점마다 동작이 얼마나 들어오는지 세어 보고 정한다.
# 결정: START = 500 (mark + 0.5초 ~ 2.5초) — 근거는 실행 결과 segment_start.md.  ← 논문 설정이 아닌 재현 가정
#
# 실행 방법: python step2_segment.py   (step1_filter.py 를 먼저 실행)
# 결과: segment_start.md, segment_check.png, step2_seg.npy (다음 단계 입력)

import os, sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'week1'))
from common import FS, TRIAL_LEN, CH_NAMES, find_files, load_file, trial_starts, active_starts
from step1_filter import preprocess

# 시행 구조(프로토콜): 휴식 2초 | 동작 1초 | 휴식 2초
# 아래 0.5초 시작은 실습 가정이며 논문 설정이 아님
START, LEN = 500, 2000         # mark 뒤 0.5초 지점부터 2초
SLIDE_START = 1500             # 슬라이드 예시 값 (비교용)


def cut(x, start=START, n=LEN):
    # 시행 하나 (5000, 5) -> 분석 구간 (2000, 5)
    return x[start:start + n]


def load_filtered_trials(path):
    # 파일 하나: 연속 기록 전체에 필터를 건 뒤 mark로 시행을 자른다 -> (시행 수, 5000, 5)
    df, x = load_file(path)
    xf = preprocess(x)
    starts, _ = trial_starts(df, len(xf))
    return np.stack([xf[s:s + TRIAL_LEN] for s in starts])


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── 예시 시행 하나를 잘라 모양 확인 (슬라이드 확인 방법) ───────────────────
    ex = np.load(os.path.join(BASE_DIR, 'step1_example.npz'))
    trial = ex['xf'][ex['starts'][0]:ex['starts'][0] + TRIAL_LEN]   # 예시 파일의 첫 시행 (필터 후)
    seg = cut(trial)
    print('시행 모양:', trial.shape)
    print('세그먼트 모양:', seg.shape)   # (2000, 5)
    np.save(os.path.join(BASE_DIR, 'step2_seg.npy'), seg)

    # ── 시작점 후보 비교: 전체 1,920시행 ───────────────────────────────────────
    # act  = 시행마다 근활성이 가장 큰 1초의 시작 위치 (1주차 common.active_starts, 라벨을 쓰지 않는다)
    # cum  = 휴식 수준(시행 안 중앙값)을 넘는 에너지의 누적합 -> 어떤 구간이 동작 에너지의 몇 %를 담는지 계산용
    print('전체 파일 읽는 중 ...')
    act, cum, env = [], [], []
    for p in find_files():
        tr = load_filtered_trials(p)
        act.append(active_starts(tr))
        e = np.sum((tr / np.sqrt(np.mean(tr ** 2, axis=(0, 1), keepdims=True))) ** 2, axis=2)   # (30, 5000)
        over = np.clip(e - np.median(e, axis=1, keepdims=True), 0, None)
        cum.append(np.cumsum(np.pad(over, ((0, 0), (1, 0))), axis=1))
        a = np.abs(tr)
        env.append((a / a.mean(axis=(0, 1), keepdims=True)).mean(axis=2))
    act, cum, env = np.concatenate(act), np.concatenate(cum), np.concatenate(env)

    def score(s):
        inside = np.mean((act >= s) & (act + FS <= s + LEN)) * 100                 # 활성 1초가 통째로 들어온 시행
        overlap = np.mean(np.clip(np.minimum(act + FS, s + LEN) - np.maximum(act, s), 0, None) / FS) * 100
        energy = np.mean((cum[:, s + LEN] - cum[:, s]) / cum[:, -1]) * 100         # 담긴 동작 에너지
        return inside, overlap, energy

    cands = list(range(0, TRIAL_LEN - LEN + 1, 250))
    with open(os.path.join(BASE_DIR, 'segment_start.md'), 'w', encoding='utf-8') as f:
        f.write('# 2초 세그먼트 시작점 정하기 (필터 후 신호, 전체 1,920시행)\n\n')
        f.write('- 활성 1초 = 시행마다 5채널 에너지 합이 가장 큰 1초 (`common.active_starts`, 라벨을 쓰지 않음)\n')
        f.write('- 동작 에너지 = 시행 안 중앙값(휴식 수준)을 넘는 에너지\n\n')
        f.write('| 시작점 (mark 기준) | 구간 | 활성 1초가 통째로 들어온 시행 | 활성 1초와 겹치는 비율(평균) | 담긴 동작 에너지(평균) |\n')
        f.write('|---|---|---|---|---|\n')
        for s in cands:
            tag = ' ← 선택' if s == START else (' (슬라이드 예시)' if s == SLIDE_START else '')
            f.write(f'| {s / FS:.2f}초{tag} | {s / FS:.2f}~{(s + LEN) / FS:.2f}초 | '
                    + ' | '.join(f'{v:.1f}%' for v in score(s)) + ' |\n')
        q = np.percentile(act / FS, [5, 25, 50, 75, 95])
        f.write('\n- 활성 1초의 시작 위치(초) 5/25/50/75/95 백분위: ' + ' / '.join(f'{v:.2f}' for v in q) + '\n')
        f.write(f'- 선택한 구간({START / FS:.1f}~{(START + LEN) / FS:.1f}초) 밖에서 활성 1초가 시작하는 시행: '
                f'앞쪽 {int((act < START).sum())}개, 뒤쪽 {int((act + FS > START + LEN).sum())}개\n')

    best = max(cands, key=lambda s: score(s)[0])
    print(f'활성 1초가 통째로 들어오는 비율이 가장 높은 시작점: {best / FS:.2f}초')
    for s in (START, SLIDE_START):
        print(f'  시작 {s / FS:.2f}초 -> 포함 {score(s)[0]:.1f}% / 겹침 {score(s)[1]:.1f}% / 에너지 {score(s)[2]:.1f}%')

    # ── 그림: 자른 구간을 다시 그려 확인 ───────────────────────────────────────
    # 맨 위 = 전체 시행 평균 포락선, 아래 5줄 = 예시 시행의 5채널.  초록 띠 = 선택한 2초, 빨간 점선 = 슬라이드 예시(1.5~3.5초)
    t = np.arange(TRIAL_LEN) / FS
    grand = np.convolve(env.mean(axis=0), np.ones(100) / 100, mode='same')
    fig, ax = plt.subplots(6, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t, grand, 'k', lw=1)
    ax[0].set_ylabel('mean rectified\n(all trials)', fontsize=8)
    for ch in range(5):
        ax[ch + 1].plot(t, trial[:, ch], lw=0.5)
        ax[ch + 1].set_ylabel(CH_NAMES[ch], fontsize=8)
    for a in ax:
        a.axvspan(START / FS, (START + LEN) / FS, color='g', alpha=0.15, zorder=0)
        a.axvline(SLIDE_START / FS, color='r', ls='--', lw=0.8)
        a.axvline((SLIDE_START + LEN) / FS, color='r', ls='--', lw=0.8)
    ax[0].set_title(f'green: selected segment {START / FS:.1f}-{(START + LEN) / FS:.1f} s, '
                    f'red dashed: slide example {SLIDE_START / FS:.1f}-{(SLIDE_START + LEN) / FS:.1f} s   '
                    '(top: all 1,920 trials, below: a_30 fist trial 1)', fontsize=9)
    ax[5].set_xlabel('time from mark (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'segment_check.png'), dpi=120)
    plt.close(fig)
    print('저장 완료: segment_start.md, segment_check.png, step2_seg.npy')
