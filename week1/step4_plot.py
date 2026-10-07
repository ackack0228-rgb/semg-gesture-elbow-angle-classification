# Step 2 · 1주차 실습 · 단계 4 : 채널별 신호 그리기
# 목적: 휴식 - 동작 - 휴식 3구간이 보이는지, 채널마다 다르게 반응하는지,
#       같은 동작이 30°와 60°에서 어떻게 달라지는지를 눈으로 확인한다.
#
# 실행 방법: semg-arm\week1 폴더에서 Step 1 가상환경 파이썬으로
#   python step4_plot.py
# 결과: overview_example.png       (연속 기록 앞 30초 + 시행 시작 위치 — 시행 경계가 맞는지 확인용)
#       ch5_example.png            (제출물 ③ 시행 1개의 5채널 파형, 동작 구간 표시)
#       motion_timing.png / .md    (동작이 시행 안에서 실제로 언제 나타나는지 — 전체 1,920시행)
#       angle_compare_<동작>.png   (제출물 ④ 같은 동작의 30° / 60° 나란히 비교, 동작 4종)
#       angle_envelope.png         (④ 전체 피험자·전체 시행 평균 포락선으로 본 30° / 60°)
#       angle_rms_table.md         (④ 서술을 뒷받침할 클래스별·채널별 RMS와 60°/30° 비)
#       signal_checks.md           (이미 필터링된 신호인지, 휴식 구간의 주기적 파형 간격)

import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.stats import wilcoxon
from common import (BASE_DIR, FS, TRIAL_LEN, CH_NAMES, ANGLES, GESTURES, CLASSES,
                    find_files, label_of, subject_of, load_file, trial_starts, load_trials, active_starts)

EXAMPLE = ('a', '30_fist')   # 예시 그림에 쓸 (피험자, 클래스)
WIN = FS                     # 동작 구간 길이 1초 (프로토콜의 동작 수행 시간)

files = find_files()
key = {f: (subject_of(f), label_of(f)) for f in files}
subjects = sorted({s for s, _ in key.values()})
ex_file = next(f for f in files if key[f] == EXAMPLE)

# 전체 파일을 한 번만 읽어 둔다 (64개 x 30시행 x 5000샘플 x 5채널, float32로 약 190MB).
print('파일 읽는 중 ...')
trials = {f: load_trials(f).astype(np.float32) for f in files}
act = {f: active_starts(trials[f], WIN) for f in files}   # 시행별 "가장 활성이 큰 1초"의 시작 위치
t = np.arange(TRIAL_LEN) / FS


def mark_motion(a, start):
    # 빨간 점선 = 프로토콜상의 동작 구간(2~3초), 회색 띠 = 이 시행에서 실제로 활성이 가장 큰 1초
    a.axvline(2.0, color='r', ls='--', lw=0.8)
    a.axvline(3.0, color='r', ls='--', lw=0.8)
    a.axvspan(start / FS, (start + WIN) / FS, color='0.85', zorder=0)


# ── 그림 0: 연속 기록 앞 30초 ─────────────────────────────────────────────────
# "동작 구간이 안 보이면 전체를 그려보고 경계 위치를 눈으로 찾기" (슬라이드 15쪽).
# 초록 선 = mark로 찾은 시행 시작. 시행마다 파형이 한 번씩 커지면 자르는 방식이 맞는 것.
df, x = load_file(ex_file)
starts, how = trial_starts(df, len(x))
n = min(len(x), 30 * FS)
fig, ax = plt.subplots(5, 1, figsize=(12, 7), sharex=True)
for ch in range(5):
    ax[ch].plot(np.arange(n) / FS, x[:n, ch], lw=0.4)
    ax[ch].set_ylabel(CH_NAMES[ch], fontsize=8)
    for s in starts[starts < n]:
        ax[ch].axvline(s / FS, color='g', ls=':')
ax[4].set_xlabel('time (s)')
ax[0].set_title(f'{os.path.basename(ex_file)} — first 30 s, trial starts by: {how}', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'overview_example.png'), dpi=120)
plt.close(fig)

# ── 그림 1: 시행 1개의 5채널 파형 ─────────────────────────────────────────────
fig, ax = plt.subplots(5, 1, figsize=(10, 7), sharex=True)
for ch in range(5):
    ax[ch].plot(t, trials[ex_file][0][:, ch], lw=0.5)
    ax[ch].set_ylabel(CH_NAMES[ch], fontsize=8)
    mark_motion(ax[ch], act[ex_file][0])
ax[4].set_xlabel('time (s)')
ax[0].set_title(f'{os.path.basename(ex_file)} — trial 1 ({EXAMPLE[1]})   '
                'red dashed: nominal motion 2-3 s, gray: most active 1 s', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'ch5_example.png'), dpi=120)
plt.close(fig)

# ── 그림 2: 동작이 시행 안에서 실제로 언제 나타나는가 ─────────────────────────
# 채널별 크기를 맞춘 정류 신호(절댓값)를 전체 1,920시행에 대해 평균하면, 프로토콜대로라면 2~3초에
# 봉우리가 있어야 한다. 실제 봉우리 위치와 시행별 활성 구간 시작 위치 분포를 함께 그린다.
env_all, start_all = [], []
for f in files:
    a = np.abs(trials[f])
    env_all.append((a / a.mean(axis=(0, 1), keepdims=True)).mean(axis=2))   # (30, 5000)
    start_all.append(act[f])
env_all, start_all = np.concatenate(env_all), np.concatenate(start_all)
grand = np.convolve(env_all.mean(axis=0), np.ones(100) / 100, mode='same')  # 100ms 이동평균
fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
ax[0].plot(t, grand, 'k', lw=1)
ax[0].axvline(2.0, color='r', ls='--'); ax[0].axvline(3.0, color='r', ls='--')
ax[0].set_ylabel('mean rectified EMG\n(normalized, 5 ch)', fontsize=8)
ax[0].set_title(f'all {len(env_all)} trials aligned to mark — red dashed: nominal motion 2-3 s', fontsize=9)
ax[1].hist(start_all / FS, bins=np.arange(0, 4.01, 0.1), color='0.5')
ax[1].axvline(2.0, color='r', ls='--'); ax[1].axvline(3.0, color='r', ls='--')
ax[1].set_ylabel('trials', fontsize=8)
ax[1].set_xlabel('time from mark (s) — bottom: start of the most active 1 s in each trial')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'motion_timing.png'), dpi=120)
plt.close(fig)

q = np.percentile(start_all / FS, [5, 25, 50, 75, 95])
in_nominal = np.mean((start_all >= 1.5 * FS) & (start_all <= 2.5 * FS))   # 2~3초와 절반 이상 겹치는 시행
with open(os.path.join(BASE_DIR, 'motion_timing.md'), 'w', encoding='utf-8') as f:
    f.write('# 동작 구간의 실제 위치 (mark 기준, 전체 시행)\n\n')
    f.write(f'- 시행 수: {len(start_all)}\n')
    f.write(f'- 전체 평균 포락선의 정점: {np.argmax(grand) / FS:.2f}초\n')
    f.write('- 시행별 "가장 활성이 큰 1초"의 시작 위치(초) 5/25/50/75/95 백분위: '
            + ' / '.join(f'{v:.2f}' for v in q) + '\n')
    f.write(f'- 그 1초가 프로토콜상의 2~3초와 절반 이상 겹치는 시행 비율: {in_nominal * 100:.1f}%\n\n')
    f.write('| 피험자 | 활성 구간 시작 중앙값(초) | 최소 | 최대 |\n|---|---|---|---|\n')
    for s in subjects:
        v = np.concatenate([act[p] for p in files if key[p][0] == s]) / FS
        f.write(f'| {s} | {np.median(v):.2f} | {v.min():.2f} | {v.max():.2f} |\n')

# ── 그림 3: 같은 동작의 30° / 60° 를 나란히 (동작 4종 각각) ───────────────────
# 왼쪽 열 = 30°, 오른쪽 열 = 60°. 같은 채널(행)끼리 y축을 맞춰야 진폭 차이를 공정하게 볼 수 있다.
# 피험자가 달라지면 개인차가 섞이므로 반드시 같은 피험자의 첫 시행끼리 비교한다.
subj = EXAMPLE[0]
for g in GESTURES:
    fig, ax = plt.subplots(5, 2, figsize=(12, 7), sharex=True, sharey='row')
    for col, a in enumerate(ANGLES):
        p = next(f for f in files if key[f] == (subj, f'{a}_{g}'))
        for ch in range(5):
            ax[ch, col].plot(t, trials[p][0][:, ch], lw=0.5)
            mark_motion(ax[ch, col], act[p][0])
            ax[ch, 0].set_ylabel(CH_NAMES[ch], fontsize=8)
        ax[0, col].set_title(f'{a}_{g}  (subject {subj}, trial 1)', fontsize=9)
        ax[4, col].set_xlabel('time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f'angle_compare_{g}.png'), dpi=120)
    plt.close(fig)

# ── 그림 4: 전체 평균 포락선으로 본 30° / 60° ─────────────────────────────────
# 시행 하나만 보면 우연이 섞이므로, 활성 구간 시작에 맞춰 정렬한 정류 신호를 8명 x 30시행 평균한다.
# 행 = 채널, 열 = 동작. 실선 = 30°, 점선 = 60°.
PRE, POST = 500, 1500                                  # 활성 구간 시작 0.5초 전 ~ 1.5초 후
tt = np.arange(-PRE, POST) / FS
smooth = np.ones(50) / 50
fig, ax = plt.subplots(5, 4, figsize=(12, 8), sharex=True, sharey='row')
for col, g in enumerate(GESTURES):
    for a, style in zip(ANGLES, ['k-', 'r--']):
        seg = []
        for p in files:
            if key[p][1] != f'{a}_{g}':
                continue
            for tr, s in zip(trials[p], act[p]):
                if s - PRE >= 0 and s + POST <= TRIAL_LEN:
                    seg.append(np.abs(tr[s - PRE:s + POST]))
        m = np.mean(seg, axis=0)
        for ch in range(5):
            ax[ch, col].plot(tt, np.convolve(m[:, ch], smooth, mode='same'), style, lw=1, label=f'{a} deg')
    ax[0, col].set_title(g, fontsize=9)
    ax[4, col].set_xlabel('time from active-window start (s)', fontsize=8)
for ch in range(5):
    ax[ch, 0].set_ylabel(CH_NAMES[ch], fontsize=8)
ax[0, 0].legend(fontsize=7)
fig.suptitle('mean rectified EMG, all subjects x all trials (solid: 30 deg, dashed: 60 deg)', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'angle_envelope.png'), dpi=120)
plt.close(fig)

# ── 표: 클래스별·채널별 RMS, 60°/30° 비 ───────────────────────────────────────
# 동작 구간 = 시행마다 찾은 "가장 활성이 큰 1초". 휴식 구간 = 시행의 마지막 1초(4~5초).
# rms[피험자][클래스] = 그 피험자 30시행의 채널별 RMS 평균 (5,)
rms, rest = defaultdict(dict), defaultdict(dict)
for p in files:
    s, lab = key[p]
    seg = np.stack([tr[a0:a0 + WIN] for tr, a0 in zip(trials[p], act[p])])
    rms[s][lab] = np.sqrt(np.mean(seg ** 2, axis=1)).mean(axis=0)
    rest[s][lab] = np.sqrt(np.mean(trials[p][:, 4 * FS:] ** 2, axis=1)).mean(axis=0)

with open(os.path.join(BASE_DIR, 'angle_rms_table.md'), 'w', encoding='utf-8') as f:
    f.write('# 클래스별 · 채널별 RMS (전체 피험자 8명 · 클래스당 240시행)\n\n')
    f.write('동작 구간 = 시행마다 근활성이 가장 큰 1초, 휴식 구간 = 시행의 마지막 1초(4~5초). 단위는 CSV 값 그대로.\n\n')
    f.write('## 동작 구간 RMS 평균\n\n| 클래스 | ' + ' | '.join(CH_NAMES) + ' |\n|---|' + '---|' * 5 + '\n')
    for c in CLASSES:
        v = np.mean([rms[s][c] for s in subjects], axis=0)
        f.write(f'| {c} | ' + ' | '.join(f'{x:.4f}' for x in v) + ' |\n')
    v = np.mean([rest[s][c] for s in subjects for c in CLASSES], axis=0)
    f.write('| (휴식 구간, 전 클래스) | ' + ' | '.join(f'{x:.4f}' for x in v) + ' |\n')

    # 같은 피험자 안에서 60° RMS / 30° RMS 를 구한 뒤 8명의 중앙값을 낸다 (개인차를 지우기 위해).
    # 괄호 = 8명 중 60°에서 더 커진 사람 수. * = Wilcoxon 부호순위 검정 p < 0.05 (n = 8, 다중비교 보정 없음).
    f.write('\n## 60° / 30° RMS 비 — 피험자 8명의 중앙값 (60°에서 커진 사람 수 / 8)\n\n')
    f.write('| 동작 | ' + ' | '.join(CH_NAMES) + ' |\n|---|' + '---|' * 5 + '\n')
    for g in GESTURES:
        r30 = np.array([rms[s][f'30_{g}'] for s in subjects])
        r60 = np.array([rms[s][f'60_{g}'] for s in subjects])
        cells = []
        for ch in range(5):
            pval = wilcoxon(r60[:, ch], r30[:, ch]).pvalue
            cells.append(f'{np.median(r60[:, ch] / r30[:, ch]):.2f} ({int((r60[:, ch] > r30[:, ch]).sum())}/8)'
                         + ('*' if pval < 0.05 else ''))
        f.write(f'| {g} | ' + ' | '.join(cells) + ' |\n')

    f.write('\n## 피험자별 60° / 30° RMS 비 (Biceps 채널)\n\n| 동작 | ' + ' | '.join(subjects) + ' |\n|---|'
            + '---|' * len(subjects) + '\n')
    for g in GESTURES:
        f.write(f'| {g} | ' + ' | '.join(f"{rms[s][f'60_{g}'][3] / rms[s][f'30_{g}'][3]:.2f}" for s in subjects) + ' |\n')

    # 각도 차이가 동작 차이에 비해 얼마나 작은지 보려고, 같은 각도 안에서 동작끼리의 RMS 비도 낸다.
    # 값 = 피험자 8명의 중앙값 (최소~최대), 괄호 = 비가 1보다 큰 사람 수.
    f.write('\n## 같은 각도 안에서 동작 간 RMS 비 — 피험자 8명의 중앙값, 최소~최대 (1보다 큰 사람 수 / 8)\n\n')
    f.write('| 비교 | 30° | 60° |\n|---|---|---|\n')
    others = lambda s, a, ch: np.mean([rms[s][f'{a}_{g}'][ch] for g in GESTURES if g != 'supination'])
    contrasts = [
        ('FDS: open / fist', lambda s, a: rms[s][f'{a}_open'][2] / rms[s][f'{a}_fist'][2]),
        ('ECR: fist / open', lambda s, a: rms[s][f'{a}_fist'][0] / rms[s][f'{a}_open'][0]),
        ('ED: open / pronation', lambda s, a: rms[s][f'{a}_open'][1] / rms[s][f'{a}_pronation'][1]),
        ('Biceps: supination / 나머지 세 동작 평균', lambda s, a: rms[s][f'{a}_supination'][3] / others(s, a, 3)),
    ]
    for name, fn in contrasts:
        cells = []
        for a in ANGLES:
            v = np.array([fn(s, a) for s in subjects])
            cells.append(f'{np.median(v):.2f}, {v.min():.2f}~{v.max():.2f} ({int((v > 1).sum())}/8)')
        f.write(f'| {name} | ' + ' | '.join(cells) + ' |\n')

    # 피험자마다 신호 크기 자체가 얼마나 다른지 (8클래스 평균 RMS)
    f.write('\n## 피험자별 동작 구간 RMS (8클래스 평균)\n\n| 채널 | ' + ' | '.join(subjects) + ' | 최대/최소 |\n|---|'
            + '---|' * (len(subjects) + 1) + '\n')
    for ch in range(5):
        v = np.array([np.mean([rms[s][c][ch] for c in CLASSES]) for s in subjects])
        f.write(f'| {CH_NAMES[ch]} | ' + ' | '.join(f'{x:.3f}' for x in v) + f' | {v.max() / v.min():.1f}배 |\n')

# ── 확인 1: 배포된 신호가 이미 필터링된 것인가 ───────────────────────────────
# 컬럼 이름(bandpass)과 README만 믿지 않고 스펙트럼으로 확인한다. 논문 전처리는 60 Hz 노치 + 20~450 Hz 대역통과.
psd = []
for p in files:
    fq, pw = welch(trials[p].reshape(-1, 5).astype(float), fs=FS, nperseg=4096, axis=0)
    psd.append(pw)
psd = np.mean(psd, axis=(0, 2))                                   # 64파일 x 5채널 평균
frac = lambda lo, hi: psd[(fq >= lo) & (fq < hi)].sum() / psd.sum() * 100
around = (psd[(fq >= 52) & (fq < 57)].mean() + psd[(fq >= 63) & (fq < 68)].mean()) / 2
notch = psd[(fq >= 59) & (fq <= 61)].mean() / around

# ── 확인 2: 휴식 구간에 반복되는 짧은 파형의 간격 ─────────────────────────────
# 시행의 마지막 1.7초(3.3~5초, 동작이 끝난 뒤)에서 정류 포락선의 봉우리 사이 간격을 잰다.
def rest_intervals(subject, ch):
    iv = []
    for p in files:
        if key[p][0] != subject:
            continue
        for tr in trials[p]:
            env = np.convolve(np.abs(tr[3300:, ch]), np.ones(40) / 40, mode='same')
            base = np.median(env)
            pk, _ = find_peaks(env, height=3 * base, distance=250, prominence=2 * base)
            iv += list(np.diff(pk))
    return np.array(iv)

with open(os.path.join(BASE_DIR, 'signal_checks.md'), 'w', encoding='utf-8') as f:
    f.write('# 신호 점검 (전체 64개 파일)\n\n## 주파수 대역별 전력 비율 (Welch, 5채널 평균)\n\n')
    f.write('| 대역 | 전력 비율 |\n|---|---|\n')
    f.write(f'| 0~20 Hz | {frac(0, 20):.2f}% |\n| 20~450 Hz | {frac(20, 450):.2f}% |\n| 450~500 Hz | {frac(450, 501):.3f}% |\n')
    f.write(f'\n- 60 Hz 부근(59~61 Hz) 전력 / 주변(52~57, 63~68 Hz) 전력 = {notch:.2f}\n')
    f.write('\n## 휴식 구간(시행의 3.3~5초)에 반복되는 파형의 간격\n\n')
    f.write('| 피험자 | ECR: 간격 수 | ECR: 중앙값 (25~75%) ms | Triceps: 간격 수 | Triceps: 중앙값 (25~75%) ms |\n|---|---|---|---|---|\n')
    for s in subjects:
        cells = []
        for ch in (0, 4):
            iv = rest_intervals(s, ch)
            cells += [str(len(iv)), f'{np.median(iv):.0f} ({np.percentile(iv, 25):.0f}~{np.percentile(iv, 75):.0f})' if len(iv) >= 20 else '-']
        f.write(f'| {s} | ' + ' | '.join(cells) + ' |\n')

print('저장 완료: overview_example.png, ch5_example.png, motion_timing.png/.md, '
      'angle_compare_*.png, angle_envelope.png, angle_rms_table.md, signal_checks.md')
