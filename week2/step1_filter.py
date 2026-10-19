# Step 2 · 2주차 실습 · 단계 1 : 필터 적용 (+ 제출물 ① 필터 전후 스펙트럼 비교)
# 목적: 신호에서 60 Hz 전원선 잡음과 20~450 Hz 밖의 성분을 제거한다 (논문 전처리 조건).
#
# 이 파일은 두 가지로 쓰인다 (Step 1의 week2/step1_filter.py 와 같은 방식, 대역만 20~450 Hz로 바꿈).
#   1) 단독 실행: 예시 파일의 필터 전후 표준편차를 출력하고, 전체 64개 파일의 스펙트럼을 비교한다.
#   2) 다른 단계에서 `from step1_filter import preprocess` 로 함수만 가져다 쓴다.
#
# 실행 방법: semg-arm\week2 폴더에서 Step 1 가상환경 파이썬으로
#   python step1_filter.py
# 결과: filter_check.png (제출물 ①), filter_check.md, step1_example.npz (다음 단계 입력)

import os, sys
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'week1'))   # 1주차 공용 함수(파일 찾기 · 시행 자르기) 재사용
from common import FS, CH_NAMES, find_files, label_of, subject_of, load_file, trial_starts

EXAMPLE = ('a', '30_fist')   # 예시로 쓸 (피험자, 클래스) — 1주차 그림과 같은 파일


def preprocess(x, fs=FS):
    # (시간, 채널) 신호에 60 Hz 노치 -> 20~450 Hz 대역통과를 차례로 건다.
    # 논문 조건은 "60 Hz 노치 + 20~450 Hz 4차 Butterworth"까지이고,
    # 노치 Q=30 과 filtfilt(영위상)는 실습 설정이다.
    # axis=0 을 빼면 시간축이 아니라 채널축(5개)에 필터가 걸린다.
    # 시행으로 자르기 전의 연속 기록 전체에 건다 (자른 뒤에 걸면 조각마다 양 끝이 왜곡된다).
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20 / (fs / 2), 450 / (fs / 2)], btype='band')
    return filtfilt(b, a, x, axis=0)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # 창을 띄우지 않고 파일로만 저장
    import matplotlib.pyplot as plt
    from scipy.signal import welch, freqz

    files = find_files()
    ex_file = next(f for f in files if (subject_of(f), label_of(f)) == EXAMPLE)

    # ── 예시 파일 하나: 필터 전후 표준편차 (슬라이드 확인 방법) ────────────────
    df, x = load_file(ex_file)
    xf = preprocess(x)
    print('파일:', os.path.basename(ex_file), ' 모양:', x.shape)
    print('전:', x.std(0).round(4))
    print('후:', xf.std(0).round(4))
    starts, _ = trial_starts(df, len(x))
    np.savez(os.path.join(BASE_DIR, 'step1_example.npz'), x=x, xf=xf, starts=starts)

    # ── 전체 64개 파일: 스펙트럼과 표준편차 비 ─────────────────────────────────
    # 배포된 CSV는 이미 대역통과를 거친 신호(1주차 signal_checks.md)라서 전후 차이가 작을 수 있다.
    # 그래서 예시 하나만 보지 않고 전체 파일에서 대역별 전력이 얼마나 바뀌는지 숫자로 확인한다.
    print('전체 파일 스펙트럼 계산 중 ...')
    P0, P1, ratio = [], [], []
    for p in files:
        _, v = load_file(p)
        vf = preprocess(v)
        fq, a0 = welch(v, fs=FS, nperseg=4096, axis=0)
        _, a1 = welch(vf, fs=FS, nperseg=4096, axis=0)
        P0.append(a0); P1.append(a1)
        ratio.append(vf.std(0) / v.std(0))
    P0, P1, ratio = np.array(P0), np.array(P1), np.array(ratio)   # (64, 주파수, 5), (64, 5)
    m0, m1 = P0.mean(axis=(0, 2)), P1.mean(axis=(0, 2))           # 64파일 x 5채널 평균 스펙트럼
    band = lambda lo, hi: (fq >= lo) & (fq < hi)

    # ── 그림: (위) 예시 파일 채널별 전후, (가운데) 전체 평균 전후, (아래) 설계한 필터의 응답 ──
    ei = files.index(ex_file)
    fig, ax = plt.subplots(3, 1, figsize=(10, 9))
    for ch in range(5):
        ax[0].semilogy(fq, P0[ei][:, ch], color=f'C{ch}', lw=0.6, alpha=0.5)
        ax[0].semilogy(fq, P1[ei][:, ch], color=f'C{ch}', lw=0.9, ls='--', label=CH_NAMES[ch])
    ax[0].set_title(f'{os.path.basename(ex_file)} — PSD per channel (thin solid: before, dashed: after)', fontsize=9)
    ax[0].legend(fontsize=7, ncol=5)
    ax[1].semilogy(fq, m0, 'k', lw=0.9, label='before')
    ax[1].semilogy(fq, m1, 'r', lw=0.9, label='after')
    ax[1].set_title(f'mean PSD of all {len(files)} files x 5 channels', fontsize=9)
    ax[1].legend(fontsize=8)
    # filtfilt는 같은 필터를 앞뒤로 두 번 걸기 때문에 크기 응답이 |H|^2 이 된다.
    bn, an = iirnotch(60, 30, FS)
    b, a = butter(4, [20 / (FS / 2), 450 / (FS / 2)], btype='band')
    w, hn = freqz(bn, an, worN=8192, fs=FS)
    _, hb = freqz(b, a, worN=8192, fs=FS)
    gain = 20 * np.log10(np.maximum(np.abs(hn * hb) ** 2, 1e-12))
    ax[2].plot(w, gain, 'b', lw=0.9)
    ax[2].set_ylim(-80, 5)
    ax[2].set_ylabel('gain (dB)')
    ax[2].set_title('designed filter response with filtfilt: 60 Hz notch (Q=30) + 20-450 Hz Butterworth', fontsize=9)
    ax[2].set_xlabel('frequency (Hz)')
    for k in range(3):
        for v in (20, 60, 450):
            ax[k].axvline(v, color='0.5', ls=':', lw=0.8)
        ax[k].set_xlim(0, 500)
    ax[0].set_ylabel('power'); ax[1].set_ylabel('power')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'filter_check.png'), dpi=120)
    plt.close(fig)

    # ── 표: 대역별 전력 변화 ───────────────────────────────────────────────────
    rows = [('0~20 Hz', band(0, 20)), ('20~450 Hz', band(20, 450)), ('450~500 Hz', band(450, 501)),
            ('59~61 Hz (노치 자리)', band(59, 61.01))]
    with open(os.path.join(BASE_DIR, 'filter_check.md'), 'w', encoding='utf-8') as f:
        f.write('# 필터 전후 비교 (전체 64개 파일)\n\n')
        f.write('필터: 60 Hz 노치(Q=30) → 20~450 Hz Butterworth(`butter(4, ..., btype="band")`), 둘 다 filtfilt.\n\n')
        f.write('## 예시 파일의 채널별 표준편차\n\n')
        f.write(f'`{os.path.basename(ex_file)}` | ' + ' | '.join(CH_NAMES) + ' |\n|---|' + '---|' * 5 + '\n')
        f.write('| 전 | ' + ' | '.join(f'{v:.4f}' for v in x.std(0)) + ' |\n')
        f.write('| 후 | ' + ' | '.join(f'{v:.4f}' for v in xf.std(0)) + ' |\n')
        f.write('\n## 대역별 전력 (Welch, 64파일 × 5채널 평균 스펙트럼)\n\n')
        f.write('| 대역 | 필터 전 전력에서 차지하는 비율 | 필터 후 남은 비율 (후/전) |\n|---|---|---|\n')
        for name, bmask in rows:
            f.write(f'| {name} | {m0[bmask].sum() / m0.sum() * 100:.3f}% | {m1[bmask].sum() / m0[bmask].sum() * 100:.1f}% |\n')
        f.write(f'| 전체 | 100% | {m1.sum() / m0.sum() * 100:.1f}% |\n')
        f.write('\n## 표준편차 비 (후/전) — 파일 64개의 분포\n\n')
        f.write('| 채널 | 최소 | 중앙값 | 최대 |\n|---|---|---|---|\n')
        for ch in range(5):
            f.write(f'| {CH_NAMES[ch]} | {ratio[:, ch].min():.3f} | {np.median(ratio[:, ch]):.3f} | {ratio[:, ch].max():.3f} |\n')
        # 표준편차가 가장 많이 줄어든 (파일, 채널) 5개와, 거기서 줄어든 전력이 어느 대역에 있었는지
        f.write('\n## 표준편차가 가장 많이 줄어든 5곳 — 줄어든 전력이 있던 대역\n\n')
        f.write('| 파일 | 채널 | 후/전 | 0~20 Hz | 59~61 Hz | 450~500 Hz | 그 밖(20~450 Hz) |\n|---|---|---|---|---|---|---|\n')
        for i in np.argsort(ratio, axis=None)[:5]:
            fi, ch = divmod(int(i), 5)
            lost = np.clip(P0[fi][:, ch] - P1[fi][:, ch], 0, None)
            part = [lost[band(0, 20)].sum(), lost[band(59, 61.01)].sum(), lost[band(450, 501)].sum()]
            part.append(lost.sum() - sum(part))
            f.write(f'| {os.path.basename(files[fi])} | {CH_NAMES[ch]} | {ratio[fi, ch]:.3f} | '
                    + ' | '.join(f'{v / lost.sum() * 100:.0f}%' for v in part) + ' |\n')

    print('표준편차 비(후/전) 전체 범위:', ratio.min().round(3), '~', ratio.max().round(3))
    print('저장 완료: filter_check.png, filter_check.md, step1_example.npz')
