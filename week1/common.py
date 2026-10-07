# Step 2 · 1주차 실습 공용 함수
# 단계 2~4 스크립트가 똑같은 "파일 찾기 / 라벨 뽑기 / CSV 읽기 / 시행 자르기"를 반복하지 않도록 한곳에 모아둔다.
#
# 데이터 형식 (데이터 저장소 README 기준):
#   파일명  [피험자]_[각도] [동작].csv   예) a_30 fist.csv   (동작 표기: fist / open / pro / extor)
#   컬럼    bandpass 1~5 (5채널 sEMG), time (초), onset, mark   ← 첫 줄은 헤더
#   한 파일 = 한 피험자의 한 조건을 30회 반복한 연속 기록 (시행 하나 = 5초 = 5,000샘플)

import glob, os
import numpy as np
import pandas as pd

# __file__ 기준 절대경로 — 어느 폴더에서 실행하든 같은 결과가 나오게 한다 (Step 1과 같은 방식).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 데이터는 주차 폴더가 아니라 semg-arm 바로 아래(../data)에 둔다. 2·3주차도 같이 쓰기 때문.
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

FS = 1000                                            # 샘플링 주파수 1 kHz (논문 IX-BIO8 설정)
TRIAL_LEN = 5 * FS                                   # 시행 하나 = 휴식 2초 + 동작 1초 + 휴식 2초
CH_NAMES = ['ECR', 'ED', 'FDS', 'Biceps', 'Triceps']  # 채널 1~5 (전완 3 + 상완 2)
ANGLES = ['30', '60']
GESTURES = ['fist', 'open', 'pronation', 'supination']
CLASSES = [f'{a}_{g}' for a in ANGLES for g in GESTURES]  # 30_fist ... 60_supination (8개)
# 파일명에 쓰인 동작 표기 -> 강의/논문의 클래스 이름
GESTURE_ALIAS = {'fist': 'fist', 'open': 'open', 'pro': 'pronation', 'extor': 'supination'}


def find_files():
    # 슬라이드 단계 2: data/ 아래의 csv를 폴더 깊이에 상관없이 전부 찾는다.
    files = sorted(glob.glob(os.path.join(DATA_DIR, '**', '*.csv'), recursive=True))
    if not files:
        raise SystemExit(f"data/**/*.csv 에서 파일을 찾지 못했습니다. 확인한 경로: {os.path.abspath(DATA_DIR)}")
    return files


def label_of(path):
    # 슬라이드 예시(30_fist_trial01.csv)와 실제 파일명 규칙이 달라서 데이터에 맞게 고친 버전
    # (슬라이드 15쪽 "클래스 개수가 안 맞음 -> label_of 함수를 데이터에 맞게 수정").
    # 'a_30 fist.csv' -> '30_fist',  'a_60 extor.csv' -> '60_supination'
    name = os.path.splitext(os.path.basename(path))[0].lower()
    head, gesture = name.rsplit(' ', 1)
    return f"{head.split('_')[-1]}_{GESTURE_ALIAS.get(gesture, gesture)}"


def subject_of(path):
    # 'a_30 fist.csv' -> 'a' (익명화된 피험자 ID)
    return os.path.basename(path).split('_')[0].lower()


def load_file(path):
    # 헤더가 있는 CSV라서 pandas로 읽는다. 앞의 5개 컬럼이 5채널 sEMG.
    df = pd.read_csv(path)
    x = df.iloc[:, :5].to_numpy(dtype=float)   # (전체 샘플 수, 5)
    return df, x


def trial_starts(df, n):
    # 연속 기록에서 각 시행이 시작하는 샘플 위치를 찾는다.
    # mark 컬럼이 1로 찍히는 지점을 시행 경계로 쓴다. 실제 데이터에서 확인한 결과(step3 출력):
    # 64개 파일 모두 mark가 30번, 4999~5001샘플(5초) 간격으로 찍혀 있고 첫 mark가 파일 첫 줄이다.
    # mark로 5초짜리 시행을 하나도 못 자르면 처음부터 5초씩 균등하게 자른다.
    if 'mark' in df.columns:
        m = pd.to_numeric(df['mark'], errors='coerce')
        if m.isna().all():                       # 숫자가 아니라 글자로 표시된 경우
            on = df['mark'].notna().to_numpy()
        else:
            on = m.fillna(0).to_numpy() != 0
        starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        starts = starts[starts + TRIAL_LEN <= n]
        if len(starts):
            return starts, 'mark'
    return np.arange(0, n - TRIAL_LEN + 1, TRIAL_LEN), '5초 균등 분할'


def load_trials(path):
    # 한 파일을 시행 단위로 잘라 (시행 수, 5000, 5) 배열로 돌려준다.
    df, x = load_file(path)
    starts, _ = trial_starts(df, len(x))
    return np.stack([x[s:s + TRIAL_LEN] for s in starts])


def active_starts(trials, win=FS):
    # 시행마다 "근활성이 가장 큰 1초"가 시작하는 샘플 위치를 찾는다. -> (시행 수,)
    # 필요한 이유: 프로토콜은 휴식 2초 - 동작 1초 - 휴식 2초지만, 실제 기록에서는 mark 기준으로
    # 동작이 2~3초에 고정돼 있지 않고 파일·시행마다 위치가 다르다 (step4의 motion_timing 결과).
    # 방법: 채널마다 크기가 다르므로 파일 전체 RMS로 나눠 맞춘 뒤, 5채널 에너지 합의 1초 이동합이 최대인 곳.
    scale = np.sqrt(np.mean(trials ** 2, axis=(0, 1), keepdims=True))
    e = np.sum((trials / scale) ** 2, axis=2)                    # (시행 수, 5000)
    c = np.cumsum(np.pad(e, ((0, 0), (1, 0))), axis=1)
    return np.argmax(c[:, win:] - c[:, :-win], axis=1)
