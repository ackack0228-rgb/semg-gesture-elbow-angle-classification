# Step 2 · 3주차 실습 공용 함수
# 단계 1~6과 심화 실험이 같은 모델 설정 · 같은 데이터 · 같은 표본 구성을 쓰도록 한곳에 모아둔다.
#
# ── 입력 표본 구성 (재현 가정 — 논문에 명시 없음, 본 강의자료 24 · 53쪽) ─────────────────────────
# 논문은 "시행 1,920개, 윈도우당 25차원, 학습 1,536 / 테스트 384"만 밝히고, 한 시행의 윈도우 19개를
# 모델 표본으로 어떻게 묶었는지는 밝히지 않았다. 그래서 세 가지를 같은 조건에서 나란히 비교한다.
#   A  윈도우 학습 + 다수결 : 윈도우 1개(25차원)를 표본 1개로 학습 (슬라이드 코드 그대로, 학습 29,184개).
#                            시행의 윈도우 19개 예측을 다수결로 묶어 시행 하나를 채점한다.
#   B  시행 평균           : 시행 하나의 윈도우 19개 특징을 평균해 25차원 표본 1개 (학습 1,536개).
#   C  시행 이어붙임       : 시행 하나의 윈도우 19개 특징을 시간 순서대로 이어 475차원 표본 1개 (학습 1,536개).
# 세 구성 모두 평가 단위는 시행(테스트 384개 = 논문과 같은 수)이다. A는 윈도우 단위 정확도도 함께 기록한다.
# 표준화(StandardScaler)는 세 구성 모두 학습 표본으로만 fit 한다.

import os, sys, pickle, time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEEK2_DIR = os.path.join(BASE_DIR, '..', 'week2')
MODEL_DIR = os.path.join(BASE_DIR, 'models')      # 단계 1에서 학습한 모델 (gitignore)
CACHE_DIR = os.path.join(BASE_DIR, 'cache')       # 심화 실험용 2초 구간 캐시 (gitignore)

CLASSES = ['30_fist', '30_open', '30_pronation', '30_supination',
           '60_fist', '60_open', '60_pronation', '60_supination']
# 논문 Fig. 5 표기 (30F = 30° 주먹 ...)
SHORT = ['30F', '30O', '30P', '30S', '60F', '60O', '60P', '60S']
MODELS = ['SVM', 'RF', 'XGB']
COMPS = ['A', 'B', 'C']
COMP_NAME = {'A': 'A 윈도우 학습+다수결', 'B': 'B 시행 평균(25차원)', 'C': 'C 시행 이어붙임(475차원)'}
NW = 19                                           # 시행 하나의 윈도우 수 (200 ms, 50% 오버랩)


def make_models():
    # 논문 설정 그대로 (본 강의자료 32~34쪽, 3주차 실습 9쪽).
    # XGBoost의 objective · reg_lambda · tree_method 는 슬라이드 코드에는 없지만 논문 설정이라 적어 둔다
    # (xgboost 3.x 기본값과 같은 값이라 결과는 바뀌지 않는다).
    return {
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr'),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGB': XGBClassifier(objective='multi:softprob', n_estimators=400, max_depth=6,
                             learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                             reg_lambda=1.0, tree_method='hist', random_state=42),
    }


# ── 데이터 ───────────────────────────────────────────────────────────────────────────────────
def to_trials(X, y, g, s, nw=NW):
    # 윈도우 행 (시행 수 x nw, f) -> 시행 배열 (시행 수, nw, f).  2주차 build()는 시행마다 윈도우를 연달아 쌓았다.
    n = len(X) // nw
    G = g.reshape(n, nw)
    assert len(X) == n * nw and (G == G[:, :1]).all(), '윈도우가 시행별로 연속해 있지 않습니다'
    return X.reshape(n, nw, -1), y[::nw], g[::nw], s[::nw]


def load_holdout():
    # 2주차 분할 그대로 (표준화 전 특징 step5_features.npz). 순서도 2주차 dataset.npz 와 같다.
    d = np.load(os.path.join(WEEK2_DIR, 'step5_features.npz'))
    tr = to_trials(d['Xtr'], d['ytr'], d['gtr'], d['str'])
    te = to_trials(d['Xte'], d['yte'], d['gte'], d['ste'])
    return tr, te       # 각각 (T, y, g, s)


def load_all_trials():
    # 1,920시행 전체를 시행 번호 순으로 (교차검증 · LOSO 용)
    (Ta, ya, ga, sa), (Tb, yb, gb, sb) = load_holdout()
    T, y, g, s = (np.concatenate(v) for v in [(Ta, Tb), (ya, yb), (ga, gb), (sa, sb)])
    o = np.argsort(g)
    return T[o], y[o], g[o], s[o]


def cv_splits(y, s, k=5):
    # 5-fold 층화 교차검증 — 시행 단위로 나눈다 (슬라이드 12쪽 "시행(파일) 단위로 나눈다").
    # 층 = 피험자 x 클래스 (2주차 hold-out 분할과 같은 기준) -> 폴드마다 피험자 x 클래스가 6시행씩 들어간다.
    strata = np.array([f'{a}_{b}' for a, b in zip(s, y)])
    skf = StratifiedKFold(k, shuffle=True, random_state=42)
    return list(skf.split(np.zeros(len(y)), strata))


# ── 표본 구성 · 학습 · 예측 ────────────────────────────────────────────────────────────────────
def represent(T, comp):
    n, nw, f = T.shape
    if comp == 'A':
        return T.reshape(n * nw, f)
    if comp == 'B':
        return T.mean(axis=1)
    if comp == 'C':
        return T.reshape(n, nw * f)
    raise ValueError(comp)


def vote(pw, nw, n_classes=8):
    # 윈도우 예측 (시행 수 x nw,) -> 시행 예측. 동점이면 번호가 작은 클래스 (np.argmax 규칙).
    P = pw.reshape(-1, nw)
    C = np.stack([np.bincount(r, minlength=n_classes) for r in P])
    ties = int(((C == C.max(1, keepdims=True)).sum(1) > 1).sum())
    return C.argmax(1), ties


def fit(comp, model, Ta, ya):
    # 학습 표본으로만 scaler 를 fit 하고 모델을 학습한다. 반환: (scaler, model, 학습 시간 초)
    Xa = represent(Ta, comp)
    yy = np.repeat(ya, Ta.shape[1]) if comp == 'A' else ya
    sc = StandardScaler().fit(Xa)
    t0 = time.perf_counter()
    model.fit(sc.transform(Xa), yy)
    return sc, model, time.perf_counter() - t0


def predict(comp, sc, model, Tb):
    # 반환: (시행 예측, 윈도우 예측 또는 None, 다수결 동점 수)
    pw = model.predict(sc.transform(represent(Tb, comp)))
    if comp == 'A':
        pt, ties = vote(pw, Tb.shape[1])
        return pt, pw, ties
    return pw, None, 0


def fit_predict(comp, model, Ta, ya, Tb):
    sc, model, t = fit(comp, model, Ta, ya)
    pt, pw, ties = predict(comp, sc, model, Tb)
    return pt, pw, ties, t


# ── 채점 보조 ─────────────────────────────────────────────────────────────────────────────────
def gesture_of(c):
    return np.asarray(c) % 4          # 0 fist, 1 open, 2 pronation, 3 supination


def angle_of(c):
    return np.asarray(c) // 4         # 0 = 30°, 1 = 60°


def draw_confusion(ax, cm, title, vmax=48):
    # 8x8 혼동행렬 한 장 (step3_confusion.py, ext5_confusion.py 공용). 행 = 실제, 열 = 예측.
    # 빨간 점선 = 같은 동작 · 다른 각도 칸 (30X <-> 60X) — 논문이 주요 오류로 설명한 자리
    import matplotlib.pyplot as plt
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=vmax)
    ax.set_xticks(range(8), SHORT, rotation=45)
    ax.set_yticks(range(8), SHORT)
    for i in range(8):
        for j in range(8):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                        color='white' if cm[i, j] > 30 else 'black')
    for i in range(8):
        j = (i + 4) % 8
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='tab:red', lw=1, ls='--'))
    ax.set_xlabel('예측'); ax.set_ylabel('실제')
    ax.set_title(title, fontsize=10)
    return im


def save_single_confusion(path, cm, title):
    # 모델 하나의 혼동행렬을 그림 파일 하나로 저장 (README의 모델별 그림)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = draw_confusion(ax, cm, title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def error_types(y, p):
    # 오류를 세 종류로 센다: 같은 동작 · 다른 각도 / 같은 각도 · 다른 동작 / 둘 다 다름
    y, p = np.asarray(y), np.asarray(p)
    e = y != p
    ga = e & (gesture_of(y) == gesture_of(p))
    an = e & (angle_of(y) == angle_of(p))
    return int(e.sum()), int(ga.sum()), int(an.sum()), int((e & ~ga & ~an).sum())


# ── 저장 · 불러오기 ───────────────────────────────────────────────────────────────────────────
def save_model(name, obj):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f'{name}.pkl'), 'wb') as f:
        pickle.dump(obj, f)


def load_model(name):
    path = os.path.join(MODEL_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        raise SystemExit(f'{path} 가 없습니다. step1_train.py 를 먼저 실행하세요.')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_segments():
    # 심화 실험(윈도우 길이 · 주파수 특징 · 1D-CNN)용: 1,920시행의 필터 후 2초 구간 (시행 번호 순).
    # 2주차 step5_dataset.py 와 같은 함수 · 같은 순서로 만들고, 처음 한 번만 CSV를 읽어 cache/ 에 저장한다.
    path = os.path.join(CACHE_DIR, 'segments.npy')
    if os.path.exists(path):
        return np.load(path)
    sys.path.insert(0, WEEK2_DIR)
    sys.path.insert(0, os.path.join(BASE_DIR, '..', 'week1'))
    from common import find_files
    from step5_dataset import file_segments
    print('CSV 64개를 읽어 2초 구간을 만드는 중 (처음 한 번만) ...')
    S = np.concatenate([file_segments(p) for p in find_files()])     # (1920, 2000, 5)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, S)
    return S


def load_rest_segments():
    # 한계 점검(step7)용: 1,920시행의 mark 뒤 4~5초 (휴식 구간, 필터 후) — 시행 번호 순, cache/rest.npy
    # 시행 구조 = 휴식 2초 | 동작 1초 | 휴식 2초 이고, 1주차 확인에서 동작은 mark 뒤 약 0.4~2.6초 안에 있었다.
    path = os.path.join(CACHE_DIR, 'rest.npy')
    if os.path.exists(path):
        return np.load(path)
    sys.path.insert(0, WEEK2_DIR)
    sys.path.insert(0, os.path.join(BASE_DIR, '..', 'week1'))
    from common import find_files
    from step2_segment import load_filtered_trials
    print('CSV 64개를 읽어 휴식 구간을 만드는 중 (처음 한 번만) ...')
    R = np.concatenate([load_filtered_trials(p)[:, 4000:5000] for p in find_files()])   # (1920, 1000, 5)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, R)
    return R


def freq_features(w, fs=1000):
    # 주파수 특징 (심화 실험): 윈도우 하나 (L, 5) -> MNF 5채널 + MDF 5채널 (10,)
    #   MNF (평균 주파수)   = sum(f * P(f)) / sum(P(f))
    #   MDF (중앙 주파수)   = 누적 파워가 전체의 절반이 되는 주파수
    # P(f) 는 윈도우의 파워 스펙트럼 |FFT|^2 (창 함수 없이, 0 Hz 제외)
    P = np.abs(np.fft.rfft(w, axis=0)) ** 2
    f = np.fft.rfftfreq(len(w), 1 / fs)
    P, f = P[1:], f[1:]
    mnf = (f[:, None] * P).sum(0) / P.sum(0)
    cum = np.cumsum(P, axis=0)
    mdf = f[np.argmax(cum >= cum[-1] / 2, axis=0)]
    return np.concatenate([mnf, mdf])


def segment_features(S, win=200, hop=100, extra=False):
    # 2초 구간 (시행 수, 2000, 5) -> 시행 배열 (시행 수, 윈도우 수, 25 또는 35). 2주차 함수를 그대로 쓴다.
    sys.path.insert(0, WEEK2_DIR)
    sys.path.insert(0, os.path.join(BASE_DIR, '..', 'week1'))
    from step3_window import make_windows
    from step4_features import td_features
    out = []
    for seg in S:
        W = make_windows(seg, win, hop)
        F = [td_features(w) for w in W]
        if extra:
            F = [np.concatenate([a, freq_features(w)]) for a, w in zip(F, W)]
        out.append(np.stack(F))
    return np.stack(out)


def env_info():
    import platform, sklearn, xgboost, scipy
    try:
        import subprocess
        cpu = subprocess.run(['powershell', '-NoProfile', '-c', '(Get-CimInstance Win32_Processor).Name'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        cpu = platform.processor()
    return (f'{cpu}, 논리 코어 {os.cpu_count()}개, {platform.system()} {platform.release()}, '
            f'Python {platform.python_version()}, numpy {np.__version__}, scikit-learn {sklearn.__version__}, '
            f'xgboost {xgboost.__version__}, scipy {scipy.__version__}')
