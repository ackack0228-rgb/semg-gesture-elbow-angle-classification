# sEMG 손동작 · 팔꿈치 각도 동시 분류 — 논문 재현

대구가톨릭대학교 「딥러닝프로그래밍」 Step 2 과제 (21113103 김근영)

- 재현 대상 논문: Lee, S., Kim, J., & Choi, S.-I. (2026). *Simultaneous classification of hand gestures and elbow angles using sEMG signals: a comparison of machine learning performances.* Biomedical Engineering Letters. DOI 10.1007/s13534-026-00607-7
- 데이터: [TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset](https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset)
- 문제: 전완 3채널 + 상완 2채널 sEMG(1 kHz)로 손동작 4종(fist · open · pronation · supination) × 팔꿈치 각도 2종(30° · 60°) = 8클래스를 한 번에 분류한다.

## 진행 상황

| 주차 | 내용 | 상태 |
|---|---|---|
| 1주차 | 문제 정의 · 데이터 구조 파악 · 5채널 신호 관찰 · 클래스 설계안 | 완료 (`week1/`) |
| 2주차 | 전처리 · 200 ms 윈도우 · 시간영역 특징 5종(윈도우당 25차원) | 예정 |
| 3주차 | SVM · RF · XGBoost 비교, 5-fold 교차검증, 혼동행렬 | 예정 |

## 실행 방법

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset.git data

cd week1
python step1_env_check.py     # 패키지 버전        -> env_versions.txt
python step2_data_check.py    # 파일 수 · 폴더 구조
python step3_explore.py       # 클래스별 시행 수   -> data_summary.md
python step4_plot.py          # 그림 · RMS 표      -> *.png, motion_timing.md, angle_rms_table.md
```

Windows에서는 `week1/run_all.bat` 으로 네 단계를 한 번에 실행할 수 있다(가상환경 경로는 파일 안에서 수정).

## 1주차 결과

### 데이터 구조 (직접 센 결과)

| 항목 | 결과 |
|---|---|
| 파일 수 | 64개 = 피험자 8명(a~h) × 8조건 |
| 파일명 | `[피험자]_[각도] [동작].csv` (pro = 회내, extor = 회외) |
| 컬럼 | `bandpass 1`~`bandpass 5` (5채널), `time`, `onset`, `mark` |
| 파일 1개 | (150,721 ~ 153,601, 5) — 한 조건의 30회 반복이 연속 기록 |
| 시행 경계 | `mark = 1` 이 파일마다 30번, 5,000샘플(5초) 간격 |
| 시행 1개 | (5000, 5) |
| 전체 시행 | 1,920 (클래스당 240, 피험자당 240 — 완전 균형) |

강의 예시(시행마다 파일 하나)와 달리 조건마다 파일 하나에 30회가 이어져 있어, `mark` 컬럼으로 5초씩 잘라 시행을 만든다(`week1/common.py`).

### 동작 구간의 위치

프로토콜은 휴식 2초 → 동작 1초 → 휴식 2초이지만, 실제 기록에서 근활성은 `mark` 뒤 2~3초가 아니라 약 1~2.5초에 나타난다. 전체 1,920시행 평균 신호의 정점은 1.65초이고, 시행별로 근활성이 가장 큰 1초는 중앙값 1.05초에서 시작한다(`week1/motion_timing.md`). 그래서 수치 비교에는 고정 구간 대신 시행마다 근활성이 가장 큰 1초를 동작 구간으로 썼다(`common.active_starts`).

![motion timing](week1/motion_timing.png)

### 5채널 파형

![5-channel example](week1/ch5_example.png)

### 같은 동작의 30° / 60° 비교

![angle envelope](week1/angle_envelope.png)

60° / 30° 동작 구간 RMS 비 — 같은 피험자 안에서 구한 비의 8명 중앙값 (괄호: 60°에서 커진 사람 수, * Wilcoxon p < 0.05, 보정 없음)

| 동작 | ECR | ED | FDS | Biceps | Triceps |
|---|---|---|---|---|---|
| fist | 0.98 (4/8) | 0.94 (3/8) | 0.96 (3/8) | 1.09 (7/8) | 1.16 (6/8) |
| open | 1.16 (8/8)* | 1.04 (5/8) | 0.96 (3/8) | 1.35 (7/8)* | 1.13 (6/8) |
| pronation | 1.19 (7/8)* | 1.09 (7/8)* | 1.12 (5/8) | 1.19 (6/8)* | 1.15 (7/8)* |
| supination | 1.29 (8/8)* | 1.07 (5/8) | 1.34 (8/8)* | 1.25 (7/8)* | 1.21 (7/8)* |

- 어느 채널이 반응하는지는 각도가 바뀌어도 같고, 각도는 크기를 조금 바꾼다.
- 60°에서는 상완 채널(Biceps · Triceps)이 네 동작 모두에서 커지는 경향이 있다.
- 각도에 따른 차이(0.94~1.35배)는 동작에 따른 차이(중앙값 1.5~3.7배)보다 훨씬 작다. 논문이 같은 동작의 각도 혼동을 주요 오류로 든 것과 같은 방향이다.
- 피험자 간 차이가 크다(Biceps 60°/30° 비 0.91~2.16).

동작별 나란히 비교 그림은 `week1/angle_compare_*.png`, 전체 표는 `week1/angle_rms_table.md`, 클래스 설계안은 `week1/class_design.md` 에 있다.

### 한계

- 30°와 60°는 따로 기록한 파일이라, 각도 외에 기록 사이의 차이(전극 상태 · 피로 · 힘)가 섞였을 수 있다.
- 배포된 신호는 이미 대역통과 필터를 거친 것이다(원신호 아님).
- `mark` 가 찍힌 기준은 데이터 설명에 없어, 동작 구간 위치는 신호에서 추정한 것이다.
