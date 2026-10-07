# Step 2 · 1주차 실습 · 단계 3 : 데이터 구조 파악
# 목적: 8클래스가 고르게 들어 있는지, 신호 길이와 채널 수가 일정한지 "숫자로" 먼저 센다.
#
# 실행 방법: semg-arm\week1 폴더에서 Step 1 가상환경 파이썬으로
#   python step3_explore.py
# 결과: 화면 출력 + data_summary.md (제출물 ② 클래스별 시행 수 · 배열 모양 · 채널 수 표)

import os
from collections import Counter, defaultdict
import numpy as np
from common import BASE_DIR, CLASSES, FS, find_files, label_of, subject_of, load_file, trial_starts

files = find_files()
labels = [label_of(f) for f in files]
print(Counter(labels))                # 클래스별 파일 수 — 피험자 8명이면 클래스마다 8개

# 파일 하나를 자세히 본다: 컬럼 구성, 길이, 값 범위, onset/mark 가 어떻게 찍혀 있는지
df, x = load_file(files[0])
print('컬럼:', list(df.columns))
print('배열 모양:', x.shape)           # (전체 샘플 수, 5) — 30회 x 5초 x 1000Hz = 150,000행 예상
print('채널 수:', x.shape[1])
print('기록 길이(초):', len(x) / FS)
print('값 범위:', x.min(), '~', x.max())
for col in ('onset', 'mark'):
    if col in df.columns:
        print(f'{col} 값 종류(상위 5개):', df[col].value_counts(dropna=False).head(5).to_dict())

# 전체 파일을 돌면서 클래스별·피험자별 시행 수와 파일 길이를 센다.
trials = defaultdict(int)      # 클래스 -> 시행 수
per_subject = defaultdict(int)  # 피험자 -> 시행 수
lengths, n_ch, methods, gaps, n_marks = [], Counter(), Counter(), [], Counter()
first_mark, tail, onset_same = [], [], 0
for f, lab in zip(files, labels):
    df, x = load_file(f)
    starts, how = trial_starts(df, len(x))
    trials[lab] += len(starts)
    per_subject[subject_of(f)] += len(starts)
    lengths.append(len(x))
    n_ch[x.shape[1]] += 1
    n_marks[int((df['mark'] != 0).sum())] += 1
    first_mark.append(int(starts[0]))
    tail.append(len(x) - int(starts[-1]) - 5000)                 # 30번째 시행 뒤에 남는 행 수
    onset_rows = np.flatnonzero(df['onset'].to_numpy() != 0)
    onset_same += int(len(onset_rows) == 1 and onset_rows[0] == starts[0])   # onset의 1이 한 번뿐이고 첫 mark와 같은 행
    methods[how] += 1
    gaps += list(np.diff(starts))
    if len(starts) != 30:
        print(f'  시행 수가 30이 아님: {os.path.basename(f)} -> {len(starts)}회')

print('클래스별 시행 수:', dict(trials))          # 논문 기준 클래스당 8명 x 30회 = 240
print('피험자별 시행 수:', dict(per_subject))      # 논문 기준 피험자당 8클래스 x 30회 = 240
# 파일마다 길이가 조금씩 다르다(30번째 시행 뒤에 남는 꼬리 길이 차이). 시행 30개 = 150,000행은 모두 들어 있다.
print('파일 길이(행) 최소/중앙/최대:', min(lengths), int(np.median(lengths)), max(lengths))
print('채널 수 분포:', dict(n_ch), '/ 파일당 mark 개수 분포:', dict(n_marks))
print('첫 mark 위치(행) 최소/최대:', min(first_mark), max(first_mark), '/ 30번째 시행 뒤 남는 행 최소/최대:', min(tail), max(tail))
print('onset의 1이 한 번뿐이고 첫 mark와 같은 행인 파일 수:', onset_same, '/', len(files))
print('시행 경계를 찾은 방법:', dict(methods))
if gaps:
    print('시행 시작 간격(샘플) 최소/중앙/최대:', min(gaps), int(np.median(gaps)), max(gaps))  # 5000이면 정상

# 예상한 8클래스에 없는 라벨이 나오면 파일명 규칙이 다른 것이므로 label_of 를 고쳐야 한다.
unknown = sorted(set(trials) - set(CLASSES))
if unknown:
    print('예상 밖 라벨:', unknown)
if max(trials.values()) != min(trials.values()):
    print('클래스 불균형 있음 — 리포트에 기록할 것')

with open(os.path.join(BASE_DIR, 'data_summary.md'), 'w', encoding='utf-8') as f:
    f.write('# 1주차 실습 · 단계 3 — 데이터 요약\n\n')
    f.write(f'- 전체 파일 수: {len(files)} (피험자 {len(per_subject)}명)\n')
    f.write(f'- 파일 1개 배열 모양: ({min(lengths):,} ~ {max(lengths):,}, 5) — 한 조건 30회 연속 기록, 중앙값 {int(np.median(lengths)):,}행\n')
    f.write(f'- 컬럼: {list(df.columns)}\n')
    f.write(f'- 파일당 mark 개수: {dict(n_marks)}, 시행 시작 간격(샘플) 최소/중앙/최대: {min(gaps)} / {int(np.median(gaps))} / {max(gaps)}\n')
    f.write(f'- 첫 mark 위치(행): {min(first_mark)} ~ {max(first_mark):,}, 30번째 시행 뒤 남는 행: {min(tail)} ~ {max(tail):,}\n')
    f.write(f'- onset 컬럼: 1이 한 번뿐이고 첫 mark와 같은 행인 파일 {onset_same} / {len(files)}\n')
    f.write(f'- 피험자별 시행 수: {dict(per_subject)}\n')
    f.write(f'- 전체 시행 수: {sum(trials.values())}\n\n')
    f.write('| 클래스 | 파일 수 | 시행 수 | 시행 1개 배열 모양 | 채널 수 |\n|---|---|---|---|---|\n')
    n_files = Counter(labels)
    for c in sorted(trials):
        f.write(f'| {c} | {n_files[c]} | {trials[c]} | (5000, 5) | 5 |\n')
print('저장 완료: data_summary.md')
