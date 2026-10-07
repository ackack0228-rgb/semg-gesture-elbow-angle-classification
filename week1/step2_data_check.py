# Step 2 · 1주차 실습 · 단계 2 : 데이터 확보 확인
# 목적: semg-arm/data/ 에 압축을 푼 다채널 sEMG 파일이 제대로 들어왔는지 개수와 폴더 구조로 확인한다.
#
# 실행 방법: semg-arm\week1 폴더에서 Step 1 가상환경 파이썬으로
#   python step2_data_check.py

import os
from collections import Counter
from common import DATA_DIR, find_files

files = find_files()
print('파일 수:', len(files))   # 논문 기준이면 8명 x 8클래스 x 30회 = 1,920개
print('예시:', os.path.relpath(files[0], DATA_DIR))

# 폴더 구조를 눈으로 확인: 어떤 하위 폴더에 몇 개씩 들어 있는지
folders = Counter(os.path.relpath(os.path.dirname(f), DATA_DIR) for f in files)
print('폴더별 파일 수:')
for folder, n in sorted(folders.items()):
    print(f'  {folder:30s} {n}')

# 파일 이름에 클래스 정보(각도_동작)가 들어 있는지 앞쪽 몇 개만 본다
print('파일명 예시:')
for f in files[:5]:
    print('  ', os.path.basename(f))
