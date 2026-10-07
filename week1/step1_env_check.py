# Step 2 · 1주차 실습 · 단계 1 : 환경 점검
# 목적: Step 1에서 만든 가상환경을 그대로 쓰고, 이번 주에 추가한 xgboost까지 import 되는지 확인한다.
#
# 실행 방법 (Step 1의 가상환경 파이썬으로 실행):
#   "..\..\..\1.sEMG를 이용한 인증 프로젝트\semg-auth\.venv\Scripts\python" step1_env_check.py
# 결과: 화면 출력과 같은 내용이 env_versions.txt 로 저장된다 (제출물 ① 패키지 버전 출력).

import os, sys
import numpy, scipy, sklearn, xgboost

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

lines = [
    f'python   {sys.version.split()[0]}',
    f'numpy    {numpy.__version__}',
    f'scipy    {scipy.__version__}',
    f'sklearn  {sklearn.__version__}',
    f'xgboost  {xgboost.__version__}',
]
print('\n'.join(lines))  # 슬라이드 기준: 네 줄(numpy·scipy·sklearn·xgboost)이 모두 나오면 성공

with open(os.path.join(BASE_DIR, 'env_versions.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
