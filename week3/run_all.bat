@echo off
rem 3주차 실습 단계 1~7 + 심화 1~5 를 한 번에 실행한다.
rem 전체 약 1시간 10분 (단계 4 교차검증 약 12분, 심화 3 약 30분). 2주차 run_all.bat 을 먼저 실행해 둘 것.
rem 결과: holdout_metrics.md, confusion8.png, confusion_{svm,rf,xgb,cnn1d}.png, cv_results.md, ttest.md, cost.md, rest_check.md, ext1~5_*.md
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
rem 파이썬 고르기: 저장소의 .venv(README 1.5절) > 작성자 PC의 Step 1 가상환경 > PATH의 python
set PY=python
if exist "%~dp0..\..\..\1.sEMG를 이용한 인증 프로젝트\semg-auth\.venv\Scripts\python.exe" set PY="%~dp0..\..\..\1.sEMG를 이용한 인증 프로젝트\semg-auth\.venv\Scripts\python.exe"
if exist "%~dp0..\.venv\Scripts\python.exe" set PY="%~dp0..\.venv\Scripts\python.exe"
cd /d "%~dp0"
%PY% step1_train.py || goto :fail
%PY% step2_holdout.py || goto :fail
%PY% step3_confusion.py || goto :fail
%PY% step4_cv.py || goto :fail
%PY% step5_ttest.py || goto :fail
%PY% step6_cost.py || goto :fail
%PY% step7_rest_check.py || goto :fail
%PY% ext1_loso.py || goto :fail
%PY% ext2_channels.py || goto :fail
%PY% ext3_freq_window.py || goto :fail
%PY% ext4_hierarchical.py || goto :fail
%PY% ext5_cnn1d.py || goto :fail
%PY% ext5_confusion.py || goto :fail
echo.
echo 전체 단계 완료
pause
exit /b 0
:fail
echo.
echo 중간에 실패했습니다. 위 메시지를 확인하세요.
pause
exit /b 1
