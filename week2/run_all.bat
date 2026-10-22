@echo off
rem 2주차 단계 1~6을 한 번에 실행한다.
rem 결과: filter_check.png/.md, segment_start.md, segment_check.png, feature_order.md,
rem       step5_features.npz, dataset.npz, dataset_summary.md  — week2_submission.docx 는 건드리지 않는다.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
rem 파이썬 고르기: 저장소의 .venv(README 1.5절) > 작성자 PC의 Step 1 가상환경 > PATH의 python
set PY=python
if exist "%~dp0..\..\..\1.sEMG를 이용한 인증 프로젝트\semg-auth\.venv\Scripts\python.exe" set PY="%~dp0..\..\..\1.sEMG를 이용한 인증 프로젝트\semg-auth\.venv\Scripts\python.exe"
if exist "%~dp0..\.venv\Scripts\python.exe" set PY="%~dp0..\.venv\Scripts\python.exe"
cd /d "%~dp0"
%PY% step1_filter.py || goto :fail
%PY% step2_segment.py || goto :fail
%PY% step3_window.py || goto :fail
%PY% step4_features.py || goto :fail
%PY% step5_dataset.py || goto :fail
%PY% step6_scale_save.py || goto :fail
echo.
echo 전체 단계 완료
pause
exit /b 0
:fail
echo.
echo 중간에 실패했습니다. 위 메시지를 확인하세요.
pause
exit /b 1
