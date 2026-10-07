@echo off
rem 데이터가 ..\data 에 들어온 뒤 단계 1~4를 한 번에 실행한다 (Step 1 가상환경 사용).
rem 결과: env_versions.txt, data_summary.md, overview_example.png, ch5_example.png,
rem       angle_compare_*.png, angle_rms_table.md  — week1_submission.docx 는 건드리지 않는다.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY="%~dp0..\..\..\1.sEMG를 이용한 인증 프로젝트\semg-auth\.venv\Scripts\python.exe"
cd /d "%~dp0"
%PY% step1_env_check.py || goto :fail
%PY% step2_data_check.py || goto :fail
%PY% step3_explore.py || goto :fail
%PY% step4_plot.py || goto :fail
echo.
echo 전체 단계 완료
pause
exit /b 0
:fail
echo.
echo 중간에 실패했습니다. 위 메시지를 확인하세요.
pause
exit /b 1
