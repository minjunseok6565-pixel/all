@echo off
setlocal ENABLEDELAYEDEXPANSION

REM -------------------------------------------------------
REM NBA 시뮬 GM - one-click launcher (Windows)
REM Usage:
REM   RUN_GAME.bat
REM   RUN_GAME.bat --import-excel path\to\roster.xlsx
REM -------------------------------------------------------

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if not exist "saves" mkdir "saves"
set "LEAGUE_DB_PATH=%ROOT_DIR%saves\_active_runtime.sqlite3"

echo [INFO] ROOT_DIR=%ROOT_DIR%
echo [INFO] LEAGUE_DB_PATH=%LEAGUE_DB_PATH%

if "%~1"=="--import-excel" (
  if "%~2"=="" (
    echo [ERROR] --import-excel 옵션에는 엑셀 파일 경로가 필요합니다.
    echo [HINT] RUN_GAME.bat --import-excel data\roster.xlsx
    exit /b 1
  )

  set "EXCEL_PATH=%~2"
  if not exist "%EXCEL_PATH%" (
    echo [ERROR] 엑셀 파일을 찾을 수 없습니다: %EXCEL_PATH%
    exit /b 1
  )

  echo [INFO] 엑셀 로스터를 DB로 import 합니다...
  python league_repo.py import_roster --db "%LEAGUE_DB_PATH%" --excel "%EXCEL_PATH%" --mode replace
  if errorlevel 1 (
    echo [ERROR] import_roster 실행 실패
    exit /b 1
  )
)

echo [INFO] 브라우저를 엽니다: http://127.0.0.1:8000/static/NBA.html
start "NBA SIM GM" http://127.0.0.1:8000/static/NBA.html

echo [INFO] FastAPI 서버를 시작합니다...
python -m uvicorn server:app --host 0.0.0.0 --port 8000
if errorlevel 1 (
  echo [ERROR] python -m uvicorn 실행 실패. uvicorn 설치 여부를 확인하세요.
  echo [HINT] pip install uvicorn fastapi
  exit /b 1
)

endlocal
