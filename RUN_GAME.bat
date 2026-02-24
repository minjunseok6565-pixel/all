@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM -------------------------------------------------------
REM NBA SIM GM - one-click launcher (Windows)
REM Usage:
REM   RUN_GAME.bat
REM   RUN_GAME.bat --import-excel path\\to\\roster.xlsx
REM -------------------------------------------------------

cd /d "%~dp0"
set "ROOT_DIR=%CD%\"

if not exist "saves" mkdir "saves"
set "LEAGUE_DB_PATH=%ROOT_DIR%saves\_active_runtime.sqlite3"

echo [INFO] ROOT_DIR=%ROOT_DIR%
echo [INFO] LEAGUE_DB_PATH=%LEAGUE_DB_PATH%

if /I "%~1"=="--import-excel" (
  if "%~2"=="" (
    echo [ERROR] --import-excel requires an Excel file path.
    echo [HINT] RUN_GAME.bat --import-excel data\roster.xlsx
    pause
    exit /b 1
  )

  set "EXCEL_PATH=%~2"
  if not exist "%EXCEL_PATH%" (
    echo [ERROR] Excel file not found: %EXCEL_PATH%
    pause
    exit /b 1
  )

  echo [INFO] Importing Excel roster into DB...
  py -3 league_repo.py import_roster --db "%LEAGUE_DB_PATH%" --excel "%EXCEL_PATH%" --mode replace
  if errorlevel 1 (
    echo [ERROR] import_roster failed.
    pause
    exit /b 1
  )
)

echo [INFO] Opening browser: http://127.0.0.1:8000/static/NBA.html
start "NBA SIM GM" "http://127.0.0.1:8000/static/NBA.html"

echo [INFO] Starting FastAPI server...
py -3 -m uvicorn server:app --host 0.0.0.0 --port 8000
if errorlevel 1 (
  echo [ERROR] Failed to run uvicorn.
  echo [HINT] py -3 -m pip install uvicorn fastapi
  pause
  exit /b 1
)

endlocal
