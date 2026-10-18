@echo off
cd /d "%~dp0"
:menu
cls
echo FIFA World Cup 2026 Manager
echo ============================
echo 1. Update data
echo 2. Start server
echo 3. Update and start
echo 0. Exit
echo ============================
set /p choice=Select option (0-3): 
if "%choice%"=="1" goto update
if "%choice%"=="2" goto serve
if "%choice%"=="3" goto update_and_serve
if "%choice%"=="0" exit
goto menu

:update
python loader_stats.py
python ingest_data.py
pause
goto menu

:update_and_serve
python loader_stats.py
python ingest_data.py
goto serve

:serve
start http://localhost:8000
python -m http.server 8000
pause
goto menu
