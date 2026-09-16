@echo off
REM Bismuth Prompt Studio -- update to the latest build (downloads only what changed)
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.10 or newer is required and must be on PATH. & pause & exit /b 1)
python tools\update.py %*
pause
