@echo off
REM Bismuth Prompt Studio -- local web UI on http://localhost:7801
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.10 or newer is required and must be on PATH. & pause & exit /b 1)
python -m promptstudio.ui.studio %*
pause
