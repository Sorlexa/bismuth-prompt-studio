@echo off
REM Bismuth Prompt Studio -- install the language-model engine (llama.cpp + a model)
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.10 or newer is required and must be on PATH. & pause & exit /b 1)
python tools\setup_models.py %*
pause
