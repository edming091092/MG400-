@echo off
chcp 65001 >nul
cd /d "%~dp0"
C:\Users\user\miniconda3\envs\coin\python.exe tune_gemini_display_roi.py
pause
