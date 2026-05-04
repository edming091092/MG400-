@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe capture_quality_calib.py --board-w 9 --board-h 6
pause


