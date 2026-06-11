@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe capture_one_stereo_side.py --side gemini --out-dir stereo_calib_pairs_6x8 --board-w 6 --board-h 8
pause
