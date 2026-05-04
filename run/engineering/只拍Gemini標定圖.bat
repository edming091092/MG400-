@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe capture_one_stereo_side.py --side gemini --out-dir stereo_calib_pairs_7x9 --board-w 9 --board-h 6
pause


