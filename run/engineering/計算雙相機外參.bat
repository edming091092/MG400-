@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe calibrate_stereo_extrinsics.py --pair-dir stereo_calib_pairs_7x9 --board-w 9 --board-h 6 --square-mm 26 --zero-gemini-dist
pause


