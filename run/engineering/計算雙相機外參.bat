@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe calibrate_stereo_extrinsics.py --pair-dir stereo_calib_pairs_6x8 --board-w 6 --board-h 8 --square-mm 26 --zero-gemini-dist
pause
