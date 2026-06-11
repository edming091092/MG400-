@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe check_stereo_pair_detection.py --pair-dir stereo_calib_pairs_6x8 --board-w 6 --board-h 8
pause
