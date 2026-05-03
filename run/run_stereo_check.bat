@echo off
chcp 65001 >nul
cd /d "%~dp0"
C:\Users\user\miniconda3\envs\coin\python.exe check_stereo_pair_detection.py --pair-dir stereo_calib_pairs_7x9 --board-w 9 --board-h 6
pause
