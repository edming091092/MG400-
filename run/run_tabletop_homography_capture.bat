@echo off
chcp 65001 >nul
cd /d "%~dp0"
C:\Users\user\miniconda3\envs\coin\python.exe capture_stereo_calib_pairs.py --out-dir tabletop_homography_pairs_9x6 --board-w 9 --board-h 6
pause
