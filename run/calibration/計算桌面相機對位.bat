@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe calibrate_quality_to_gemini_homography.py --pair-dir tabletop_homography_pairs_6x8 --board-w 6 --board-h 8
pause
