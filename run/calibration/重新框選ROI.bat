@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe select_quality_roi.py
pause


