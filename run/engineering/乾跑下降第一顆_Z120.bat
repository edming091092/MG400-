@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe hover_robot_target.py --safe-z 100 --lower-z -120
pause


