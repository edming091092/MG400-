@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe coin_robot_ui.py
pause


