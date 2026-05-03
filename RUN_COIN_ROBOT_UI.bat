@echo off
setlocal
cd /d "%~dp0src"
call C:\Users\user\miniconda3\Scripts\activate.bat coin
python coin_robot_ui.py
pause
