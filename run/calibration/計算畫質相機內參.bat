@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\src"
C:\Users\user\miniconda3\envs\coin\python.exe calibrate_camera.py --img-dir quality_calib_images --out-json quality_camera_calib.json --preview-dir quality_calib_preview --board-w 9 --board-h 6 --square-mm 26
pause


