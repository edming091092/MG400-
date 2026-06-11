@echo off
cd /d "%~dp0"
echo ============================================================
echo YOLOv8m 960 visible training
echo Output: %~dp0runs_yolo_coin\yolov8m_960_visible
echo ============================================================
C:\Users\user\miniconda3\envs\coin\python.exe src\train_yolov8m_visible.py
echo.
echo ============================================================
echo Training process ended. Press any key to close.
echo ============================================================
pause
