@echo off
setlocal
cd /d "%~dp0src"

if "%CLEAN_OLD_VISION%"=="1" (
  echo [CoinVision] checking old camera/vision processes...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$patterns='dual_camera_live|camera_preview_once|select_quality_roi|capture_quality_calib|capture_stereo_calib_pairs|calibrate_robot_tabletop_homography'; $procs=Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'coin_mg400_product_package' -and $_.CommandLine -match $patterns }; foreach ($p in $procs) { Write-Host ('[CoinVision] stopping old vision process {0}' -f $p.ProcessId); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }; Start-Sleep -Milliseconds 300"
) else (
  echo [CoinVision] skip old camera cleanup
)

start "" C:\Users\user\miniconda3\envs\coin\pythonw.exe coin_robot_ui.py
exit /b 0
