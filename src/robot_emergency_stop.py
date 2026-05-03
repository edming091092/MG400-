# -*- coding: utf-8 -*-
"""Stop current robot script and disable MG400."""

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).parent
GEMINI_LIBS = Path(r"C:\Users\user\Desktop\sam3+座標轉換與夾取")

if str(GEMINI_LIBS) not in sys.path:
    sys.path.append(str(GEMINI_LIBS))

from core.robot import MG400


def kill_robot_scripts():
    ps = (
        "$self=$PID; "
        "$procs=Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -match 'coin_classifier' -and $_.CommandLine -match 'hover_robot_target' "
        "}; "
        "foreach ($p in $procs) { if ($p.ProcessId -ne $self) { "
        "try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } catch {} "
        "} }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace")


def main():
    kill_robot_scripts()
    try:
        robot = MG400()
        robot.connect()
        try:
            robot.clear_error()
        except Exception:
            pass
        robot.disable()
        print("[ESTOP] MG400 DisableRobot sent")
    except Exception as e:
        print(f"[ESTOP] Disable failed or robot not connected: {e}")


if __name__ == "__main__":
    main()
