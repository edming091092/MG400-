# -*- coding: utf-8 -*-
"""Stop current robot script, turn off suction DO13, and disable MG400."""

import subprocess
import sys
import socket
import time
from pathlib import Path


HERE = Path(__file__).parent
GEMINI_LIBS = Path(r"C:\Users\user\Desktop\sam3+座標轉換與夾取")

if str(GEMINI_LIBS) not in sys.path:
    sys.path.append(str(GEMINI_LIBS))

from core.robot import MG400


ROBOT_IP = "192.168.1.6"
DASH_PORT = 29999
SUCTION_DO = 13


def kill_robot_scripts():
    ps = (
        "$self=$PID; "
        "$procs=Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -match 'coin_classifier' -and $_.CommandLine -match 'hover_robot_target|unstack_coins' "
        "}; "
        "foreach ($p in $procs) { if ($p.ProcessId -ne $self) { "
        "try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } catch {} "
        "} }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace")


def raw_do13_off(label):
    cmds = [f"DO({SUCTION_DO},0)", f"DOExecute({SUCTION_DO},0)", f"DO({SUCTION_DO},0)"]
    try:
        with socket.create_connection((ROBOT_IP, DASH_PORT), timeout=2.0) as sock:
            sock.settimeout(2.0)
            for cmd in cmds:
                sock.sendall((cmd + "\n").encode("utf-8"))
                time.sleep(0.08)
                try:
                    resp = sock.recv(4096).decode("utf-8", errors="replace").strip()
                except Exception as exc:
                    resp = f"recv failed: {exc}"
                print(f"[ESTOP] {label} {cmd} -> {resp}")
    except Exception as exc:
        print(f"[ESTOP] {label} raw DO13 OFF failed: {exc}")


def main():
    kill_robot_scripts()
    raw_do13_off("before-disable")
    try:
        robot = MG400()
        robot.connect()
        try:
            robot.set_do(SUCTION_DO, 0)
            print("[ESTOP] DO13 OFF sent")
        except Exception as e:
            print(f"[ESTOP] DO13 OFF failed: {e}")
        try:
            robot.clear_error()
        except Exception:
            pass
        robot.disable()
        print("[ESTOP] MG400 DisableRobot sent")
        try:
            robot.disconnect()
        except Exception:
            pass
    except Exception as e:
        print(f"[ESTOP] Disable failed or robot not connected: {e}")
    raw_do13_off("after-disable")


if __name__ == "__main__":
    main()
