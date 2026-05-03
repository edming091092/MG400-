"""
MG400 TCP/IP 控制模組
參考現有 border_collect_manual.py 與 do_test.py 整合而成
"""

import socket
import time
import re
from typing import Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class RobotError(Exception):
    pass


class MG400:
    def __init__(self, ip: str = config.ROBOT_IP,
                 dash_port: int = config.DASH_PORT,
                 move_port: int = config.MOVE_PORT,
                 timeout: float = 3.0):
        self.ip = ip
        self.dash_port = dash_port
        self.move_port = move_port
        self.timeout = timeout
        self._dash: Optional[socket.socket] = None
        self._move: Optional[socket.socket] = None
        self.last_errors = []
        self.last_response = ""

    # ------------------------------------------------------------------ #
    #  連線 / 斷線
    # ------------------------------------------------------------------ #
    def connect(self):
        self._dash = self._open_socket(self.dash_port)
        self._move = self._open_socket(self.move_port)
        print(f"[robot] 連線成功 {self.ip}  dash={self.dash_port}  move={self.move_port}")

    def disconnect(self):
        for s in (self._dash, self._move):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        self._dash = self._move = None
        print("[robot] 已斷線")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------ #
    #  基本指令
    # ------------------------------------------------------------------ #
    def enable(self):
        """啟用機器人"""
        resp = self._send(self._dash, "EnableRobot()")
        print(f"[robot] EnableRobot → {resp}")
        return resp

    def disable(self):
        resp = self._send(self._dash, "DisableRobot()")
        print(f"[robot] DisableRobot → {resp}")
        return resp

    def clear_error(self):
        try:
            self._send(self._dash, "ClearError()")
        except Exception:
            pass
        try:
            self._send(self._dash, "Continue()")
        except Exception:
            pass

    def get_pose(self) -> Optional[Tuple[float, float, float, float]]:
        """回傳 (x, y, z, r) mm / deg，失敗回 None"""
        resp = self._send(self._dash, "GetPose()")
        return self._parse_pose(resp)

    def get_errors(self):
        resp = self._send(self._dash, "GetErrorID()")
        nums = [int(x) for x in re.findall(r"-?\d+", resp)]
        return nums[1:] if len(nums) > 1 else []

    def set_speed(self, pct: int):
        self._send(self._dash, f"SpeedFactor({max(1, min(100, pct))})")

    # ------------------------------------------------------------------ #
    #  探測下降：緩慢下降直到碰撞，回傳接觸 Z（mm）
    # ------------------------------------------------------------------ #
    def probe_z(self, x: float, y: float,
                target_z: float = -250.0,
                timeout_s: float = 20.0) -> Optional[float]:
        """
        固定 6% 速度緩慢下降，碰到障礙物停止。
        回傳接觸時的 Z (mm)；超時或未偵測到碰撞回 None。
        """
        self.clear_error()
        self._send(self._dash, "SetCollisionLevel(1)")
        self._send(self._dash, "SpeedFactor(6)")
        cmd = f"MovL({x:.3f},{y:.3f},{target_z:.3f},0.000)\n"
        self._move.sendall(cmd.encode("utf-8"))
        time.sleep(0.1)

        contact_z: Optional[float] = None
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            errs = self.get_errors()
            if errs:
                pose = self.get_pose()
                if pose:
                    contact_z = pose[2]
                self.clear_error()
                break
            pose = self.get_pose()
            if pose and abs(pose[2] - target_z) < 1.5:
                break
            time.sleep(0.04)

        self._send(self._dash, f"SpeedFactor({config.MOVE_SPEED_PCT})")
        self._send(self._dash, "SetCollisionLevel(3)")
        return contact_z

    # ------------------------------------------------------------------ #
    #  MovL：直線移動並等待到位
    # ------------------------------------------------------------------ #
    def movl(self, x: float, y: float, z: float, r: float = 0.0,
             timeout_s: float = config.MOVE_TIMEOUT_S,
             tol_mm: float = config.MOVE_TOL_MM) -> bool:
        """
        直線移動到 (x, y, z, r)，等待到位。
        回傳 True=到位, False=超時或報錯。
        """
        self.clear_error()
        resp = self._send(self._move, f"MovL({x:.3f},{y:.3f},{z:.3f},{r:.3f})")
        self.last_response = resp
        if not resp.startswith("0"):
            print(f"[robot] MovL 指令被拒: {resp}")
            return False

        t0 = time.time()
        while True:
            errs = self.get_errors()
            if errs:
                self.last_errors = errs
                print(f"[robot] MovL 中斷 error={errs}")
                self.clear_error()
                return False

            pose = self.get_pose()
            if pose is not None:
                dist = ((pose[0]-x)**2 + (pose[1]-y)**2 + (pose[2]-z)**2) ** 0.5
                if dist <= tol_mm:
                    return True

            if time.time() - t0 > timeout_s:
                print(f"[robot] MovL 超時 target=({x},{y},{z})")
                self.last_response = f"MovL timeout target=({x},{y},{z})"
                return False

            time.sleep(0.05)

    # ------------------------------------------------------------------ #
    #  DO 數位輸出（吸盤/夾爪）
    # ------------------------------------------------------------------ #
    def set_do(self, index: int, state: int):
        """state: 1=ON, 0=OFF"""
        self._send(self._dash, f"DOExecute({index},{state})")

    def movj(self, x: float, y: float, z: float, r: float = 0.0,
             timeout_s: float = config.MOVE_TIMEOUT_S,
             tol_mm: float = config.MOVE_TOL_MM) -> bool:
        """關節移動到 (x, y, z, r)，用於高空轉移，避免直線路徑規劃失敗。"""
        self.clear_error()
        resp = self._send(self._move, f"MovJ({x:.3f},{y:.3f},{z:.3f},{r:.3f})")
        self.last_response = resp
        if not resp.startswith("0"):
            print(f"[robot] MovJ 指令被拒: {resp}")
            return False

        t0 = time.time()
        while True:
            errs = self.get_errors()
            if errs:
                self.last_errors = errs
                print(f"[robot] MovJ 中斷 error={errs}")
                self.clear_error()
                return False
            pose = self.get_pose()
            if pose is not None:
                dist = ((pose[0]-x)**2 + (pose[1]-y)**2 + (pose[2]-z)**2) ** 0.5
                if dist <= tol_mm:
                    return True
            if time.time() - t0 > timeout_s:
                print(f"[robot] MovJ 超時 target=({x},{y},{z})")
                self.last_response = f"MovJ timeout target=({x},{y},{z})"
                return False
            time.sleep(0.05)

    # ------------------------------------------------------------------ #
    #  內部工具
    # ------------------------------------------------------------------ #
    def _open_socket(self, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.ip, port))
        return s

    def _send(self, sock: socket.socket, cmd: str) -> str:
        if not cmd.endswith("\n"):
            cmd += "\n"
        sock.sendall(cmd.encode("utf-8"))
        time.sleep(0.02)
        try:
            data = sock.recv(4096).decode("utf-8", errors="ignore").strip()
            return data
        except socket.timeout:
            return ""

    @staticmethod
    def _parse_pose(resp: str) -> Optional[Tuple[float, float, float, float]]:
        lb = resp.find("{")
        rb = resp.find("}")
        if lb < 0 or rb <= lb:
            return None
        parts = resp[lb+1:rb].split(",")
        if len(parts) < 4:
            return None
        try:
            return tuple(float(p.strip()) for p in parts[:4])
        except ValueError:
            return None
