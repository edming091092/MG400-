# -*- coding: utf-8 -*-
"""
Manual Gemini pixel -> MG400 tabletop XY calibration.

Move the MG400 TCP to several tabletop points, click the same point in the
Gemini image, then press Enter/C to record the current robot X/Y.
"""

import datetime
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from gemini_controls import apply_color_controls, set_gemini_stream_env
from dual_camera_live import load_config


HERE = Path(__file__).parent
GEMINI_LIBS = Path(r"C:\Users\user\Desktop\sam3+座標轉換與夾取")
OUT_JSON = HERE / "robot_tabletop_homography.json"
DEBUG_IMAGE = HERE / "test_output" / "robot_tabletop_homography_debug.jpg"

if str(GEMINI_LIBS) not in sys.path:
    sys.path.append(str(GEMINI_LIBS))

from core.camera import Gemini2Camera
from core.robot import MG400


clicked = [None]


def on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked[0] = (float(x), float(y))


def try_connect_robot():
    try:
        robot = MG400()
        robot.connect()
        robot.enable()
        print("[MG400] 連線成功，按 Enter/C 會讀取目前 X/Y")
        return robot
    except Exception as e:
        print(f"[MG400] 連線失敗，改用手動輸入 X Y：{e}")
        return None


def get_robot_xy(robot):
    if robot is not None:
        pose = robot.get_pose()
        if pose is not None:
            return float(pose[0]), float(pose[1])
        print("[MG400] GetPose 失敗，請重試或手動輸入")
    while True:
        text = input("輸入 MG400 X Y，例如 300 -120，或 q 離開: ").strip()
        if text.lower() in ("q", "quit", "exit"):
            return None
        parts = text.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
        print("格式錯誤，請輸入兩個數字")


def draw_points(frame, image_points, robot_points):
    out = frame.copy()
    for i, ((u, v), (x, y)) in enumerate(zip(image_points, robot_points), 1):
        cv2.drawMarker(out, (int(round(u)), int(round(v))), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(out, f"{i}: X{x:.0f} Y{y:.0f}", (int(u) + 12, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    if clicked[0] is not None:
        u, v = clicked[0]
        cv2.drawMarker(out, (int(u), int(v)), (0, 120, 255), cv2.MARKER_TILTED_CROSS, 26, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 54), (20, 20, 20), -1)
    cv2.putText(out, "Click TCP point, Enter/C=record, S=save, U=undo, Q=quit",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(out, f"points={len(image_points)}  need >=4, spread across table",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def save_homography(image_points, robot_points, frame, cfg):
    if len(image_points) < 4:
        print("[校正] 至少需要 4 點")
        return False
    src = np.array(image_points, dtype=np.float32)
    dst = np.array(robot_points, dtype=np.float32)
    h_mat, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if h_mat is None:
        print("[校正] findHomography 失敗，點位可能太集中或共線")
        return False

    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
    errors = np.linalg.norm(proj - dst, axis=1)
    payload = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "manual_tabletop_points",
        "gemini_to_robot_homography": h_mat.tolist(),
        "image_points_px": [[float(u), float(v)] for u, v in image_points],
        "robot_points_xy_mm": [[float(x), float(y)] for x, y in robot_points],
        "robot_table_z_mm": float(cfg.get("robot_table_z_mm", -160.0)),
        "mean_error_mm": float(np.mean(errors)),
        "max_error_mm": float(np.max(errors)),
        "inliers": int(mask.sum()) if mask is not None else len(image_points),
        "n_points": len(image_points),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    DEBUG_IMAGE.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(DEBUG_IMAGE), draw_points(frame, image_points, robot_points))
    print(f"[校正] 已儲存 {OUT_JSON}")
    print(f"[校正] mean={payload['mean_error_mm']:.2f}mm  max={payload['max_error_mm']:.2f}mm")
    print(f"[校正] 診斷圖 {DEBUG_IMAGE}")
    return True


def main():
    cfg = load_config()
    set_gemini_stream_env(cfg)
    robot = try_connect_robot()
    camera = Gemini2Camera(align_depth_to_color=True)
    image_points = []
    robot_points = []
    win = "MG400 tabletop calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    try:
        camera.open()
        apply_color_controls(camera, cfg)
        while True:
            color, _depth = camera.get_frames(timeout_ms=1000)
            if color is None:
                continue
            vis = draw_points(color, image_points, robot_points)
            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (13, 10, ord("c"), ord("C")):
                if clicked[0] is None:
                    print("[校正] 先在畫面點手臂 TCP 所在位置")
                    continue
                xy = get_robot_xy(robot)
                if xy is None:
                    break
                image_points.append(clicked[0])
                robot_points.append(xy)
                print(f"[校正] P{len(image_points)} image=({clicked[0][0]:.1f},{clicked[0][1]:.1f}) robot=({xy[0]:.1f},{xy[1]:.1f})")
                clicked[0] = None
            elif key in (ord("u"), ord("U")):
                if image_points:
                    image_points.pop()
                    robot_points.pop()
                    print(f"[校正] undo，剩 {len(image_points)} 點")
            elif key in (ord("s"), ord("S")):
                save_homography(image_points, robot_points, color, cfg)
    finally:
        if robot is not None:
            robot.disconnect()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
