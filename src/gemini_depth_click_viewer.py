# -*- coding: utf-8 -*-
"""Gemini 2 live viewer: click the image to read aligned depth in mm."""

import argparse
import json
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from core.camera import Gemini2Camera


HERE = Path(__file__).parent
CONFIG_CANDIDATES = [
    HERE / "dual_camera_config.json",
    HERE.parent / "config" / "dual_camera_config.json",
]
WIN = "Gemini2 Depth Click Viewer"
BASELINE_FILE = HERE.parent / "config" / "gemini_table_depth_baseline.npz"


def load_config():
    for path in CONFIG_CANDIDATES:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def set_gemini_env(cfg):
    for key, env_name in (
        ("gemini_color_width", "GEMINI_COLOR_WIDTH"),
        ("gemini_color_height", "GEMINI_COLOR_HEIGHT"),
        ("gemini_color_fps", "GEMINI_COLOR_FPS"),
        ("gemini_color_format", "GEMINI_COLOR_FORMAT"),
    ):
        value = cfg.get(key)
        if value is not None:
            os.environ[env_name] = str(value)


def average_depth_stack(frames):
    if not frames:
        return None
    stack = np.stack(list(frames), axis=0).astype(np.float32)
    valid = stack > 0
    count = valid.sum(axis=0)
    avg = np.where(valid, stack, 0).sum(axis=0) / count.clip(min=1)
    avg[count == 0] = 0
    return avg.astype(np.float32)


def load_table_baseline():
    if not BASELINE_FILE.exists():
        return None
    try:
        data = np.load(str(BASELINE_FILE))
        baseline = data["depth_mm"].astype(np.float32)
        if baseline.ndim != 2:
            return None
        print(f"[calib] loaded table baseline: {BASELINE_FILE}")
        return baseline
    except Exception as exc:
        print(f"[calib] failed to load table baseline: {exc}")
        return None


def save_table_baseline(depth_mm):
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(BASELINE_FILE), depth_mm=depth_mm.astype(np.float32))
    print(f"[calib] saved table baseline: {BASELINE_FILE}")


def sample_depth_mm(depth_mm, x, y, radius):
    if depth_mm is None:
        return None
    h, w = depth_mm.shape[:2]
    x = int(round(x))
    y = int(round(y))
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    patch = depth_mm[y1:y2, x1:x2]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def sample_height_from_table_mm(depth_mm, table_depth_mm, x, y, radius):
    obj_z = sample_depth_mm(depth_mm, x, y, radius)
    table_z = sample_depth_mm(table_depth_mm, x, y, radius)
    if obj_z is None or table_z is None:
        return obj_z, table_z, None
    return obj_z, table_z, float(table_z - obj_z)


def parse_roi(value):
    if value is None:
        return None
    if len(value) != 4:
        return None
    x, y, w, h = [int(round(float(v))) for v in value]
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def crop_to_roi(image, roi):
    if roi is None:
        return image, (0, 0)
    h, w = image.shape[:2]
    x, y, rw, rh = roi
    x1 = max(0, min(w - 1, x))
    y1 = max(0, min(h - 1, y))
    x2 = max(x1 + 1, min(w, x + rw))
    y2 = max(y1 + 1, min(h, y + rh))
    return image[y1:y2, x1:x2], (x1, y1)


def fit_for_screen(image, max_w, max_h):
    h, w = image.shape[:2]
    scale = min(float(max_w) / max(1, w), float(max_h) / max(1, h), 1.0)
    if scale >= 0.999:
        return image, 1.0
    out = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def draw_hud(image, last_click, last_depth, last_table_depth, last_height, source_xy,
             paused, depth_shape, baseline_ready, calibrating, calib_count, calib_total):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 118), (0, 0, 0), -1)
    status = "PAUSED" if paused else "LIVE"
    cv2.putText(out, f"{status}  Click depth/height  c calibrate empty table  q/ESC quit",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "SPACE pause/resume",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 220, 255), 1, cv2.LINE_AA)
    if calibrating:
        calib_text = f"Table baseline: calibrating {calib_count}/{calib_total}"
        calib_color = (0, 220, 255)
    else:
        calib_text = "Table baseline: ready" if baseline_ready else "Table baseline: none"
        calib_color = (0, 255, 120) if baseline_ready else (80, 80, 255)
    cv2.putText(out, f"Depth map: {depth_shape[1]}x{depth_shape[0]} px   {calib_text}",
                (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, calib_color, 1, cv2.LINE_AA)
    if last_click is not None:
        x, y = last_click
        cv2.drawMarker(out, (x, y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
        sx, sy = source_xy
        if last_depth is None:
            msg = f"({sx}, {sy}) depth: invalid"
            color = (80, 80, 255)
        elif last_height is not None and last_table_depth is not None:
            msg = (
                f"({sx}, {sy}) depth: {last_depth:.1f} mm  "
                f"table: {last_table_depth:.1f} mm  height: {last_height:.1f} mm"
            )
            color = (0, 255, 120) if last_height >= -2.0 else (0, 220, 255)
        else:
            msg = f"({sx}, {sy}) depth: {last_depth:.1f} mm  ({last_depth / 10.0:.2f} cm)"
            color = (0, 255, 120)
        y_text = min(max(142, y - 18), out.shape[0] - 18)
        cv2.putText(out, msg, (12, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, msg, (12, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Show Gemini2 color image and read depth by mouse click.")
    parser.add_argument("--full", action="store_true", help="show full Gemini image instead of gemini_display_roi")
    parser.add_argument("--radius", type=int, default=5, help="median sample radius in source pixels")
    parser.add_argument("--avg-frames", type=int, default=5, help="number of depth frames to average")
    parser.add_argument("--calib-frames", type=int, default=45, help="frames to collect for empty-table calibration")
    parser.add_argument("--max-width", type=int, default=1280, help="maximum display width")
    parser.add_argument("--max-height", type=int, default=800, help="maximum display height")
    args = parser.parse_args()

    cfg = load_config()
    set_gemini_env(cfg)
    roi = None if args.full else parse_roi(cfg.get("gemini_display_roi"))
    depth_frames = deque(maxlen=max(1, args.avg_frames))
    table_baseline = [load_table_baseline()]
    calib_frames = []
    state = {
        "display_scale": 1.0,
        "roi_offset": (0, 0),
        "latest_depth": None,
        "last_click": None,
        "last_depth": None,
        "last_table_depth": None,
        "last_height": None,
        "last_source_xy": None,
        "paused": False,
        "calibrating": False,
    }

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        scale = max(state["display_scale"], 1e-6)
        ox, oy = state["roi_offset"]
        sx = int(round(x / scale + ox))
        sy = int(round(y / scale + oy))
        if table_baseline[0] is not None and table_baseline[0].shape == state["latest_depth"].shape:
            depth_value, table_value, height_value = sample_height_from_table_mm(
                state["latest_depth"], table_baseline[0], sx, sy, max(0, args.radius)
            )
        else:
            depth_value = sample_depth_mm(state["latest_depth"], sx, sy, max(0, args.radius))
            table_value = None
            height_value = None
        state["last_click"] = (x, y)
        state["last_depth"] = depth_value
        state["last_table_depth"] = table_value
        state["last_height"] = height_value
        state["last_source_xy"] = (sx, sy)
        if depth_value is None:
            print(f"[click] x={sx} y={sy} depth=invalid")
        elif height_value is not None and table_value is not None:
            print(
                f"[click] x={sx} y={sy} depth={depth_value:.1f} mm "
                f"table={table_value:.1f} mm height={height_value:.1f} mm"
            )
        else:
            print(f"[click] x={sx} y={sy} depth={depth_value:.1f} mm ({depth_value / 10.0:.2f} cm)")

    camera = Gemini2Camera(align_depth_to_color=True)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    try:
        camera.open()
        print("[Gemini2] 左鍵點畫面讀深度/桌面高度；c 空桌面校正；SPACE 暫停/繼續；q 或 ESC 離開。")
        last_color = None
        while True:
            if not state["paused"]:
                color, depth = camera.get_frames(timeout_ms=1000)
                if color is None or depth is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    continue
                last_color = color
                depth_frames.append(depth.astype(np.float32))
                state["latest_depth"] = average_depth_stack(depth_frames)
                if state["calibrating"]:
                    calib_frames.append(depth.astype(np.float32))
                    if len(calib_frames) >= max(1, args.calib_frames):
                        table_baseline[0] = average_depth_stack(calib_frames)
                        save_table_baseline(table_baseline[0])
                        calib_frames.clear()
                        state["calibrating"] = False
                        print("[calib] empty-table calibration done")

            if last_color is None or state["latest_depth"] is None:
                continue

            view, offset = crop_to_roi(last_color, roi)
            display, scale = fit_for_screen(view, args.max_width, args.max_height)
            state["display_scale"] = scale
            state["roi_offset"] = offset
            shown = draw_hud(
                display,
                state["last_click"],
                state["last_depth"],
                state["last_table_depth"],
                state["last_height"],
                state["last_source_xy"] or (0, 0),
                state["paused"],
                state["latest_depth"].shape,
                table_baseline[0] is not None and table_baseline[0].shape == state["latest_depth"].shape,
                state["calibrating"],
                len(calib_frames),
                max(1, args.calib_frames),
            )
            cv2.imshow(WIN, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:
                state["paused"] = not state["paused"]
                print("[Gemini2] pause" if state["paused"] else "[Gemini2] live")
            elif key in (ord("c"), ord("C")):
                calib_frames.clear()
                state["calibrating"] = True
                state["paused"] = False
                print(f"[calib] clear the table; collecting {max(1, args.calib_frames)} frames...")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
