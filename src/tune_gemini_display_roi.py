# tune_gemini_display_roi.py
# -*- coding: utf-8 -*-

"""用滑桿即時調整 Gemini 顯示裁切範圍，並存回 dual_camera_config.json。"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from calibrate_camera import resize_for_show
from gemini_controls import set_gemini_stream_env


HERE = Path(__file__).parent
CONFIG_FILE = HERE / "dual_camera_config.json"
GEMINI_LIBS = Path(r"C:\Users\user\Desktop\sam3+座標轉換與夾取")

if str(GEMINI_LIBS) not in sys.path:
    sys.path.append(str(GEMINI_LIBS))

from core.camera import Gemini2Camera


def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {}


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def clamp_roi(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 2))
    y1 = max(0, min(int(y1), h - 2))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def make_panel(full, roi):
    h, w = full.shape[:2]
    x1, y1, x2, y2 = roi
    marked = full.copy()
    cv2.rectangle(marked, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 3)
    cv2.rectangle(marked, (0, 0), (w, 54), (20, 20, 20), -1)
    cv2.putText(
        marked,
        f"ROI=[{x1}, {y1}, {x2}, {y2}]  S=save  R=reset full  Q=quit",
        (12, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    crop = full[y1:y2, x1:x2]
    crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(crop, (0, 0), (w, 54), (20, 20, 20), -1)
    cv2.putText(
        crop,
        "Preview: Gemini display after crop",
        (12, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 120),
        2,
        cv2.LINE_AA,
    )
    return np.hstack([resize_for_show(marked, 960, 540), resize_for_show(crop, 960, 540)])


def main():
    cfg = load_config()
    camera = Gemini2Camera(align_depth_to_color=True)
    win = "Tune Gemini Display ROI"

    try:
        set_gemini_stream_env(cfg)
        camera.open()
        frame = None
        for _ in range(10):
            frame, _ = camera.get_frames(timeout_ms=1000)
            if frame is not None:
                break
        if frame is None:
            raise RuntimeError("Gemini 沒有取得畫面")

        h, w = frame.shape[:2]
        default_roi = cfg.get("gemini_display_roi") or [0, 0, w, h]
        x1, y1, x2, y2 = clamp_roi(*default_roi, w, h)

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1920, 540)
        cv2.createTrackbar("left", win, x1, w - 2, lambda _v: None)
        cv2.createTrackbar("top", win, y1, h - 2, lambda _v: None)
        cv2.createTrackbar("right", win, x2, w, lambda _v: None)
        cv2.createTrackbar("bottom", win, y2, h, lambda _v: None)

        print("=" * 60)
        print("Gemini 顯示裁切範圍調整")
        print("=" * 60)
        print("滑桿：left/top/right/bottom")
        print("S：儲存到 dual_camera_config.json")
        print("R：恢復完整畫面")
        print("Q/ESC：離開")

        while True:
            live, _ = camera.get_frames(timeout_ms=1000)
            if live is not None:
                frame = live

            x1 = cv2.getTrackbarPos("left", win)
            y1 = cv2.getTrackbarPos("top", win)
            x2 = cv2.getTrackbarPos("right", win)
            y2 = cv2.getTrackbarPos("bottom", win)
            roi = clamp_roi(x1, y1, x2, y2, w, h)
            if roi != (x1, y1, x2, y2):
                cv2.setTrackbarPos("left", win, roi[0])
                cv2.setTrackbarPos("top", win, roi[1])
                cv2.setTrackbarPos("right", win, roi[2])
                cv2.setTrackbarPos("bottom", win, roi[3])

            cv2.imshow(win, make_panel(frame, roi))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                cv2.setTrackbarPos("left", win, 0)
                cv2.setTrackbarPos("top", win, 0)
                cv2.setTrackbarPos("right", win, w)
                cv2.setTrackbarPos("bottom", win, h)
            if key in (ord("s"), ord("S")):
                cfg["gemini_display_roi"] = list(roi)
                save_config(cfg)
                print(f"[儲存] gemini_display_roi = {cfg['gemini_display_roi']}")

    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
