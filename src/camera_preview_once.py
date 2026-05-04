# -*- coding: utf-8 -*-
"""Fast camera preview snapshot without SAM3 recognition."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from core.camera import Gemini2Camera


HERE = Path(__file__).parent
CONFIG_FILE = HERE / "dual_camera_config.json"
OUT_DIR = HERE / "test_output"


def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {}


def open_quality(cfg):
    index = int(cfg.get("quality_camera_index", 0))
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.get("quality_width", 1280)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.get("quality_height", 720)))
    cap.set(cv2.CAP_PROP_FPS, int(cfg.get("quality_fps", 30)))
    return cap


def crop_roi(frame, roi):
    if frame is None or not roi:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in roi]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return frame[y1:y2, x1:x2].copy()


def resize_to_height(frame, height):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if h <= 0:
        return frame
    new_w = max(1, int(w * (height / h)))
    return cv2.resize(frame, (new_w, height), interpolation=cv2.INTER_AREA)


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


def grab_quality_frame(cfg):
    cap = open_quality(cfg)
    if not cap.isOpened():
        raise RuntimeError("畫質相機開啟失敗")
    frame = None
    try:
        for _ in range(8):
            ok, img = cap.read()
            if ok and img is not None:
                frame = img
        if frame is None:
            raise RuntimeError("畫質相機沒有影像")
        return crop_roi(frame, cfg.get("quality_roi"))
    finally:
        cap.release()


def grab_gemini_frame(cfg):
    set_gemini_env(cfg)
    camera = Gemini2Camera(align_depth_to_color=True)
    try:
        camera.open()
        frame = None
        for _ in range(8):
            color, _depth = camera.get_frames(timeout_ms=1000)
            if color is not None:
                frame = color.copy()
        if frame is None:
            raise RuntimeError("深度相機沒有影像")
        return crop_roi(frame, cfg.get("gemini_display_roi"))
    finally:
        camera.close()


def save_combined_preview(quality_frame, gemini_frame):
    target_h = 540
    q = resize_to_height(quality_frame, target_h)
    g = resize_to_height(gemini_frame, target_h)
    if q is None:
        q = np.zeros((target_h, target_h, 3), dtype=np.uint8)
    if g is None:
        g = np.zeros((target_h, target_h, 3), dtype=np.uint8)
    gap = np.zeros((target_h, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(OUT_DIR / "live_preview_combined.jpg"), np.hstack([q, gap, g]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["Quality", "Gemini", "Combined"], default="Quality")
    args = parser.parse_args()
    cfg = load_config()
    OUT_DIR.mkdir(exist_ok=True)
    if args.view == "Quality":
        cv2.imwrite(str(OUT_DIR / "live_preview_quality.jpg"), grab_quality_frame(cfg))
        return
    if args.view == "Gemini":
        cv2.imwrite(str(OUT_DIR / "live_preview_gemini.jpg"), grab_gemini_frame(cfg))
        return
    quality_frame = grab_quality_frame(cfg)
    gemini_frame = grab_gemini_frame(cfg)
    cv2.imwrite(str(OUT_DIR / "live_preview_quality.jpg"), quality_frame)
    cv2.imwrite(str(OUT_DIR / "live_preview_gemini.jpg"), gemini_frame)
    save_combined_preview(quality_frame, gemini_frame)


if __name__ == "__main__":
    main()
