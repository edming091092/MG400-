# -*- coding: utf-8 -*-
"""Fast camera preview snapshot without SAM3 recognition."""

import json
from pathlib import Path

import cv2


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


def main():
    cfg = load_config()
    OUT_DIR.mkdir(exist_ok=True)
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
        roi = cfg.get("quality_roi")
        if roi:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in roi]
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))
            frame = frame[y1:y2, x1:x2].copy()
        cv2.imwrite(str(OUT_DIR / "live_preview_quality.jpg"), frame)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
