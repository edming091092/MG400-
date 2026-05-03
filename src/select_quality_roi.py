# -*- coding: utf-8 -*-
"""Select Quality camera ROI and save it to dual_camera_config.json."""

import json
from pathlib import Path

import cv2


HERE = Path(__file__).parent
CONFIG_FILE = HERE / "dual_camera_config.json"


def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {}


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    cfg = load_config()
    index = int(cfg.get("quality_camera_index", 0))
    width = int(cfg.get("quality_width", 1280))
    height = int(cfg.get("quality_height", 720))
    fps = int(cfg.get("quality_fps", 30))

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Quality camera index={index} open failed")

    frame = None
    for _ in range(20):
        ok, img = cap.read()
        if ok and img is not None:
            frame = img
    cap.release()
    if frame is None:
        raise RuntimeError("No frame from Quality camera")

    cv2.putText(frame, "Drag coin/table ROI, ENTER=save, C=clear, ESC=cancel",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
    roi = cv2.selectROI("Select Quality ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Quality ROI")
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        cfg["quality_roi"] = None
        print("[ROI] 已清除 quality_roi")
    else:
        cfg["quality_roi"] = [x, y, x + w, y + h]
        print(f"[ROI] quality_roi={cfg['quality_roi']}")
    save_config(cfg)


if __name__ == "__main__":
    main()
