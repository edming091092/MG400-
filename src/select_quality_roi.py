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


def camera_backend(name):
    return {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }.get(str(name).lower(), cv2.CAP_DSHOW)


def apply_camera_props(cap, cfg, width, height, fps):
    fourcc = str(cfg.get("quality_fourcc", "MJPG")).strip().upper()
    if fourcc and fourcc not in ("NONE", "DEFAULT", "0"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4].ljust(4)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(cfg.get("quality_buffer_size", 1)))
    except Exception:
        pass


def fit_for_roi_selection(frame, max_w=1600, max_h=900):
    h, w = frame.shape[:2]
    scale = min(float(max_w) / max(float(w), 1.0), float(max_h) / max(float(h), 1.0), 1.0)
    if scale >= 0.999:
        return frame.copy(), 1.0
    out = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def main():
    cfg = load_config()
    index = int(cfg.get("quality_camera_index", 0))
    width = int(cfg.get("quality_width", 3840))
    height = int(cfg.get("quality_height", 2160))
    fps = int(cfg.get("quality_fps", 15))

    backend_name = str(cfg.get("quality_camera_backend", "dshow")).lower()
    backend_names = [backend_name, "dshow"] if backend_name != "msmf" else ["msmf", "dshow"]
    backend_names = list(dict.fromkeys(backend_names))
    cap = None
    for name in backend_names:
        candidate = cv2.VideoCapture(index, camera_backend(name))
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()
    if cap is None:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    apply_camera_props(cap, cfg, width, height, fps)
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

    view, scale = fit_for_roi_selection(frame)
    cv2.putText(view, "Drag coin/table ROI, ENTER=save, C=clear, ESC=cancel",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.namedWindow("Select Quality ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select Quality ROI", view, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Quality ROI")
    x, y, w, h = [int(round(float(v) / max(scale, 1e-6))) for v in roi]
    full_h, full_w = frame.shape[:2]
    x = max(0, min(x, full_w - 1))
    y = max(0, min(y, full_h - 1))
    w = max(0, min(w, full_w - x))
    h = max(0, min(h, full_h - y))
    if w <= 0 or h <= 0:
        cfg["quality_roi"] = None
        print("[ROI] 已清除 quality_roi")
    else:
        cfg["quality_roi"] = [x, y, x + w, y + h]
        print(f"[ROI] quality_roi={cfg['quality_roi']}  display_scale={scale:.3f}")
    save_config(cfg)


if __name__ == "__main__":
    main()
