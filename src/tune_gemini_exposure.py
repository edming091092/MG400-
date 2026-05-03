# tune_gemini_exposure.py
# -*- coding: utf-8 -*-

"""用滑桿即時調整 Gemini 彩色相機曝光/增益，並存回 dual_camera_config.json。"""

import json
import sys
from pathlib import Path

import cv2

from calibrate_camera import resize_for_show
from gemini_controls import apply_color_controls, get_color_control_ranges, get_color_controls, set_gemini_stream_env


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


def scale_to_slider(value, min_v, max_v):
    if max_v <= min_v:
        return 0
    value = max(min_v, min(int(value), max_v))
    return int(round((value - min_v) * 1000 / (max_v - min_v)))


def slider_to_value(pos, min_v, max_v, step):
    if max_v <= min_v:
        return min_v
    raw = min_v + int(round(pos * (max_v - min_v) / 1000))
    step = max(1, int(step))
    return int(round((raw - min_v) / step) * step + min_v)


def draw_overlay(frame, auto_exposure, exposure, gain, roi, profile_name, saved=False):
    out = frame.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 88), (20, 20, 20), -1)
    mode = "AUTO" if auto_exposure else "MANUAL"
    msg = f"Gemini: {profile_name}  exposure: {mode}  exposure={exposure}  gain={gain}"
    cv2.putText(out, msg, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"ROI=[{x1},{y1},{x2},{y2}]  A=auto/manual  P=profile  S=save  Q=quit", (14, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 120), 2, cv2.LINE_AA)
    if saved:
        cv2.putText(out, "SAVED", (out.shape[1] - 130, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def value_or_default(value, default):
    return default if value is None else value


def clamp_roi(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 2))
    y1 = max(0, min(int(y1), h - 2))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def main():
    cfg = load_config()
    camera = Gemini2Camera(align_depth_to_color=True)
    win = "Tune Gemini Exposure"
    saved_ticks = 0

    try:
        set_gemini_stream_env(cfg)
        camera.open()
        apply_color_controls(camera, cfg, verbose=True)
        ranges = get_color_control_ranges(camera)
        current = get_color_controls(camera)

        exp_min, exp_max, exp_step = ranges.get("exposure", (1, 20000, 1))
        gain_min, gain_max, gain_step = ranges.get("gain", (1, 255, 1))
        auto = bool(cfg.get("gemini_color_auto_exposure", current.get("auto_exposure", True)))
        exposure = int(value_or_default(cfg.get("gemini_color_exposure"), current.get("exposure", exp_min)))
        gain = int(value_or_default(cfg.get("gemini_color_gain"), current.get("gain", gain_min)))
        h, w = 720, 1280
        roi = clamp_roi(*(cfg.get("gemini_display_roi") or [0, 0, w, h]), w, h)
        profiles = [
            ("1280x720 RGB 30", 1280, 720, "RGB", 30),
            ("640x480 RGB 30", 640, 480, "RGB", 30),
            ("1280x720 MJPG 30", 1280, 720, "MJPG", 30),
            ("640x480 MJPG 30", 640, 480, "MJPG", 30),
        ]
        current_profile = (
            int(cfg.get("gemini_color_width", 1280)),
            int(cfg.get("gemini_color_height", 720)),
            str(cfg.get("gemini_color_format", "RGB")).upper(),
            int(cfg.get("gemini_color_fps", 30)),
        )
        profile_idx = 0
        for i, (_name, pw, ph, pfmt, pfps) in enumerate(profiles):
            if (pw, ph, pfmt, pfps) == current_profile:
                profile_idx = i
                break

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.createTrackbar("exposure", win, scale_to_slider(exposure, exp_min, exp_max), 1000, lambda _v: None)
        cv2.createTrackbar("gain", win, scale_to_slider(gain, gain_min, gain_max), 1000, lambda _v: None)
        cv2.createTrackbar("left", win, roi[0], w - 2, lambda _v: None)
        cv2.createTrackbar("top", win, roi[1], h - 2, lambda _v: None)
        cv2.createTrackbar("right", win, roi[2], w, lambda _v: None)
        cv2.createTrackbar("bottom", win, roi[3], h, lambda _v: None)

        print("=" * 60)
        print("Gemini 彩色相機曝光調整")
        print("=" * 60)
        print(f"Exposure range: min={exp_min}, max={exp_max}, step={exp_step}")
        print(f"Gain range:     min={gain_min}, max={gain_max}, step={gain_step}")
        print("A：切換自動/手動曝光")
        print("P：切換下次啟動要使用的 Gemini 解析度/格式")
        print("S：儲存到 dual_camera_config.json")
        print("Q/ESC：離開")

        last_exposure = None
        last_gain = None
        last_auto = None

        while True:
            exposure = slider_to_value(cv2.getTrackbarPos("exposure", win), exp_min, exp_max, exp_step)
            gain = slider_to_value(cv2.getTrackbarPos("gain", win), gain_min, gain_max, gain_step)
            roi = clamp_roi(
                cv2.getTrackbarPos("left", win),
                cv2.getTrackbarPos("top", win),
                cv2.getTrackbarPos("right", win),
                cv2.getTrackbarPos("bottom", win),
                w,
                h,
            )
            cv2.setTrackbarPos("left", win, roi[0])
            cv2.setTrackbarPos("top", win, roi[1])
            cv2.setTrackbarPos("right", win, roi[2])
            cv2.setTrackbarPos("bottom", win, roi[3])
            if (auto, exposure, gain) != (last_auto, last_exposure, last_gain):
                temp_cfg = {
                    "gemini_color_auto_exposure": auto,
                    "gemini_color_exposure": exposure,
                    "gemini_color_gain": gain,
                }
                apply_color_controls(camera, temp_cfg, verbose=False)
                last_auto, last_exposure, last_gain = auto, exposure, gain

            frame, _ = camera.get_frames(timeout_ms=1000)
            if frame is not None:
                cv2.imshow(win, resize_for_show(draw_overlay(frame, auto, exposure, gain, roi, profiles[profile_idx][0], saved_ticks > 0), 1280, 720))
            if saved_ticks > 0:
                saved_ticks -= 1

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("a"), ord("A")):
                auto = not auto
                print(f"[模式] auto_exposure={auto}")
            if key in (ord("p"), ord("P")):
                profile_idx = (profile_idx + 1) % len(profiles)
                print(f"[解析度] 下次啟動使用：{profiles[profile_idx][0]}")
            if key in (ord("s"), ord("S")):
                _name, pw, ph, pfmt, pfps = profiles[profile_idx]
                cfg["gemini_color_auto_exposure"] = bool(auto)
                cfg["gemini_color_exposure"] = int(exposure)
                cfg["gemini_color_gain"] = int(gain)
                cfg["gemini_display_roi"] = list(roi)
                cfg["gemini_color_width"] = int(pw)
                cfg["gemini_color_height"] = int(ph)
                cfg["gemini_color_format"] = pfmt
                cfg["gemini_color_fps"] = int(pfps)
                save_config(cfg)
                saved_ticks = 60
                print(f"[儲存] auto={auto}, exposure={exposure}, gain={gain}, roi={roi}, profile={_name}")

    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
