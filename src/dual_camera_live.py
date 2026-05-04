# dual_camera_live.py
# -*- coding: utf-8 -*-
"""
雙相機硬幣辨識第一版

Gemini 2：負責 SAM3 偵測、深度 Z、直徑 mm、幣別分類。
高畫質相機：只負責即時清晰畫面、ROI、blur score，不參與 mm 換算。

按鍵：
  SPACE = 快照/回到即時（快照會累積多幀深度）
  S     = 儲存目前一筆 log 到 dual_measure_log.csv
  Q/ESC = 離開
"""

import csv
import datetime
import json
import os
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from gemini_controls import apply_color_controls, set_gemini_stream_env


HERE = Path(__file__).parent
CONFIG_FILE = HERE / "dual_camera_config.json"
LOG_FILE = HERE / "dual_measure_log.csv"
OUT_DIR = HERE / "test_output"
HOMOGRAPHY_FILE = HERE / "quality_to_gemini_homography.json"
STEREO_EXTRINSICS_FILE = HERE / "stereo_extrinsics.json"
ROBOT_TARGETS_FILE = HERE / "robot_targets.json"
ROBOT_TABLETOP_H_FILE = HERE / "robot_tabletop_homography.json"
COIN_DEPTH_DIR = Path(r"C:\Users\user\Desktop\coin_depth")
GEMINI_LIBS = Path(r"C:\Users\user\Desktop\sam3+座標轉換與夾取")
ROBOT_DATA_DIR = GEMINI_LIBS / "data"
CALIB_FILE = COIN_DEPTH_DIR / "calibration.json"

for p in (GEMINI_LIBS,):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from core.camera import Gemini2Camera


DISP_H = 720
GEMINI_DISP_W = 960
QUALITY_DISP_W = 640
PANEL_W = 300
TOTAL_W = GEMINI_DISP_W + QUALITY_DISP_W + PANEL_W

CALIB_TOL_MM = 1.5
CLASSES = ["1yuan", "5yuan", "10yuan", "50yuan"]
LABEL_NAME = {
    "1yuan": "1NT",
    "5yuan": "5NT",
    "10yuan": "10NT",
    "50yuan": "50NT",
    "?": "?",
}
COIN_COLOR = {
    "1yuan": (255, 200, 0),
    "5yuan": (0, 200, 0),
    "10yuan": (0, 100, 255),
    "50yuan": (0, 200, 255),
    "?": (100, 100, 100),
}
COIN_VALUE = {
    "1yuan": 1,
    "5yuan": 5,
    "10yuan": 10,
    "50yuan": 50,
}
PICK_X_MIN = 120.0
PICK_X_MAX = 380.0
PICK_Y_MIN = -250.0
PICK_Y_MAX = 190.0

CIRCULARITY_MIN = 0.65
_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

LOG_FIELDS = [
    "timestamp",
    "fusion_mode",
    "quality_camera_index",
    "quality_blur_score",
    "quality_roi_x1",
    "quality_roi_y1",
    "quality_roi_x2",
    "quality_roi_y2",
    "coin_index",
    "center_x",
    "center_y",
    "diameter_mm",
    "depth_z_mm",
    "world_x_mm",
    "world_y_mm",
    "world_z_mm",
    "gemini_x_px",
    "gemini_y_px",
    "predicted_class",
    "class_diff_mm",
    "valid",
]


def load_config():
    defaults = {
        "quality_camera_index": 1,
        "gemini_camera_index": 2,
        "quality_width": 1280,
        "quality_height": 720,
        "quality_fps": 30,
        "quality_roi": None,
        "quality_min_area": 350,
        "quality_max_area": 20000,
        "quality_min_axis": 12,
        "quality_max_axis": 180,
        "quality_detection_method": "sam3_ellipse",
        "sam3_model_path": "",
        "quality_sam3_interval_frames": 30,
        "quality_diameter_history": 5,
        "quality_min_class_margin_mm": 0.45,
        "gemini_snapshot_frames": 80,
        "gemini_depth_avg_frames": 15,
        "gemini_display_roi": [330, 0, 1220, 720],
        "gemini_color_auto_exposure": True,
        "gemini_color_exposure": None,
        "gemini_color_gain": None,
        "gemini_color_width": 1280,
        "gemini_color_height": 720,
        "gemini_color_format": "RGB",
        "gemini_color_fps": 30,
        "gemini_min_diameter_mm": 18.0,
        "gemini_max_diameter_mm": 32.0,
        "coin_class_diameters_mm": {
            "1yuan": 20.0,
            "5yuan": 22.0,
            "10yuan": 26.0,
            "50yuan": 28.0,
        },
        "robot_output_enabled": True,
        "robot_tabletop_homography_path": str(ROBOT_TABLETOP_H_FILE),
        "robot_h_path": str(ROBOT_DATA_DIR / "H.npy"),
        "robot_extrinsics_path": str(ROBOT_DATA_DIR / "camera_extrinsics.npz"),
        "robot_table_z_mm": -160.0,
        "robot_target_offset_x_mm": 0.0,
        "robot_target_offset_y_mm": 0.0,
        "quality_fallback_width": 640,
        "quality_fallback_height": 480,
        "quality_fallback_fps": 15,
    }
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(defaults, indent=2, ensure_ascii=False), encoding="utf-8")
        return defaults
    data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return defaults | data


def load_calib(cfg=None):
    fixed = (cfg or {}).get("coin_class_diameters_mm")
    if fixed:
        print("[分類] 使用設定檔固定直徑")
        for label in CLASSES:
            if label in fixed:
                print(f"  {LABEL_NAME.get(label, label)}: {float(fixed[label]):.2f}mm")
        return {label: {"mean": float(mm), "samples": []} for label, mm in fixed.items()}

    if CALIB_FILE.exists():
        data = json.loads(CALIB_FILE.read_text(encoding="utf-8"))
        print(f"[校準] 載入 {CALIB_FILE}")
        for k, v in data.items():
            print(f"  {k}: 直徑 {v['mean']:.2f}mm ({len(v.get('samples', []))} samples)")
        return data
    print(f"[校準] 找不到 {CALIB_FILE}，分類會顯示 ?")
    return {}


def load_quality_to_gemini_homography():
    if not HOMOGRAPHY_FILE.exists():
        print(f"[對位] 找不到 {HOMOGRAPHY_FILE.name}，Quality 偵測結果暫時不讀 Gemini depth")
        return None
    data = json.loads(HOMOGRAPHY_FILE.read_text(encoding="utf-8"))
    h_mat = np.array(data["quality_to_gemini_homography"], dtype=np.float64)
    print(f"[對位] 載入 {HOMOGRAPHY_FILE.name}  mean_error={data.get('mean_error_px', '?')}px")
    return h_mat


def load_stereo_extrinsics():
    if not STEREO_EXTRINSICS_FILE.exists():
        print(f"[外參] 找不到 {STEREO_EXTRINSICS_FILE.name}，世界座標先使用 Gemini 相機座標")
        return None
    data = json.loads(STEREO_EXTRINSICS_FILE.read_text(encoding="utf-8"))
    r = np.array(data["R_gemini_to_quality"], dtype=np.float64)
    t = np.array(data["T_gemini_to_quality_mm"], dtype=np.float64).reshape(3, 1)
    print(f"[外參] 載入 {STEREO_EXTRINSICS_FILE.name}  rms={data.get('rms_px', '?')}px")
    return {"R_gemini_to_quality": r, "T_gemini_to_quality_mm": t, "raw": data}


def load_robot_calibration(cfg):
    if not cfg.get("robot_output_enabled", True):
        return None
    tabletop_path = Path(cfg.get("robot_tabletop_homography_path", ROBOT_TABLETOP_H_FILE))
    if tabletop_path.exists():
        try:
            data = json.loads(tabletop_path.read_text(encoding="utf-8"))
            h_mat = np.array(data["gemini_to_robot_homography"], dtype=np.float64)
            table_z = float(data.get("robot_table_z_mm", cfg.get("robot_table_z_mm", -160.0)))
            print(f"[手臂座標] 載入桌面 H={tabletop_path.name}  mean_error={data.get('mean_error_mm', '?')}mm")
            return {
                "H": h_mat,
                "rvec": None,
                "tvec": None,
                "table_z_mm": table_z,
                "source": "tabletop_homography",
            }
        except Exception as e:
            print(f"[手臂座標] 桌面 H 載入失敗，改用舊 H.npy：{e}")

    h_path = Path(cfg.get("robot_h_path", ROBOT_DATA_DIR / "H.npy"))
    if not h_path.exists():
        print(f"[手臂座標] 找不到 {h_path}，暫時不輸出 MG400 座標")
        return None

    h_mat = np.load(str(h_path)).astype(np.float64)
    rvec = None
    tvec = None
    ext_path = Path(cfg.get("robot_extrinsics_path", ROBOT_DATA_DIR / "camera_extrinsics.npz"))
    if ext_path.exists():
        try:
            data = np.load(str(ext_path))
            rvec = data.get("rvec")
            tvec = data.get("tvec")
        except Exception as e:
            print(f"[手臂座標] 外參載入失敗，Z 使用固定桌面高度：{e}")
    print(f"[手臂座標] 載入 H={h_path.name}  Z={'PnP' if rvec is not None and tvec is not None else 'table'}")
    return {
        "H": h_mat,
        "rvec": rvec,
        "tvec": tvec,
        "table_z_mm": float(cfg.get("robot_table_z_mm", -160.0)),
        "source": "aruco_h",
    }


def map_quality_to_gemini_xy(homography, x, y):
    if homography is None:
        return None
    src = np.array([[[float(x), float(y)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography).reshape(2)
    return float(dst[0]), float(dst[1])


def sample_depth_mm(depth_mm, x, y, radius=7):
    if depth_mm is None:
        return None
    h, w = depth_mm.shape[:2]
    gx = int(round(x))
    gy = int(round(y))
    if gx < 0 or gx >= w or gy < 0 or gy >= h:
        return None
    x1 = max(0, gx - radius)
    x2 = min(w, gx + radius + 1)
    y1 = max(0, gy - radius)
    y2 = min(h, gy + radius + 1)
    patch = depth_mm[y1:y2, x1:x2]
    valid = patch[patch > 0]
    if valid.size < 3:
        return None
    return float(np.median(valid))


def deproject_gemini_pixel(intrinsics, x, y, z_mm):
    if z_mm is None or z_mm <= 0:
        return None
    x_mm = (float(x) - float(intrinsics.cx)) * float(z_mm) / float(intrinsics.fx)
    y_mm = (float(y) - float(intrinsics.cy)) * float(z_mm) / float(intrinsics.fy)
    return np.array([x_mm, y_mm, float(z_mm)], dtype=np.float64)


def estimate_quality_diameter_mm(item, homography, intrinsics):
    if homography is None or intrinsics is None or item.get("depth_z_mm") is None:
        return None
    if item["depth_z_mm"] <= 0 or "axes" not in item:
        return None

    cx = float(item["cx"])
    cy = float(item["cy"])
    axis_a, axis_b = [float(v) for v in item["axes"]]
    angle = np.deg2rad(float(item.get("angle", 0.0)))
    axes = [
        (np.cos(angle), np.sin(angle), axis_a / 2.0),
        (-np.sin(angle), np.cos(angle), axis_b / 2.0),
    ]
    diameters = []
    for dx, dy, radius_px in axes:
        p1 = map_quality_to_gemini_xy(homography, cx - dx * radius_px, cy - dy * radius_px)
        p2 = map_quality_to_gemini_xy(homography, cx + dx * radius_px, cy + dy * radius_px)
        if p1 is None or p2 is None:
            continue
        z_mm = float(item["depth_z_mm"])
        xyz1 = deproject_gemini_pixel(intrinsics, p1[0], p1[1], z_mm)
        xyz2 = deproject_gemini_pixel(intrinsics, p2[0], p2[1], z_mm)
        if xyz1 is not None and xyz2 is not None:
            diameters.append(float(np.linalg.norm(xyz2 - xyz1)))
    if not diameters:
        return None
    return float(np.mean(diameters))


def gemini_to_quality_xyz(point_gemini, stereo_extrinsics):
    if point_gemini is None or stereo_extrinsics is None:
        return None
    p = point_gemini.reshape(3, 1)
    q = stereo_extrinsics["R_gemini_to_quality"] @ p + stereo_extrinsics["T_gemini_to_quality_mm"]
    return q.reshape(3)


def gemini_pixel_to_robot(u, v, depth_mm, intrinsics, robot_calib):
    if robot_calib is None or u is None or v is None:
        return None
    pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
    world_xy = cv2.perspectiveTransform(pt, robot_calib["H"])[0, 0]

    z_robot = float(robot_calib["table_z_mm"])
    rvec = robot_calib.get("rvec")
    tvec = robot_calib.get("tvec")
    if depth_mm is not None and depth_mm > 0 and rvec is not None and tvec is not None:
        try:
            r_mat, _ = cv2.Rodrigues(rvec)
            p_cam = np.linalg.inv(intrinsics.K) @ np.array([float(u), float(v), 1.0], dtype=np.float64) * float(depth_mm)
            p_world = r_mat.T @ (p_cam - tvec.flatten())
            z_robot = float(p_world[2])
        except Exception:
            z_robot = float(robot_calib["table_z_mm"])
    return np.array([float(world_xy[0]), float(world_xy[1]), z_robot], dtype=np.float64)


def attach_robot_coords_to_quality_ellipses(q_ellipses, intrinsics, robot_calib):
    out = []
    for e in q_ellipses or []:
        item = dict(e)
        robot_xyz = gemini_pixel_to_robot(
            item.get("gemini_x"),
            item.get("gemini_y"),
            item.get("depth_z_mm"),
            intrinsics,
            robot_calib,
        )
        if robot_xyz is not None:
            item["robot_xyz_mm"] = robot_xyz.tolist()
        out.append(item)
    return out


def attach_gemini_depth_to_quality_ellipses(ellipses, homography, depth_mm, intrinsics=None, stereo_extrinsics=None):
    out = []
    for e in ellipses or []:
        item = dict(e)
        mapped = map_quality_to_gemini_xy(homography, item["cx"], item["cy"])
        if mapped is not None:
            gx, gy = mapped
            item["gemini_x"] = gx
            item["gemini_y"] = gy
            item["depth_z_mm"] = sample_depth_mm(depth_mm, gx, gy)
            if intrinsics is not None:
                gemini_xyz = deproject_gemini_pixel(intrinsics, gx, gy, item["depth_z_mm"])
                if gemini_xyz is not None:
                    item["gemini_xyz_mm"] = gemini_xyz.tolist()
                    item["world_xyz_mm"] = gemini_xyz.tolist()
                    quality_xyz = gemini_to_quality_xyz(gemini_xyz, stereo_extrinsics)
                    if quality_xyz is not None:
                        item["quality_xyz_mm"] = quality_xyz.tolist()
                quality_diam = estimate_quality_diameter_mm(item, homography, intrinsics)
                if quality_diam is not None:
                    item["quality_diameter_mm"] = quality_diam
        out.append(item)
    return out


def classify_by_size(diameter_mm, calib, tol=CALIB_TOL_MM, min_margin=0.0):
    if not calib:
        return "?", None
    ranked = []
    for label, v in calib.items():
        diff = abs(diameter_mm - float(v["mean"]))
        ranked.append((diff, label))
    ranked.sort(key=lambda item: item[0])
    best_diff, best_label = ranked[0]
    if len(ranked) > 1 and ranked[1][0] - best_diff < float(min_margin):
        return "?", best_diff
    return (best_label, best_diff) if best_diff <= tol else ("?", best_diff)


def classify_quality_ellipses(q_ellipses, calib, cfg=None):
    out = []
    margin = float((cfg or {}).get("quality_min_class_margin_mm", 0.45))
    for e in q_ellipses or []:
        item = dict(e)
        diam = item.get("quality_diameter_mm")
        if diam is not None:
            label, diff = classify_by_size(float(diam), calib, min_margin=margin)
            item["predicted_class"] = label
            item["class_diff_mm"] = diff
        out.append(item)
    return out


def smooth_quality_measurements(current, prev, cfg):
    history = max(1, int(cfg.get("quality_diameter_history", 5)))
    max_dist = 45.0
    matched = set()
    out = []
    for e in current or []:
        item = dict(e)
        ex = item.get("gemini_x", item["cx"])
        ey = item.get("gemini_y", item["cy"])
        best = None
        best_i = -1
        best_d = float("inf")
        for i, old in enumerate(prev or []):
            ox = old.get("gemini_x", old["cx"])
            oy = old.get("gemini_y", old["cy"])
            dist = float(np.hypot(float(ex) - float(ox), float(ey) - float(oy)))
            if dist < best_d:
                best = old
                best_i = i
                best_d = dist
        if best is not None and best_d < max_dist and best_i not in matched:
            diam_buf = best.get("quality_diam_buf", deque(maxlen=history))
            z_buf = best.get("depth_buf", deque(maxlen=history))
            matched.add(best_i)
        else:
            diam_buf = deque(maxlen=history)
            z_buf = deque(maxlen=history)

        if item.get("quality_diameter_mm") is not None:
            diam_buf.append(float(item["quality_diameter_mm"]))
            item["quality_diameter_raw_mm"] = float(item["quality_diameter_mm"])
            item["quality_diameter_mm"] = float(np.median(diam_buf))
        if item.get("depth_z_mm") is not None:
            z_buf.append(float(item["depth_z_mm"]))
            item["depth_z_mm"] = float(np.median(z_buf))
        item["quality_diam_buf"] = diam_buf
        item["depth_buf"] = z_buf
        item["smooth_count"] = len(diam_buf)
        out.append(item)
    return out


def is_reasonable_gemini_coin(meas, cfg):
    diam = float(meas.get("diameter_mm", 0.0))
    min_d = float(cfg.get("gemini_min_diameter_mm", 18.0))
    max_d = float(cfg.get("gemini_max_diameter_mm", 32.0))
    if diam < min_d or diam > max_d:
        return False
    cx = int(meas.get("cx", 0))
    cy = int(meas.get("cy", 0))
    r = int(round(float(meas.get("r_px", 0))))
    mask = meas.get("mask")
    if mask is not None:
        h, w = mask.shape[:2]
        margin = max(r // 2, 8)
        if cx < margin or cy < margin or cx >= w - margin or cy >= h - margin:
            return False
    return True


_sam3 = None
_sam3_model_path = None


def resolve_sam3_model(cfg=None):
    """Find SAM3 model without depending on one hard-coded desktop location."""
    candidates = []
    for key in ("SAM3_MODEL", "SAM3_MODEL_PATH"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    if cfg and cfg.get("sam3_model_path"):
        candidates.append(Path(str(cfg["sam3_model_path"])))
    candidates.extend([
        HERE / "sam3.pt",
        HERE.parent / "sam3.pt",
        Path.home() / "Desktop" / "sam3.pt",
        Path.home() / "Desktop" / "專題" / "sam3.pt",
    ])
    seen = set()
    checked = []
    for path in candidates:
        path = path.expanduser()
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        checked.append(str(path))
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "找不到 SAM3 模型 sam3.pt。\n"
        "請把模型放到 C:\\Users\\user\\Desktop\\sam3.pt，"
        "或在 dual_camera_config.json 設定 sam3_model_path。\n"
        "已檢查：\n" + "\n".join(f"  - {p}" for p in checked)
    )


def get_sam3():
    global _sam3, _sam3_model_path
    if _sam3 is None:
        _sam3_model_path = resolve_sam3_model(load_config())
        print(f"載入 SAM3: {_sam3_model_path}")
        from ultralytics.models.sam import SAM3SemanticPredictor

        _sam3 = SAM3SemanticPredictor(
            overrides=dict(
                model=_sam3_model_path,
                task="segment",
                mode="predict",
                conf=0.1,
                save=False,
                verbose=False,
            )
        )
        print("SAM3 OK")
    return _sam3


def _preprocess_for_sam(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    return cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)


def _circularity(mask_bool):
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    return 0.0 if peri == 0 else float(4 * np.pi * area / (peri**2))


def _clean_mask(mask_bool):
    m = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_OPEN, _ERODE_KERNEL, iterations=2)
    return m.astype(bool)


def detect_masks(color_bgr):
    pred = get_sam3()
    enhanced = _preprocess_for_sam(color_bgr)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp = f.name
    try:
        cv2.imwrite(tmp, enhanced)
        pred.set_image(tmp)
        res = pred(text=["coin", "硬幣", "round metal coin", "circle", "disc"])
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    raw_masks = []
    if res and res[0].masks is not None:
        raw = res[0].masks.data.cpu().numpy()
        h, w = color_bgr.shape[:2]
        for i in np.argsort([raw[k].sum() for k in range(len(raw))]):
            m = cv2.resize(raw[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.sum() >= 50:
                raw_masks.append(m)

    masks = []
    for raw in raw_masks:
        m = _clean_mask(raw)
        if m.sum() < 50:
            continue
        circ = _circularity(m)
        if circ < CIRCULARITY_MIN:
            continue
        duplicate = any(np.logical_and(m, old).sum() / np.logical_or(m, old).sum() > 0.5 for old in masks)
        if not duplicate:
            masks.append(m)
    return masks


def _fit_circle_ls(pts):
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    a = np.column_stack([x, y, np.ones(len(x))])
    b = x**2 + y**2
    res, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    cx = res[0] / 2.0
    cy = res[1] / 2.0
    r = float(np.sqrt(max(res[2] + cx**2 + cy**2, 0.0)))
    return cx, cy, r


def measure_coin(mask, depth_mm, intrinsics):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest[:, 0, :]
    if len(pts) < 6:
        return None

    fcx, fcy, r_px = _fit_circle_ls(pts)
    cx = int(round(fcx))
    cy = int(round(fcy))
    if r_px < 3:
        return None

    h, w = depth_mm.shape
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    bx = np.clip((cx + r_px * np.cos(angles)).astype(int), 0, w - 1)
    by = np.clip((cy + r_px * np.sin(angles)).astype(int), 0, h - 1)
    valid_bd = depth_mm[by, bx]
    valid_bd = valid_bd[valid_bd > 0]
    if valid_bd.size < 5:
        patch = depth_mm[max(0, cy - 8) : min(h, cy + 8), max(0, cx - 8) : min(w, cx + 8)]
        valid_bd = patch[patch > 0]
    if valid_bd.size < 3:
        return None

    z = float(np.median(valid_bd))
    f_avg = (intrinsics.fx + intrinsics.fy) / 2
    diam = 2 * r_px * z / f_avg
    return {"mask": mask, "cx": cx, "cy": cy, "diameter_mm": float(diam), "z_mm": z, "r_px": float(r_px)}


def open_quality_camera(cfg):
    index = int(cfg["quality_camera_index"])
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[畫質相機] index={index} 開啟失敗")
        return None, "open_failed"

    def apply_props(width, height, fps):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, int(fps))

    apply_props(cfg["quality_width"], cfg["quality_height"], cfg["quality_fps"])
    ok, frame = cap.read()
    if ok and frame is not None:
        print(f"[畫質相機] index={index} OK  frame={frame.shape[1]}x{frame.shape[0]}")
        return cap, "ok"

    print("[畫質相機] 高解析讀取失敗，嘗試 fallback")
    apply_props(cfg["quality_fallback_width"], cfg["quality_fallback_height"], cfg["quality_fallback_fps"])
    ok, frame = cap.read()
    if ok and frame is not None:
        print(f"[畫質相機] fallback OK  frame={frame.shape[1]}x{frame.shape[0]}")
        return cap, "fallback"

    cap.release()
    print("[畫質相機] fallback 仍失敗，Gemini 將繼續單獨運作")
    return None, "read_failed"


def normalize_roi(roi, frame_shape):
    h, w = frame_shape[:2]
    if not roi:
        return 0, 0, w, h
    x1, y1, x2, y2 = [int(v) for v in roi]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def crop_display_roi(frame, roi):
    x1, y1, x2, y2 = normalize_roi(roi, frame.shape)
    return frame[y1:y2, x1:x2]


def map_quality_roi_to_gemini_polygon(cfg, homography):
    roi = cfg.get("quality_roi")
    if homography is None or not roi:
        return None
    x1, y1, x2, y2 = [float(v) for v in roi]
    pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2).astype(np.int32)


def blur_score(frame, roi):
    x1, y1, x2, y2 = normalize_roi(roi, frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_quality_ellipses(frame, cfg):
    method = str(cfg.get("quality_detection_method", "sam3_ellipse")).lower()
    if method == "opencv_ellipse":
        return detect_quality_ellipses_opencv(frame, cfg)
    return detect_quality_ellipses_sam3(frame, cfg)


def detect_quality_ellipses_sam3(frame, cfg):
    """Use SAM3 masks from the quality camera, then fit ellipses for display only."""
    if frame is None:
        return []
    x1, y1, x2, y2 = normalize_roi(cfg.get("quality_roi"), frame.shape)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return []

    pred = get_sam3()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp = f.name
    try:
        cv2.imwrite(tmp, _preprocess_for_sam(crop))
        pred.set_image(tmp)
        res = pred(text=["coin", "硬幣", "round metal coin", "metal disc", "elliptical coin"])
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    if not (res and res[0].masks is not None):
        return []

    raw = res[0].masks.data.cpu().numpy()
    h, w = crop.shape[:2]
    ellipses = []
    for i in np.argsort([raw[k].sum() for k in range(len(raw))]):
        mask = cv2.resize(raw[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask = _clean_mask(mask)
        ellipse = fit_quality_mask_ellipse(mask, cfg, x1, y1)
        if ellipse is not None:
            ellipses.append(ellipse)

    ellipses.sort(key=lambda e: e["area"], reverse=True)
    return dedupe_quality_ellipses(ellipses)


def fit_quality_mask_ellipse(mask, cfg, offset_x=0, offset_y=0):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 8:
        return None
    area = float(cv2.contourArea(cnt))
    min_area = float(cfg.get("quality_min_area", 350))
    max_area = float(cfg.get("quality_max_area", 20000))
    if area < min_area or area > max_area:
        return None
    try:
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(cnt)
    except cv2.error:
        return None
    major = max(axis_a, axis_b)
    minor = min(axis_a, axis_b)
    if minor < float(cfg.get("quality_min_axis", 12)) or major > float(cfg.get("quality_max_axis", 180)):
        return None
    axis_ratio = minor / max(major, 1.0)
    if axis_ratio < 0.12:
        return None
    ellipse_area = np.pi * (axis_a / 2.0) * (axis_b / 2.0)
    fill_ratio = area / max(ellipse_area, 1.0)
    if fill_ratio < 0.15 or fill_ratio > 1.45:
        return None
    return {
        "cx": float(cx + offset_x),
        "cy": float(cy + offset_y),
        "axes": (float(axis_a), float(axis_b)),
        "angle": float(angle),
        "area": area,
        "axis_ratio": float(axis_ratio),
        "fill_ratio": float(fill_ratio),
        "source": "sam3",
    }


def detect_quality_ellipses_opencv(frame, cfg):
    """Detect coin-like ellipses with OpenCV edges for display only."""
    if frame is None:
        return []
    x1, y1, x2, y2 = normalize_roi(cfg.get("quality_roi"), frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 35, 110)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _ERODE_KERNEL, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    min_area = float(cfg.get("quality_min_area", 350))
    max_area = float(cfg.get("quality_max_area", 20000))
    min_axis = float(cfg.get("quality_min_axis", 12))
    max_axis = float(cfg.get("quality_max_axis", 180))
    for cnt in contours:
        if len(cnt) < 8:
            continue
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        try:
            (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        major = max(axis_a, axis_b)
        minor = min(axis_a, axis_b)
        if minor < min_axis or major > max_axis:
            continue
        axis_ratio = minor / max(major, 1.0)
        if axis_ratio < 0.18:
            continue
        ellipse_area = np.pi * (axis_a / 2.0) * (axis_b / 2.0)
        fill_ratio = area / max(ellipse_area, 1.0)
        if fill_ratio < 0.18 or fill_ratio > 1.35:
            continue
        ellipses.append({
            "cx": float(cx + x1),
            "cy": float(cy + y1),
            "axes": (float(axis_a), float(axis_b)),
            "angle": float(angle),
            "area": area,
            "axis_ratio": float(axis_ratio),
            "fill_ratio": float(fill_ratio),
            "source": "opencv",
        })

    ellipses.sort(key=lambda e: e["area"], reverse=True)
    return dedupe_quality_ellipses(ellipses)


def dedupe_quality_ellipses(ellipses):
    deduped = []
    for e in ellipses:
        if any(np.hypot(e["cx"] - old["cx"], e["cy"] - old["cy"]) < 0.35 * max(e["axes"]) for old in deduped):
            continue
        deduped.append(e)
    return deduped[:20]


def fit_to_box(img, width, height):
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    out = np.zeros((height, width, 3), dtype=np.uint8)
    resized = cv2.resize(img, (nw, nh))
    ox, oy = (width - nw) // 2, (height - nh) // 2
    out[oy : oy + nh, ox : ox + nw] = resized
    return out, scale, ox, oy


def draw_gemini(color_bgr, coins, detecting, snapshot_state, intrinsics, quality_ellipses=None, display_roi=None, quality_roi_poly=None):
    out = color_bgr.copy()
    f_avg = (intrinsics.fx + intrinsics.fy) / 2
    for idx, coin in enumerate(coins):
        label = coin["predicted_class"]
        color = COIN_COLOR.get(label, COIN_COLOR["?"])
        mask = coin["mask"]
        cx, cy = coin["cx"], coin["cy"]
        diam, z = coin["diameter_mm"], coin["z_mm"]
        r_px = max(int(diam * f_avg / (2 * z)) if z > 0 else int(coin["r_px"]), 4)

        overlay = out.copy()
        overlay[mask] = (np.array(color) * 0.35 + out[mask] * 0.65).astype(np.uint8)
        out = overlay
        cv2.circle(out, (cx, cy), r_px, color, 2)
        cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1)
        diff = coin["class_diff_mm"]
        diff_str = f" +/-{diff:.1f}" if diff is not None else ""
        cv2.putText(out, f"#{idx+1} {LABEL_NAME.get(label,label)}{diff_str}", (cx + 8, max(24, cy - r_px - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(out, f"d={diam:.2f}mm Z={z:.0f}", (cx + 8, min(out.shape[0] - 8, cy + r_px + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    for idx, e in enumerate(quality_ellipses or [], 1):
        if "gemini_x" not in e or "gemini_y" not in e:
            continue
        gx = int(round(e["gemini_x"]))
        gy = int(round(e["gemini_y"]))
        if gx < 0 or gy < 0 or gx >= out.shape[1] or gy >= out.shape[0]:
            continue
        z = e.get("depth_z_mm")
        cv2.drawMarker(out, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        label = f"Q{idx}"
        if z is not None:
            label += f" Z={z:.0f}mm"
        cv2.putText(out, label, (gx + 10, max(24, gy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA)

    if quality_roi_poly is not None and len(quality_roi_poly) >= 4:
        cv2.polylines(out, [quality_roi_poly.reshape(-1, 1, 2)], True, (0, 180, 255), 2, cv2.LINE_AA)
        p = quality_roi_poly[0]
        cv2.putText(out, "Detection ROI", (int(p[0]) + 8, int(p[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

    banner = (0, 60, 20) if snapshot_state == "snapshot" else (25, 25, 25)
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), banner, -1)
    status = "SAM3 detecting..." if detecting else "Gemini depth diameter  |  SPACE=snapshot  S=log  Q=quit"
    if snapshot_state == "accumulating":
        status = "Snapshot accumulating depth..."
    elif snapshot_state == "snapshot":
        status = "Snapshot mode  |  SPACE=back to live"
    cv2.putText(out, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 230, 120), 2, cv2.LINE_AA)
    if display_roi:
        out = crop_display_roi(out, display_roi)
    disp, _, _, _ = fit_to_box(out, GEMINI_DISP_W, DISP_H)
    return disp


def is_quality_pick_valid(e, cfg):
    xyz = e.get("robot_xyz_mm")
    if xyz is None:
        return False
    x_mm = float(xyz[0]) + float(cfg.get("robot_target_offset_x_mm", 0.0))
    y_mm = float(xyz[1]) + float(cfg.get("robot_target_offset_y_mm", 0.0))
    return (
        e.get("predicted_class", "?") != "?"
        and e.get("depth_z_mm") is not None
        and PICK_X_MIN <= x_mm <= PICK_X_MAX
        and PICK_Y_MIN <= y_mm <= PICK_Y_MAX
    )


def draw_quality(frame, cfg, score, cam_status, ellipses=None, detecting=False):
    if frame is None:
        out = np.zeros((DISP_H, QUALITY_DISP_W, 3), dtype=np.uint8)
        cv2.putText(out, f"Quality camera: {cam_status}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
        return out

    out = frame.copy()
    roi = normalize_roi(cfg.get("quality_roi"), out.shape)
    x1, y1, x2, y2 = roi
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
    for idx, e in enumerate(ellipses or [], 1):
        center = (int(round(e["cx"])), int(round(e["cy"])))
        axes = (max(1, int(round(e["axes"][0] / 2))), max(1, int(round(e["axes"][1] / 2))))
        valid = is_quality_pick_valid(e, cfg)
        color = (0, 255, 120) if valid else (0, 210, 255)
        status_text = "OK" if valid else "CHECK"
        cv2.ellipse(out, center, axes, e["angle"], 0, 360, color, 2, cv2.LINE_AA)
        cv2.circle(out, center, 3, (255, 255, 255), -1)
        depth_text = ""
        if e.get("depth_z_mm") is not None:
            depth_text = f" Z={e['depth_z_mm']:.0f}mm"
        diam_text = ""
        if e.get("quality_diameter_mm") is not None:
            diam_text = f" D={e['quality_diameter_mm']:.2f}mm"
        class_text = ""
        if e.get("predicted_class") is not None:
            class_text = f" {LABEL_NAME.get(e['predicted_class'], e['predicted_class'])}"
        xyz_text = ""
        if e.get("world_xyz_mm") is not None:
            x, y, z = e["world_xyz_mm"]
            xyz_text = f" XYZ=({x:.0f},{y:.0f},{z:.0f})"
        cv2.putText(out, f"Q{idx} {status_text}{class_text} {e.get('source','')} r={e['axis_ratio']:.2f}{depth_text}{diam_text}", (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if xyz_text:
            cv2.putText(out, xyz_text, (center[0] + 8, center[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (25, 25, 25), -1)
    method = str(cfg.get("quality_detection_method", "sam3_ellipse"))
    status = "detecting..." if detecting else f"{method}"
    cv2.putText(out, f"Quality cam  ellipses={len(ellipses or [])}  blur={score:.1f}  {status}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 220, 255), 2, cv2.LINE_AA)
    disp, _, _, _ = fit_to_box(out, QUALITY_DISP_W, DISP_H)
    return disp


def crop_around(frame, cx, cy, radius, out_size=260):
    if frame is None:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    radius = max(int(round(radius)), 24)
    cx = int(round(cx))
    cy = int(round(cy))
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC)


def annotate_quality_crop(crop, e, scale):
    center = (crop.shape[1] // 2, crop.shape[0] // 2)
    axes = (
        max(2, int(round((float(e["axes"][0]) / 2.0) * scale))),
        max(2, int(round((float(e["axes"][1]) / 2.0) * scale))),
    )
    cv2.ellipse(crop, center, axes, float(e["angle"]), 0, 360, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.drawMarker(crop, center, (0, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)


def annotate_gemini_crop(crop, radius_px):
    center = (crop.shape[1] // 2, crop.shape[0] // 2)
    cv2.circle(crop, center, max(3, int(round(radius_px))), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(crop, center, (0, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)


def save_debug_crops(color, q_frame, coins, q_ellipses, intrinsics, ts):
    rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, e in enumerate(q_ellipses or [], 1):
        q_radius = max(float(e["axes"][0]), float(e["axes"][1])) * 0.75 + 24
        q_crop = crop_around(q_frame, e["cx"], e["cy"], q_radius)
        annotate_quality_crop(q_crop, e, q_crop.shape[0] / max(2.0 * q_radius, 1.0))

        gx = e.get("gemini_x")
        gy = e.get("gemini_y")
        z = e.get("depth_z_mm")
        diam = e.get("quality_diameter_mm")
        g_radius = 44
        if diam is not None and z is not None and z > 0:
            f_avg = (intrinsics.fx + intrinsics.fy) / 2.0
            g_radius = max(18, int(float(diam) * f_avg / (2.0 * float(z))))
        g_crop = crop_around(color, gx if gx is not None else 0, gy if gy is not None else 0, g_radius + 32)
        annotate_gemini_crop(g_crop, g_radius * g_crop.shape[0] / max(2.0 * (g_radius + 32), 1.0))

        label = np.full((38, q_crop.shape[1] + g_crop.shape[1], 3), (20, 20, 20), dtype=np.uint8)
        cls = LABEL_NAME.get(e.get("predicted_class"), e.get("predicted_class", "?"))
        text = f"Q{idx} {cls} diameter={diam:.2f}mm  Z={z:.0f}mm" if diam is not None and z is not None else f"Q{idx} diameter=?"
        cv2.putText(label, text, (8, 26), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        row = np.vstack([label, np.hstack([q_crop, g_crop])])
        cv2.putText(row, "Quality edge", (8, 62), font, 0.52, (0, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(row, "Gemini mapped edge", (q_crop.shape[1] + 8, 62), font, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)

    if rows:
        sheet = np.vstack(rows)
        out_path = OUT_DIR / f"coin_edge_crops_{ts}.jpg"
        cv2.imwrite(str(out_path), sheet)
        print(f"[debug] 邊緣放大圖 {out_path}")

    if coins:
        g_rows = []
        for idx, coin in enumerate(coins, 1):
            radius = max(float(coin.get("r_px", 30)) + 24, 40)
            crop = crop_around(color, coin["cx"], coin["cy"], radius)
            annotate_gemini_crop(crop, float(coin.get("r_px", 20)) * crop.shape[0] / max(2.0 * radius, 1.0))
            cv2.putText(crop, f"G{idx} {coin['diameter_mm']:.2f}mm", (8, 24), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            g_rows.append(crop)
        out_path = OUT_DIR / f"gemini_coin_crops_{ts}.jpg"
        cv2.imwrite(str(out_path), np.hstack(g_rows))
        print(f"[debug] Gemini 自己偵測放大圖 {out_path}")


def summarize_quality_coins(q_ellipses, cfg):
    counts = {label: 0 for label in CLASSES}
    total_value = 0
    reminders = []
    need_history = max(1, int(cfg.get("quality_diameter_history", 5)))
    for idx, e in enumerate(q_ellipses or [], 1):
        label = e.get("predicted_class", "?")
        if label in counts:
            counts[label] += 1
            total_value += COIN_VALUE[label]
        if e.get("depth_z_mm") is None:
            reminders.append(f"Q{idx}: no depth")
        elif e.get("quality_diameter_mm") is None:
            reminders.append(f"Q{idx}: no diameter")
        elif label == "?":
            reminders.append(f"Q{idx}: unclear {e['quality_diameter_mm']:.2f}mm")
        elif int(e.get("smooth_count", need_history)) < min(3, need_history):
            reminders.append(f"Q{idx}: wait stable")
    return counts, total_value, reminders


def write_robot_targets(q_ellipses, counts, total_value, cfg=None):
    cfg = cfg or {}
    offset_x = float(cfg.get("robot_target_offset_x_mm", 0.0))
    offset_y = float(cfg.get("robot_target_offset_y_mm", 0.0))
    targets = []
    for idx, e in enumerate(q_ellipses or [], 1):
        xyz = e.get("robot_xyz_mm")
        if xyz is None:
            continue
        x_mm = float(xyz[0]) + offset_x
        y_mm = float(xyz[1]) + offset_y
        valid_for_pick = is_quality_pick_valid(e, cfg)
        targets.append({
            "index": idx,
            "label": e.get("predicted_class", "?"),
            "label_name": LABEL_NAME.get(e.get("predicted_class", "?"), "?"),
            "value_nt": COIN_VALUE.get(e.get("predicted_class"), 0),
            "diameter_mm": None if e.get("quality_diameter_mm") is None else round(float(e["quality_diameter_mm"]), 3),
            "depth_z_mm": None if e.get("depth_z_mm") is None else round(float(e["depth_z_mm"]), 3),
            "quality_x_px": round(float(e["cx"]), 3),
            "quality_y_px": round(float(e["cy"]), 3),
            "gemini_x_px": None if e.get("gemini_x") is None else round(float(e["gemini_x"]), 3),
            "gemini_y_px": None if e.get("gemini_y") is None else round(float(e["gemini_y"]), 3),
            "robot_x_mm": round(x_mm, 3),
            "robot_y_mm": round(y_mm, 3),
            "robot_z_mm": round(float(xyz[2]), 3),
            "valid_for_pick": valid_for_pick,
        })
    payload = {
        "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "coordinate_frame": "MG400 robot base, mm",
        "counts": {LABEL_NAME.get(k, k): int(v) for k, v in counts.items()},
        "total_value_nt": int(total_value),
        "targets": targets,
    }
    try:
        ROBOT_TARGETS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[手臂座標] 輸出失敗：{e}")


def draw_panel(coins, q_score, cfg, q_status, snapshot_state, q_ellipses=None, homography_loaded=False):
    panel = np.full((DISP_H, PANEL_W, 3), (30, 30, 35), dtype=np.uint8)
    counts, total_value, reminders = summarize_quality_coins(q_ellipses, cfg)
    cv2.line(panel, (0, 0), (0, DISP_H), (80, 80, 90), 1)
    cv2.putText(panel, "Dual Camera", (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2)
    cv2.line(panel, (8, 50), (PANEL_W - 8, 50), (60, 60, 70), 1)
    cv2.putText(panel, "Quality = object detect", (14, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(panel, "Gemini = depth Z", (14, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(panel, f"QCam index: {cfg['quality_camera_index']}", (14, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(panel, f"QCam status: {q_status}", (14, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(panel, f"Blur: {q_score:.1f}", (14, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
    cv2.putText(panel, f"Ellipses: {len(q_ellipses or [])}", (14, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1)
    cv2.putText(panel, f"Mode: {snapshot_state}", (14, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 160), 1)
    cv2.putText(panel, f"Q->G map: {'OK' if homography_loaded else 'missing'}", (14, 262),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255) if homography_loaded else (90, 90, 255), 1)
    cv2.putText(panel, "World: Gemini XYZ mm", (14, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1)

    cv2.line(panel, (8, 306), (PANEL_W - 8, 306), (60, 60, 70), 1)
    cv2.putText(panel, "Quality Summary", (14, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, f"Total: {total_value} NT", (150, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    y = 362
    for label in CLASSES:
        color = COIN_COLOR.get(label, COIN_COLOR["?"])
        text = f"{LABEL_NAME.get(label,label)} x {counts[label]}"
        cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        y += 24

    unknown = sum(1 for e in q_ellipses or [] if e.get("predicted_class", "?") == "?")
    cv2.putText(panel, f"Detected: {len(q_ellipses or [])}  Unknown: {unknown}", (14, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1)
    y += 38

    robot_ready = sum(1 for e in q_ellipses or [] if e.get("robot_xyz_mm") is not None)
    cv2.putText(panel, f"MG400 targets: {robot_ready}", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255) if robot_ready else (120, 120, 120), 1)
    y += 22
    for idx, e in enumerate((q_ellipses or [])[:2], 1):
        xyz = e.get("robot_xyz_mm")
        if xyz is None:
            text = f"Q{idx}: robot coord ?"
        else:
            text = f"Q{idx}: X{xyz[0]:.0f} Y{xyz[1]:.0f} Z{xyz[2]:.0f}"
        cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 210, 210), 1)
        y += 20

    cv2.line(panel, (8, y), (PANEL_W - 8, y), (60, 60, 70), 1)
    y += 26
    cv2.putText(panel, "Reposition / Check", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 220, 255) if reminders else (0, 200, 100), 1)
    y += 24
    if not reminders:
        cv2.putText(panel, "All coins OK", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 120), 1)
    for msg in reminders[:2]:
        cv2.putText(panel, msg, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
        y += 24

    cv2.line(panel, (8, DISP_H - 112), (PANEL_W - 8, DISP_H - 112), (60, 60, 70), 1)
    cv2.putText(panel, "SPACE  snapshot/live", (14, DISP_H - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    cv2.putText(panel, "S      save csv log", (14, DISP_H - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    cv2.putText(panel, "Q/ESC  quit", (14, DISP_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    return panel


def append_log(coins, q_score, cfg, q_frame, q_ellipses=None):
    roi = normalize_roi(cfg.get("quality_roi"), q_frame.shape) if q_frame is not None else ("", "", "", "")
    write_header = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        if not coins and not q_ellipses:
            writer.writerow({
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "fusion_mode": "display_only",
                "quality_camera_index": cfg["quality_camera_index"],
                "quality_blur_score": round(q_score, 2),
                "quality_roi_x1": roi[0],
                "quality_roi_y1": roi[1],
                "quality_roi_x2": roi[2],
                "quality_roi_y2": roi[3],
                "valid": False,
            })
        for i, coin in enumerate(coins, 1):
            writer.writerow({
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "fusion_mode": "display_only",
                "quality_camera_index": cfg["quality_camera_index"],
                "quality_blur_score": round(q_score, 2),
                "quality_roi_x1": roi[0],
                "quality_roi_y1": roi[1],
                "quality_roi_x2": roi[2],
                "quality_roi_y2": roi[3],
                "coin_index": i,
                "center_x": coin["cx"],
                "center_y": coin["cy"],
                "diameter_mm": round(coin["diameter_mm"], 3),
                "depth_z_mm": round(coin["z_mm"], 3),
                "world_x_mm": "",
                "world_y_mm": "",
                "world_z_mm": "",
                "gemini_x_px": coin["cx"],
                "gemini_y_px": coin["cy"],
                "predicted_class": coin["predicted_class"],
                "class_diff_mm": "" if coin["class_diff_mm"] is None else round(coin["class_diff_mm"], 3),
                "valid": True,
            })
        for i, e in enumerate(q_ellipses or [], 1):
            xyz = e.get("world_xyz_mm") or ["", "", ""]
            writer.writerow({
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "fusion_mode": "quality_detect_gemini_world",
                "quality_camera_index": cfg["quality_camera_index"],
                "quality_blur_score": round(q_score, 2),
                "quality_roi_x1": roi[0],
                "quality_roi_y1": roi[1],
                "quality_roi_x2": roi[2],
                "quality_roi_y2": roi[3],
                "coin_index": i,
                "center_x": round(e["cx"], 3),
                "center_y": round(e["cy"], 3),
                "diameter_mm": "" if e.get("quality_diameter_mm") is None else round(e["quality_diameter_mm"], 3),
                "depth_z_mm": "" if e.get("depth_z_mm") is None else round(e["depth_z_mm"], 3),
                "world_x_mm": "" if xyz[0] == "" else round(float(xyz[0]), 3),
                "world_y_mm": "" if xyz[1] == "" else round(float(xyz[1]), 3),
                "world_z_mm": "" if xyz[2] == "" else round(float(xyz[2]), 3),
                "gemini_x_px": "" if e.get("gemini_x") is None else round(e["gemini_x"], 3),
                "gemini_y_px": "" if e.get("gemini_y") is None else round(e["gemini_y"], 3),
                "predicted_class": e.get("predicted_class", "quality_object"),
                "class_diff_mm": "" if e.get("class_diff_mm") is None else round(e["class_diff_mm"], 3),
                "valid": e.get("depth_z_mm") is not None,
            })


def main():
    if "--save-once" in sys.argv:
        save_once()
        return

    cfg = load_config()
    calib = load_calib(cfg)
    quality_to_gemini_h = load_quality_to_gemini_homography()
    stereo_extrinsics = load_stereo_extrinsics()
    robot_calib = load_robot_calibration(cfg)
    get_sam3()

    camera = Gemini2Camera(align_depth_to_color=True)
    qcap = None
    worker = None
    stop_flag = [False]

    try:
        set_gemini_stream_env(cfg)
        camera.open()
        apply_color_controls(camera, cfg)
        intr = camera.intrinsics
        qcap, q_status = open_quality_camera(cfg)

        frame_buf = [None]
        depth_buf = [None]
        data_lock = threading.Lock()
        buf_lock = threading.Lock()
        detect_event = threading.Event()
        detecting = [False]
        coins_state = [[]]
        smooth_state = [[]]
        depth_accum = []
        snap_depth_accum = []
        snap_color_frame = [None]
        snap_state = ["live"]
        snap_oneshot = [False]
        latest_quality = [None]
        latest_q_score = [0.0]
        latest_q_ellipses = [[]]
        quality_smooth_state = [[]]
        quality_detecting = [False]

        def detection_worker():
            while not stop_flag[0]:
                detect_event.wait(timeout=0.5)
                if stop_flag[0]:
                    break
                if not detect_event.is_set():
                    continue
                detect_event.clear()
                with buf_lock:
                    color_snap = None if frame_buf[0] is None else frame_buf[0].copy()
                    depth_snap = None if depth_buf[0] is None else depth_buf[0].copy()
                if color_snap is None or depth_snap is None:
                    continue

                detecting[0] = True
                is_snap = snap_oneshot[0]
                snap_oneshot[0] = False
                try:
                    runs = 3 if is_snap else 1
                    all_runs = []
                    for _ in range(runs):
                        run = []
                        for mask in detect_masks(color_snap):
                            meas = measure_coin(mask, depth_snap, intr)
                            if meas is not None:
                                run.append(meas)
                        all_runs.append(run)

                    raw = all_runs[0] if runs == 1 else median_snapshot_runs(all_runs)
                    smoothed = smooth_coin_measurements(raw, smooth_state[0], is_snap)
                    visible = []
                    min_count = 1 if is_snap else 10
                    for item in smoothed:
                        if item["count"] < min_count:
                            continue
                        if not is_reasonable_gemini_coin(item, cfg):
                            continue
                        label, diff = classify_by_size(
                            item["diameter_mm"],
                            calib,
                            min_margin=float(cfg.get("quality_min_class_margin_mm", 0.45)),
                        )
                        visible.append({
                            "mask": item["mask"],
                            "cx": item["cx"],
                            "cy": item["cy"],
                            "r_px": item["r_px"],
                            "diameter_mm": item["diameter_mm"],
                            "z_mm": item["z_mm"],
                            "predicted_class": label,
                            "class_diff_mm": diff,
                        })
                    with data_lock:
                        coins_state[0] = visible
                except Exception as e:
                    import traceback

                    print(f"[SAM3] 錯誤: {e}")
                    traceback.print_exc()
                finally:
                    detecting[0] = False
                    if not stop_flag[0] and snap_state[0] != "snapshot":
                        detect_event.set()

        def median_snapshot_runs(all_runs):
            if not all_runs:
                return []
            base = all_runs[0]
            out = []
            for coin in base:
                diams = [coin["diameter_mm"]]
                for other in all_runs[1:]:
                    best = None
                    best_d = float("inf")
                    for cand in other:
                        dist = np.hypot(coin["cx"] - cand["cx"], coin["cy"] - cand["cy"])
                        if dist < 30 and dist < best_d:
                            best = cand
                            best_d = dist
                    if best is not None:
                        diams.append(best["diameter_mm"])
                merged = dict(coin)
                merged["diameter_mm"] = float(np.median(diams))
                out.append(merged)
            return out

        def smooth_coin_measurements(raw, prev, is_snap):
            history = 20
            miss_max = 2
            new_smooth = []
            matched = set()
            for coin in raw:
                best = None
                best_i = -1
                best_d = float("inf")
                for i, old in enumerate(prev):
                    dist = np.hypot(coin["cx"] - old["cx"], coin["cy"] - old["cy"])
                    if dist < best_d:
                        best, best_i, best_d = old, i, dist
                if best is not None and best_d < 120:
                    diam_buf = best["diam_buf"]
                    z_buf = best["z_buf"]
                    count = best["count"] + 1
                    matched.add(best_i)
                else:
                    diam_buf = deque(maxlen=history)
                    z_buf = deque(maxlen=history)
                    count = 1
                diam_buf.append(coin["diameter_mm"])
                z_buf.append(coin["z_mm"])
                new_smooth.append({
                    "cx": coin["cx"],
                    "cy": coin["cy"],
                    "r_px": coin["r_px"],
                    "mask": coin["mask"],
                    "diam_buf": diam_buf,
                    "z_buf": z_buf,
                    "diameter_mm": float(np.median(diam_buf)),
                    "z_mm": float(np.median(z_buf)),
                    "count": count,
                    "miss": 0,
                })
            if not is_snap:
                for i, old in enumerate(prev):
                    if i in matched:
                        continue
                    old["miss"] = old.get("miss", 0) + 1
                    old["count"] = 0
                    if old["miss"] < miss_max:
                        new_smooth.append(old)
            smooth_state[0] = new_smooth
            return new_smooth

        worker = threading.Thread(target=detection_worker, daemon=True)
        worker.start()

        win = "Dual Camera Coin Detection"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

        for _ in range(5):
            camera.get_frames()

        frame_count = 0
        detect_event.set()
        while True:
            color, depth = camera.get_frames()
            if color is None or depth is None:
                continue
            frame_count += 1

            q_frame = None
            if qcap is not None:
                ok, q_read = qcap.read()
                if ok and q_read is not None:
                    q_frame = q_read
                    latest_quality[0] = q_frame
                    latest_q_score[0] = blur_score(q_frame, cfg.get("quality_roi"))
                    interval = max(1, int(cfg.get("quality_sam3_interval_frames", 30)))
                    if frame_count == 1 or frame_count % interval == 0:
                        quality_detecting[0] = True
                        try:
                            latest_q_ellipses[0] = detect_quality_ellipses(q_frame, cfg)
                        finally:
                            quality_detecting[0] = False
                else:
                    q_status = "read_failed"
            q_score = latest_q_score[0]
            q_ellipses = attach_gemini_depth_to_quality_ellipses(
                list(latest_q_ellipses[0]),
                quality_to_gemini_h,
                depth,
                intr,
                stereo_extrinsics,
            )
            q_ellipses = smooth_quality_measurements(q_ellipses, quality_smooth_state[0], cfg)
            quality_smooth_state[0] = q_ellipses
            q_ellipses = classify_quality_ellipses(q_ellipses, calib, cfg)
            q_ellipses = attach_robot_coords_to_quality_ellipses(q_ellipses, intr, robot_calib)
            counts, total_value, _ = summarize_quality_coins(q_ellipses, cfg)
            write_robot_targets(q_ellipses, counts, total_value, cfg)

            if snap_state[0] == "accumulating":
                snap_depth_accum.append(depth.astype(np.float32))
                n_acc = len(snap_depth_accum)
                if n_acc >= int(cfg["gemini_snapshot_frames"]):
                    avg_d = average_depth_stack(snap_depth_accum)
                    with buf_lock:
                        frame_buf[0] = snap_color_frame[0]
                        depth_buf[0] = avg_d
                    smooth_state[0] = []
                    snap_oneshot[0] = True
                    snap_state[0] = "snapshot"
                    detect_event.set()
                    print("[快照] 深度累積完成，開始偵測")
            elif snap_state[0] == "live":
                with buf_lock:
                    frame_buf[0] = color.copy()
                    depth_accum.append(depth.astype(np.float32))
                    max_n = int(cfg["gemini_depth_avg_frames"])
                    if len(depth_accum) > max_n:
                        depth_accum.pop(0)
                    depth_buf[0] = average_depth_stack(depth_accum)

            with data_lock:
                coins = list(coins_state[0])

            gemini_disp = draw_gemini(
                snap_color_frame[0] if snap_state[0] == "snapshot" and snap_color_frame[0] is not None else color,
                coins,
                detecting[0],
                snap_state[0],
                intr,
                q_ellipses,
                cfg.get("gemini_display_roi"),
                map_quality_roi_to_gemini_polygon(cfg, quality_to_gemini_h),
            )
            quality_disp = draw_quality(latest_quality[0], cfg, q_score, q_status, q_ellipses, quality_detecting[0])
            panel = draw_panel(coins, q_score, cfg, q_status, snap_state[0], q_ellipses, quality_to_gemini_h is not None)
            cv2.imshow(win, np.hstack([gemini_disp, quality_disp, panel]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" ") or key == 32:
                if snap_state[0] == "live":
                    snap_color_frame[0] = color.copy()
                    snap_depth_accum.clear()
                    snap_state[0] = "accumulating"
                    print(f"[快照] 開始累積 {cfg['gemini_snapshot_frames']} 幀深度")
                else:
                    snap_state[0] = "live"
                    snap_depth_accum.clear()
                    smooth_state[0] = []
                    detect_event.set()
                    print("[快照] Back to Live")
            elif key in (ord("s"), ord("S")):
                append_log(coins, q_score, cfg, latest_quality[0], q_ellipses)
                print(f"[Log] 已寫入 {LOG_FILE.name}")

    finally:
        stop_flag[0] = True
        try:
            if "detect_event" in locals():
                detect_event.set()
        except Exception:
            pass
        if worker is not None:
            worker.join(timeout=2.0)
        if qcap is not None:
            qcap.release()
        camera.close()
        cv2.destroyAllWindows()
        print("結束")


def average_depth_stack(frames):
    stack = np.stack(frames, axis=0)
    valid = stack > 0
    count = valid.sum(axis=0)
    avg = (np.where(valid, stack, 0).sum(axis=0) / count.clip(min=1)).astype(np.float32)
    avg[count == 0] = 0
    avg_8u = cv2.normalize(avg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    filt = cv2.bilateralFilter(avg_8u, d=7, sigmaColor=20, sigmaSpace=20)
    return avg * (filt.astype(np.float32) / (avg_8u.astype(np.float32) + 1e-6))


def save_once():
    cfg = load_config()
    if "--fast" in sys.argv:
        cfg["gemini_depth_avg_frames"] = min(int(cfg.get("gemini_depth_avg_frames", 15)), 5)
        cfg["gemini_snapshot_frames"] = min(int(cfg.get("gemini_snapshot_frames", 80)), 20)
    quality_only = "--quality-only" in sys.argv or cfg.get("quality_only_save_once", False)
    calib = load_calib(cfg)
    quality_to_gemini_h = load_quality_to_gemini_homography()
    stereo_extrinsics = load_stereo_extrinsics()
    robot_calib = load_robot_calibration(cfg)
    get_sam3()
    OUT_DIR.mkdir(exist_ok=True)

    camera = Gemini2Camera(align_depth_to_color=True)
    qcap = None
    try:
        set_gemini_stream_env(cfg)
        camera.open()
        apply_color_controls(camera, cfg)
        intr = camera.intrinsics
        qcap, q_status = open_quality_camera(cfg)

        color_frames = []
        depth_frames = []
        q_frame = None
        q_score = 0.0
        n_depth = max(3, min(int(cfg["gemini_depth_avg_frames"]), 20))
        print(f"[save-once] 擷取 {n_depth} 幀 Gemini 深度平均...")
        for _ in range(n_depth):
            color, depth = camera.get_frames(timeout_ms=1000)
            if color is not None and depth is not None:
                color_frames.append(color.copy())
                depth_frames.append(depth.astype(np.float32))
            if qcap is not None:
                ok, q_read = qcap.read()
                if ok and q_read is not None:
                    q_frame = q_read.copy()

        if not color_frames or not depth_frames:
            raise RuntimeError("Gemini 沒有取得可用 color/depth frame")

        color = color_frames[-1]
        depth_avg = average_depth_stack(depth_frames)
        if q_frame is not None:
            q_score = blur_score(q_frame, cfg.get("quality_roi"))
        q_ellipses = detect_quality_ellipses(q_frame, cfg)
        q_ellipses = attach_gemini_depth_to_quality_ellipses(q_ellipses, quality_to_gemini_h, depth_avg, intr, stereo_extrinsics)
        q_ellipses = classify_quality_ellipses(q_ellipses, calib, cfg)
        q_ellipses = attach_robot_coords_to_quality_ellipses(q_ellipses, intr, robot_calib)
        counts, total_value, _ = summarize_quality_coins(q_ellipses, cfg)
        write_robot_targets(q_ellipses, counts, total_value, cfg)

        coins = []
        if not quality_only:
            print("[save-once] Gemini SAM3 偵測中...")
            for mask in detect_masks(color):
                meas = measure_coin(mask, depth_avg, intr)
                if meas is None:
                    continue
                if not is_reasonable_gemini_coin(meas, cfg):
                    print(f"[Gemini過濾] d={meas['diameter_mm']:.2f}mm  center=({meas['cx']},{meas['cy']})")
                    continue
                label, diff = classify_by_size(
                    meas["diameter_mm"],
                    calib,
                    min_margin=float(cfg.get("quality_min_class_margin_mm", 0.45)),
                )
                coins.append({
                    "mask": meas["mask"],
                    "cx": meas["cx"],
                    "cy": meas["cy"],
                    "r_px": meas["r_px"],
                    "diameter_mm": meas["diameter_mm"],
                    "z_mm": meas["z_mm"],
                    "predicted_class": label,
                    "class_diff_mm": diff,
                })
        else:
            print("[save-once] quality-only：跳過 Gemini SAM3 偵測")

        for i, coin in enumerate(coins, 1):
            print(f"[Gemini直徑] G{i}: d={coin['diameter_mm']:.2f}mm  Z={coin['z_mm']:.0f}mm  center=({coin['cx']},{coin['cy']})")
        for i, e in enumerate(q_ellipses, 1):
            diam = e.get("quality_diameter_mm")
            z = e.get("depth_z_mm")
            gx = e.get("gemini_x")
            gy = e.get("gemini_y")
            if diam is None:
                print(f"[Quality直徑] Q{i}: d=?  Gemini=({gx},{gy})")
            else:
                print(f"[Quality直徑] Q{i}: d={diam:.2f}mm  Z={z:.0f}mm  Gemini=({gx:.1f},{gy:.1f})")

        gemini_disp = draw_gemini(
            color,
            coins,
            False,
            "snapshot",
            intr,
            q_ellipses,
            cfg.get("gemini_display_roi"),
            map_quality_roi_to_gemini_polygon(cfg, quality_to_gemini_h),
        )
        quality_disp = draw_quality(q_frame, cfg, q_score, q_status, q_ellipses)
        panel = draw_panel(coins, q_score, cfg, q_status, "save-once", q_ellipses, quality_to_gemini_h is not None)
        combined = np.hstack([gemini_disp, quality_disp, panel])

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUT_DIR / f"dual_camera_snapshot_{ts}.jpg"
        gemini_path = OUT_DIR / f"gemini_view_{ts}.jpg"
        quality_path = OUT_DIR / f"quality_view_{ts}.jpg"
        cv2.imwrite(str(out_path), combined)
        cv2.imwrite(str(gemini_path), gemini_disp)
        cv2.imwrite(str(quality_path), quality_disp)
        if not quality_only:
            save_debug_crops(color, q_frame, coins, q_ellipses, intr, ts)
        append_log(coins, q_score, cfg, q_frame, q_ellipses)
        print(f"[save-once] coins={len(coins)}  quality_ellipses={len(q_ellipses)}  quality_blur={q_score:.1f}")
        print(f"[save-once] 已輸出 {out_path}")
        print(f"[save-once] Gemini view {gemini_path}")
        print(f"[save-once] Quality view {quality_path}")
    finally:
        if qcap is not None:
            qcap.release()
        camera.close()


if __name__ == "__main__":
    main()
