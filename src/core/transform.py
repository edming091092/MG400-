"""
座標轉換核心模組

流程：
  1. 讀取 marker_world_coords.json（一次性設定的 ArUco 角點世界座標）
  2. 偵測目前畫面中的 ArUco Marker 角點（像素座標）
  3. solvePnP → 算出 T_world_from_cam（4×4）
  4. 任意像素 + 深度 → 機械手臂世界座標

世界座標系 = 機械手臂座標系（單位：mm）
"""

import cv2
import numpy as np
import json
import os
from typing import Optional, Dict, List, Tuple
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from core.camera import CameraIntrinsics


# ======================================================================
#  ArUco 偵測工具
# ======================================================================

def _get_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    # 提高偵測靈敏度
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.02
    params.polygonalApproxAccuracyRate = 0.05
    return cv2.aruco.ArucoDetector(aruco_dict, params)


_detector = _get_aruco_detector()


def detect_aruco(color_bgr: np.ndarray
                 ) -> Dict[int, np.ndarray]:
    """
    在彩色影像中偵測 ArUco Marker。
    回傳 {marker_id: corners_4x2}，corners 像素座標。
    corners_4x2 順序：[top-left, top-right, bottom-right, bottom-left]
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = _detector.detectMarkers(gray)

    result: Dict[int, np.ndarray] = {}
    if ids is None:
        return result
    for corners, mid in zip(corners_list, ids.flatten()):
        if int(mid) in config.MARKER_IDS:
            result[int(mid)] = corners[0]  # shape (4, 2)
    return result


# ======================================================================
#  標靶資料載入
# ======================================================================

def load_marker_data(path: str = config.MARKER_DATA_FILE) -> dict:
    """
    載入 marker_world_coords.json
    格式：
    {
      "marker_size_mm": 50.0,
      "markers": {
        "0": {
          "corners_world": [
            [x0, y0, z0],   # top-left
            [x1, y1, z1],   # top-right
            [x2, y2, z2],   # bottom-right
            [x3, y3, z3]    # bottom-left
          ]
        },
        ...
      }
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到標靶資料：{path}\n"
            "請先執行 setup/record_markers.py 完成一次性校準設定。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================================
#  自動外參校準（每次相機開啟時呼叫）
# ======================================================================

H_SAVE_PATH = "data/H.npy"


def auto_calibrate(color_bgr: np.ndarray,
                   intrinsics: CameraIntrinsics,
                   marker_data: dict,
                   save_path: str = config.EXTRINSICS_FILE
                   ) -> np.ndarray:
    """
    偵測 ArUco Marker → Homography → 存檔。
    用各 Marker 的中心點對應，完全不依賴角點順序。

    回傳 H (3×3)，使用 cv2.perspectiveTransform 將像素轉世界 XY。
    """
    detected = detect_aruco(color_bgr)
    markers_db = marker_data["markers"]

    pixel_centers: List[List[float]] = []
    world_centers: List[List[float]] = []

    for mid_str, mdata in markers_db.items():
        mid = int(mid_str)
        if mid not in detected:
            continue
        corners_world = np.array(mdata["corners_world"], dtype=np.float64)  # (4,3)
        corners_img   = detected[mid].astype(np.float64)                    # (4,2)
        # 用中心點 — 完全不受角點順序影響
        world_centers.append(corners_world[:, :2].mean(axis=0).tolist())    # [X, Y]
        pixel_centers.append(corners_img.mean(axis=0).tolist())             # [u, v]

    n = len(pixel_centers)
    if n < 4:
        found_ids = list(detected.keys())
        needed_ids = [int(k) for k in markers_db.keys()]
        raise RuntimeError(
            f"自動校準失敗：只偵測到 {n} 個 Marker（至少需要 4 個）。\n"
            f"畫面中偵測到：{found_ids}，設定中：{needed_ids}\n"
            "請確認 ArUco 標靶在相機視野中且光線充足。"
        )

    src = np.array(pixel_centers, dtype=np.float32)   # pixel  (N,2)
    dst = np.array(world_centers, dtype=np.float32)   # world  (N,2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("findHomography 計算失敗，請確認標靶分布不共線。")

    # --- 計算重投影誤差（中心點反投影驗證）---
    src_h = src.reshape(-1, 1, 2)
    proj  = cv2.perspectiveTransform(src_h, H).reshape(-1, 2)
    errors = np.linalg.norm(proj - dst, axis=1)
    mean_err = float(np.mean(errors))
    max_err  = float(np.max(errors))
    print(f"[transform] Homography 校準完成  {n} 個 Marker 中心點")
    print(f"[transform] 世界座標重投影誤差 mean={mean_err:.2f}mm  max={max_err:.2f}mm")
    for i, (mid_str, err) in enumerate(
            zip([k for k in markers_db if int(k) in detected], errors)):
        print(f"  M{mid_str}: {err:.2f}mm")

    # --- 存檔 ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(H_SAVE_PATH, H)
    print(f"[transform] H 已儲存 → {H_SAVE_PATH}")

    # --- 診斷圖 ---
    try:
        dbg = color_bgr.copy()
        for (pc, wc) in zip(pixel_centers, world_centers):
            px = tuple(int(v) for v in pc)
            cv2.circle(dbg, px, 10, (0, 255, 0), 2)
            cv2.putText(dbg, f"({wc[0]:.0f},{wc[1]:.0f})",
                        (px[0]+12, px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(dbg, f"H err mean={mean_err:.1f}mm  max={max_err:.1f}mm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        dbg_path = os.path.join(os.path.dirname(save_path), "calibration_debug.png")
        cv2.imwrite(dbg_path, dbg)
        print(f"[transform] 診斷圖已儲存 → {dbg_path}")
    except Exception as e:
        print(f"[transform] 診斷圖儲存失敗：{e}")

    # --- solvePnP：計算相機外參，供深度→機器人 Z 轉換使用 ---
    obj_3d, img_2d = [], []
    for mid_str, mdata in markers_db.items():
        mid = int(mid_str)
        if mid not in detected:
            continue
        corners_world = np.array(mdata["corners_world"], dtype=np.float64)
        corners_img   = detected[mid].astype(np.float64)
        obj_3d.extend(corners_world.tolist())
        img_2d.extend(corners_img.tolist())

    rvec, tvec = None, None
    ret, rvec_r, tvec_r = cv2.solvePnP(
        np.array(obj_3d, dtype=np.float64),
        np.array(img_2d, dtype=np.float64),
        intrinsics.K, intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if ret:
        rvec, tvec = rvec_r, tvec_r
        np.savez(save_path, H=H, rvec=rvec, tvec=tvec)
        print(f"[transform] 外參 rvec/tvec 已儲存 → {save_path}")
    else:
        print("[transform] 警告：solvePnP 失敗，Z 將退回固定值")

    return H, rvec, tvec


def load_extrinsics(path: str = H_SAVE_PATH) -> np.ndarray:
    """載入上次存檔的 H"""
    return np.load(path)


# ======================================================================
#  座標轉換：像素 → 機械手臂世界座標（Homography）
# ======================================================================

def pixel_to_robot(u: float, v: float, depth_mm: float,
                   intrinsics: CameraIntrinsics,
                   H: np.ndarray,
                   rvec: Optional[np.ndarray] = None,
                   tvec: Optional[np.ndarray] = None) -> np.ndarray:
    """
    將影像像素座標 (u, v) 透過 Homography H 轉換為機械手臂世界 XY（mm）。
    若提供 depth_mm 及 rvec/tvec，則用相機外參計算真實 Z；否則退回固定桌面 Z。

    solvePnP 約定：P_cam = R @ P_world + t
    反推：P_world = R^T @ (P_cam - t)

    回傳：
        np.array([X, Y, Z])，單位 mm，機械手臂座標系
    """
    pt = np.array([[[u, v]]], dtype=np.float32)
    world_xy = cv2.perspectiveTransform(pt, H)[0, 0]

    if depth_mm > 0 and rvec is not None and tvec is not None:
        R, _ = cv2.Rodrigues(rvec)
        # 深度相機 Z = 光軸距離，還原相機座標系三維點
        K_inv = np.linalg.inv(intrinsics.K)
        p_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64) * depth_mm
        p_world = R.T @ (p_cam - tvec.flatten())
        z_robot = float(p_world[2])
    else:
        z_robot = float(config.TABLE_Z_MM)

    return np.array([float(world_xy[0]), float(world_xy[1]), z_robot])


# ======================================================================
#  可視化工具
# ======================================================================

def draw_aruco_overlay(img: np.ndarray,
                       detected: Dict[int, np.ndarray]) -> np.ndarray:
    """在影像上畫出偵測到的 ArUco Marker 及角點編號"""
    vis = img.copy()
    for mid, corners in detected.items():
        pts = corners.astype(int)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for i, (pt, col) in enumerate(zip(pts, colors)):
            cv2.circle(vis, tuple(pt), 6, col, -1)
            cv2.putText(vis, str(i), tuple(pt + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, col, 2)
        # 畫邊框
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        center = pts.mean(axis=0).astype(int)
        cv2.putText(vis, f"ID:{mid}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)
    return vis
