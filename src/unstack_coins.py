"""
Unstack coins into separated staging slots, then refresh recognition.

This script intentionally refuses real pick/place unless both the suction DO
channel and staging slots are configured. It reuses the existing vision target
pipeline and MG400 safety helpers.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from hover_robot_target import (
    ERROR_HINTS,
    TARGETS_FILE,
    check_bounds,
    check_lower_bounds,
    default_start_pose,
    is_auto_safe_target,
    load_config,
    load_targets,
    move_to_pose,
    parse_xyz,
    refresh_targets_after_start,
    release_camera_processes,
    safe_bounds,
    write_action_status,
)
from core.robot import MG400


DEFAULT_SAFE_Z = 100.0
DEFAULT_TRAVEL_Z = 100.0
DEFAULT_PICK_Z = -156.0
HERE = Path(__file__).parent
QUALITY_TO_GEMINI_FILE = HERE / "quality_to_gemini_homography.json"
ROBOT_TABLETOP_H_FILE = HERE / "robot_tabletop_homography.json"


def is_pick_range_target(target, cfg):
    x = float(target["robot_x_mm"])
    y = float(target["robot_y_mm"])
    x_min = float(cfg.get("robot_pick_x_min_mm", -150.0))
    x_max = float(cfg.get("robot_pick_x_max_mm", 400.0))
    y_min = float(cfg.get("robot_pick_y_min_mm", -400.0))
    y_max = float(cfg.get("robot_pick_y_max_mm", 400.0))
    return x_min <= x <= x_max and y_min <= y <= y_max


def is_safe_xy(x, y, cfg):
    bounds = safe_bounds(cfg)
    return bounds["x_min"] <= float(x) <= bounds["x_max"] and bounds["y_min"] <= float(y) <= bounds["y_max"]


def _cfg_bool(cfg, key, default=False):
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def load_homographies(cfg):
    tabletop_path = Path(cfg.get("robot_tabletop_homography_path", ROBOT_TABLETOP_H_FILE))
    if not tabletop_path.is_absolute():
        tabletop_path = HERE / tabletop_path
    qg_path = QUALITY_TO_GEMINI_FILE
    try:
        tabletop_data = json.loads(tabletop_path.read_text(encoding="utf-8"))
        if "quality_to_robot_homography" in tabletop_data:
            quality_to_robot = np.array(tabletop_data["quality_to_robot_homography"], dtype=np.float64)
            return np.linalg.inv(quality_to_robot), None
        robot_h = np.array(tabletop_data["gemini_to_robot_homography"], dtype=np.float64)
        q_to_g = np.array(json.loads(qg_path.read_text(encoding="utf-8"))["quality_to_gemini_homography"], dtype=np.float64)
    except Exception as exc:
        raise RuntimeError("找不到或無法讀取 robot/quality homography，不能限制空位到畫質相機視野") from exc
    g_to_robot_inv = np.linalg.inv(robot_h)
    g_to_q = np.linalg.inv(q_to_g)
    return g_to_robot_inv, g_to_q


def robot_xy_to_quality_xy(robot_x, robot_y, robot_to_gemini_h, gemini_to_quality_h):
    robot_pt = np.array([[[float(robot_x), float(robot_y)]]], dtype=np.float32)
    if gemini_to_quality_h is None:
        quality_pt = cv2.perspectiveTransform(robot_pt, robot_to_gemini_h).reshape(2)
        return float(quality_pt[0]), float(quality_pt[1])
    gemini_pt = cv2.perspectiveTransform(robot_pt, robot_to_gemini_h).reshape(1, 1, 2).astype(np.float32)
    quality_pt = cv2.perspectiveTransform(gemini_pt, gemini_to_quality_h).reshape(2)
    return float(quality_pt[0]), float(quality_pt[1])


def quality_roi_contains(cfg, qx, qy, margin_px=18.0):
    roi = cfg.get("quality_roi")
    if not roi:
        w = float(cfg.get("quality_width", 1280))
        h = float(cfg.get("quality_height", 720))
        x1, y1, x2, y2 = 0.0, 0.0, w, h
    else:
        x1, y1, x2, y2 = [float(v) for v in roi]
    return (x1 + margin_px) <= float(qx) <= (x2 - margin_px) and (y1 + margin_px) <= float(qy) <= (y2 - margin_px)


def release_vision_processes():
    release_camera_processes()
    ps = (
        "$names='python','pythonw'; "
        "$patterns='dual_camera_live|camera_preview_once|select_quality_roi'; "
        "$procs=Get-Process -ErrorAction SilentlyContinue | Where-Object { $names -contains $_.ProcessName }; "
        "foreach ($p in $procs) { "
        "try { $cmd=(Get-CimInstance Win32_Process -Filter \"ProcessId=$($p.Id)\" -ErrorAction SilentlyContinue).CommandLine } catch { $cmd='' }; "
        "if ($cmd -match 'coin_classifier' -and $cmd -match $patterns) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } "
        "}"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], cwd=str(HERE), text=True, capture_output=True, encoding="utf-8", errors="replace")


def fixed_staging_slots(cfg, count):
    slots = cfg.get("unstack_place_slots")
    if isinstance(slots, list) and slots:
        out = []
        for i, item in enumerate(slots[:count], 1):
            try:
                out.append({
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "z": float(item.get("z", cfg.get("unstack_place_z_mm", DEFAULT_PICK_Z))),
                    "name": str(item.get("name", f"slot{i}")),
                })
            except Exception as exc:
                raise RuntimeError(f"unstack_place_slots 第 {i} 筆格式錯誤，需包含 x/y/z") from exc
        return out

    required = ("unstack_place_start_x_mm", "unstack_place_start_y_mm", "unstack_place_z_mm")
    if not all(k in cfg for k in required):
        raise RuntimeError(
            "尚未設定暫放區。請在 dual_camera_config.json 設定 unstack_place_slots，"
            "或設定 unstack_place_start_x_mm / unstack_place_start_y_mm / unstack_place_z_mm。"
        )

    start_x = float(cfg["unstack_place_start_x_mm"])
    start_y = float(cfg["unstack_place_start_y_mm"])
    place_z = float(cfg["unstack_place_z_mm"])
    step_x = float(cfg.get("unstack_place_step_x_mm", 35.0))
    step_y = float(cfg.get("unstack_place_step_y_mm", 35.0))
    cols = max(1, int(cfg.get("unstack_place_cols", 4)))
    out = []
    for i in range(count):
        row = i // cols
        col = i % cols
        out.append({
            "x": start_x + col * step_x,
            "y": start_y + row * step_y,
            "z": place_z,
            "name": f"slot{i + 1}",
        })
    return out


def occupied_points_from_targets(data, placed_slots=None):
    points = []
    for t in data.get("targets", []):
        try:
            points.append((
                float(t["robot_x_mm"]),
                float(t["robot_y_mm"]),
                f"Q{t.get('index', '?')}",
            ))
        except Exception:
            continue
    for i, slot in enumerate(placed_slots or [], 1):
        try:
            points.append((float(slot["x"]), float(slot["y"]), f"placed{i}"))
        except Exception:
            continue
    return points


def _dist_xy(a, b):
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


def auto_empty_slot(cfg, occupied, placed_count, robot_to_gemini_h=None, gemini_to_quality_h=None):
    bounds = safe_bounds(cfg)
    x_min = max(float(cfg.get("unstack_auto_place_x_min_mm", cfg.get("robot_auto_x_min_mm", -100.0))), bounds["x_min"])
    x_max = min(float(cfg.get("unstack_auto_place_x_max_mm", cfg.get("robot_auto_x_max_mm", 370.0))), bounds["x_max"])
    y_min = max(float(cfg.get("unstack_auto_place_y_min_mm", cfg.get("robot_auto_y_min_mm", -200.0))), bounds["y_min"])
    y_max = min(float(cfg.get("unstack_auto_place_y_max_mm", cfg.get("robot_auto_y_max_mm", 190.0))), bounds["y_max"])
    step = max(10.0, float(cfg.get("unstack_auto_place_grid_step_mm", 35.0)))
    clearance = max(10.0, float(cfg.get("unstack_empty_clearance_mm", 42.0)))
    clearance_tol = max(0.0, float(cfg.get("unstack_empty_clearance_tolerance_mm", 2.0)))
    place_z = float(cfg.get("unstack_place_z_mm", cfg.get("unstack_pick_z_mm", DEFAULT_PICK_Z)))
    prefer = str(cfg.get("unstack_auto_place_prefer", "bottom_left")).lower()
    debug = _cfg_bool(cfg, "unstack_auto_place_debug", True)
    skip_used_prefix = _cfg_bool(cfg, "unstack_auto_place_skip_used_prefix", True)
    stats = {"tested": 0, "outside_roi": 0, "too_close": 0}

    def make_slot(x, y, qx, qy, nearest):
        return {
            "x": float(x),
            "y": float(y),
            "z": place_z,
            "name": f"auto_empty{placed_count + 1}",
            "nearest_occupied_mm": round(float(nearest), 3),
            "quality_x_px": None if qx is None else round(float(qx), 3),
            "quality_y_px": None if qy is None else round(float(qy), 3),
        }

    def check_slot(x, y):
        stats["tested"] += 1
        if robot_to_gemini_h is not None and gemini_to_quality_h is not None:
            qx, qy = robot_xy_to_quality_xy(x, y, robot_to_gemini_h, gemini_to_quality_h)
            if not quality_roi_contains(cfg, qx, qy, float(cfg.get("unstack_quality_roi_margin_px", 24.0))):
                stats["outside_roi"] += 1
                return None
        else:
            qx, qy = None, None

        nearest = min((_dist_xy((x, y), p) for p in occupied), default=9999.0)
        if nearest + clearance_tol < clearance:
            stats["too_close"] += 1
            return None
        return make_slot(x, y, qx, qy, nearest)

    def log_choice(slot):
        if debug:
            print(
                f"[unstack] empty scan tested={stats['tested']} outside_roi={stats['outside_roi']} "
                f"too_close={stats['too_close']} occupied={len(occupied)} "
                f"chosen=({slot['x']:.1f},{slot['y']:.1f}) nearest={slot['nearest_occupied_mm']:.1f}mm "
                f"clearance={clearance:.1f}-{clearance_tol:.1f}mm"
            )
        return slot

    def scan_points_from_cursor(points):
        if not points:
            return None
        if not _cfg_bool(cfg, "unstack_auto_place_cursor_enabled", True):
            return None
        lookback = max(0, int(cfg.get("unstack_auto_place_cursor_lookback_slots", 2)))
        start_i = max(0, min(len(points) - 1, int(placed_count) - lookback))
        for offset in range(len(points)):
            idx = (start_i + offset) % len(points)
            x, y = points[idx]
            slot = check_slot(x, y)
            if slot is not None:
                if debug:
                    print(
                        f"[unstack] empty cursor start={start_i} idx={idx} "
                        f"placed_count={placed_count} lookback={lookback}"
                    )
                return log_choice(slot)
        return None

    if prefer in ("x_then_plus_y", "plus_y_then_plus_x", "column_up"):
        points = []
        x = x_min
        while x <= x_max + 0.001:
            y = y_min
            while y <= y_max + 0.001:
                points.append((x, y))
                y += step
            x += step
        cursor_slot = scan_points_from_cursor(points)
        if cursor_slot is not None:
            return cursor_slot
        slot_i = 0
        x = x_min
        while x <= x_max + 0.001:
            y = y_min
            while y <= y_max + 0.001:
                if skip_used_prefix and slot_i < int(placed_count):
                    slot_i += 1
                    y += step
                    continue
                slot = check_slot(x, y)
                if slot is not None:
                    return log_choice(slot)
                slot_i += 1
                y += step
            x += step

    if prefer in ("x_then_minus_y", "minus_y_then_plus_x", "column_down"):
        points = []
        x = x_min
        while x <= x_max + 0.001:
            y = y_max
            while y >= y_min - 0.001:
                points.append((x, y))
                y -= step
            x += step
        cursor_slot = scan_points_from_cursor(points)
        if cursor_slot is not None:
            return cursor_slot
        slot_i = 0
        x = x_min
        while x <= x_max + 0.001:
            y = y_max
            while y >= y_min - 0.001:
                if skip_used_prefix and slot_i < int(placed_count):
                    slot_i += 1
                    y -= step
                    continue
                slot = check_slot(x, y)
                if slot is not None:
                    return log_choice(slot)
                slot_i += 1
                y -= step
            x += step

    candidates = []
    y = y_min
    while y <= y_max + 0.001:
        x = x_min
        while x <= x_max + 0.001:
            slot = check_slot(x, y)
            if slot is not None:
                if prefer == "top_right":
                    tie = (-x, -y)
                elif prefer == "top_left":
                    tie = (x, -y)
                elif prefer == "bottom_right":
                    tie = (-x, y)
                else:
                    tie = (x, y)
                candidates.append((slot["nearest_occupied_mm"], tie, slot))
            x += step
        y += step

    if not candidates:
        if debug:
            print(
                f"[unstack] empty scan failed tested={stats['tested']} outside_roi={stats['outside_roi']} "
                f"too_close={stats['too_close']} occupied={len(occupied)}"
            )
        raise RuntimeError(
            f"找不到空位：搜尋區 X={x_min:.1f}..{x_max:.1f}, Y={y_min:.1f}..{y_max:.1f}, "
            f"clearance={clearance:.1f}mm。請調整 unstack_auto_place_* 或清空暫放區。"
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _nearest, _tie, slot = candidates[0]
    return log_choice(slot)


def next_empty_slot(cfg, data, placed_slots, failed_slots, robot_to_gemini_h=None, gemini_to_quality_h=None):
    occupied = occupied_points_from_targets(data, list(placed_slots or []) + list(failed_slots or []))
    return auto_empty_slot(
        cfg,
        occupied,
        len(placed_slots or []) + len(failed_slots or []),
        robot_to_gemini_h,
        gemini_to_quality_h,
    )


def validate_real_pick_config(cfg, real_pick):
    if not real_pick:
        return None
    if not _cfg_bool(cfg, "unstack_real_pick_enabled", False):
        raise RuntimeError("真實拆堆未啟用：請設定 unstack_real_pick_enabled=true")
    if cfg.get("suction_do_index") is None:
        raise RuntimeError("尚未設定 suction_do_index，不能切換真空/夾爪 DO")
    return int(cfg["suction_do_index"])


def _target_depth(target):
    try:
        z = float(target.get("depth_z_mm"))
    except Exception:
        return None
    return z if z > 0 else None


def _nearby_depths_mm(target, all_targets, radius_mm):
    try:
        tx = float(target["robot_x_mm"])
        ty = float(target["robot_y_mm"])
        t_index = int(target.get("index", -1))
    except Exception:
        return []
    depths = []
    for other in all_targets:
        try:
            if int(other.get("index", -2)) == t_index:
                continue
            ox = float(other["robot_x_mm"])
            oy = float(other["robot_y_mm"])
        except Exception:
            continue
        if _dist_xy((tx, ty), (ox, oy)) <= float(radius_mm):
            depth = _target_depth(other)
            if depth is not None:
                depths.append(depth)
    return depths


def _estimated_plane_depth_mm(target, all_targets, radius_mm=60.0):
    """Estimate the local flat/reference depth near a coin.

    Depth values are camera distance: a higher coin has a smaller depth. The
    best plane reference is the deepest nearby coin top; if there is no local
    reference, fall back to the deep end of all visible coin tops.
    """
    z = _target_depth(target)
    nearby_depths = _nearby_depths_mm(target, all_targets, radius_mm)
    if z is None:
        return None
    if nearby_depths:
        return max(nearby_depths)
    all_depths = [
        depth for other in all_targets
        if int(other.get("index", -2)) != int(target.get("index", -1))
        for depth in [_target_depth(other)]
        if depth is not None
    ]
    if not all_depths:
        return None
    return float(np.percentile(all_depths, 80))


def _stack_height_mm(target, all_targets, radius_mm=60.0):
    try:
        top_height = target.get("top_surface_height_mm")
        if top_height not in (None, ""):
            return max(0.0, float(top_height))
    except Exception:
        pass
    try:
        actual_height = target.get("height_above_table_mm")
        if actual_height not in (None, ""):
            return max(0.0, float(actual_height))
    except Exception:
        pass
    z = _target_depth(target)
    plane_z = _estimated_plane_depth_mm(target, all_targets, radius_mm)
    if z is None or plane_z is None:
        return 0.0
    return max(0.0, plane_z - z)


def _has_actual_table_height(target):
    try:
        return target.get("height_above_table_mm") not in (None, "")
    except Exception:
        return False


def _nearby_reference_count(target, all_targets, radius_mm):
    nearby_depths = _nearby_depths_mm(target, all_targets, radius_mm)
    if nearby_depths:
        return len(nearby_depths)
    try:
        t_index = int(target.get("index", -1))
    except Exception:
        t_index = -1
    return sum(
        1 for other in all_targets
        if int(other.get("index", -2)) != t_index and _target_depth(other) is not None
    )


def _nearby_coin_distance_mm(target, all_targets):
    try:
        tx = float(target["robot_x_mm"])
        ty = float(target["robot_y_mm"])
        t_index = int(target.get("index", -1))
    except Exception:
        return None
    nearest = None
    for other in all_targets:
        try:
            if int(other.get("index", -2)) == t_index:
                continue
            ox = float(other["robot_x_mm"])
            oy = float(other["robot_y_mm"])
        except Exception:
            continue
        dist = _dist_xy((tx, ty), (ox, oy))
        nearest = dist if nearest is None else min(nearest, dist)
    return nearest


def _top_visible_sort_key(target, all_targets, cfg):
    radius = float(cfg.get("unstack_stack_neighbor_distance_mm", 60.0))
    height = _stack_height_mm(target, all_targets, radius)
    depth = _target_depth(target)
    visible = float(target.get("visible_score", target.get("top_visible_score")) or 0.0)
    score = float(target.get("top_coin_score") or 0.0)
    overlap = _bbox_overlap_risk(target, all_targets)
    clean_overlap_max = float(cfg.get("unstack_prefer_clean_overlap_max", 0.08))
    contains_count, _contains_other = _center_contains_other_count(target, all_targets, cfg)
    higher_risk, _higher = _higher_occluder_risk(target, all_targets, cfg)
    higher_risk_min = float(cfg.get("unstack_occlusion_overlap_min", 0.03))
    clean_bucket = 0 if overlap <= clean_overlap_max and contains_count == 0 else 1
    try:
        axis_ratio = float(target.get("axis_ratio") or 0.0)
    except Exception:
        axis_ratio = 0.0
    complete_visible = float(cfg.get("unstack_prefer_complete_visible_score", 0.86))
    complete_axis = float(cfg.get("unstack_prefer_complete_axis_ratio", 0.90))
    complete_bucket = 0 if visible >= complete_visible and axis_ratio >= complete_axis else 1
    # Grasp order for stacked objects should be decided by support/occlusion first:
    # after that, physical top-ness (height/depth) must win over a visually clean
    # bbox. Otherwise a lower coin with less bbox overlap can beat the real top coin.
    return (
        contains_count,
        1 if higher_risk >= higher_risk_min else 0,
        complete_bucket,
        -height,
        depth if depth is not None else 9999.0,
        clean_bucket,
        overlap,
        -visible,
        -axis_ratio,
        -score,
        int(target.get("index", 9999)),
    )


def _target_radius_px(target):
    box = target.get("bbox_xyxy")
    if box and len(box) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
            radius = 0.25 * ((x2 - x1) + (y2 - y1))
            if radius > 1.0:
                return radius
        except Exception:
            pass
    try:
        diameter = float(target.get("diameter_px"))
        if diameter > 2.0:
            return diameter * 0.5
    except Exception:
        pass
    return None


def _circle_overlap_fraction(target, other):
    try:
        ax = float(target.get("quality_x_px"))
        ay = float(target.get("quality_y_px"))
        bx = float(other.get("quality_x_px"))
        by = float(other.get("quality_y_px"))
    except Exception:
        return 0.0
    ar = _target_radius_px(target)
    br = _target_radius_px(other)
    if ar is None or br is None or ar <= 1.0 or br <= 1.0:
        return 0.0
    d = math.hypot(ax - bx, ay - by)
    if d >= ar + br:
        return 0.0
    if d <= abs(ar - br):
        return float((min(ar, br) ** 2) / (ar ** 2))
    try:
        a1 = ar * ar * math.acos((d * d + ar * ar - br * br) / (2.0 * d * ar))
        a2 = br * br * math.acos((d * d + br * br - ar * ar) / (2.0 * d * br))
        a3 = 0.5 * math.sqrt(
            max(0.0, (-d + ar + br) * (d + ar - br) * (d - ar + br) * (d + ar + br))
        )
        return float(max(0.0, a1 + a2 - a3) / (math.pi * ar * ar))
    except Exception:
        return 0.0


def _bbox_overlap_risk(target, all_targets):
    box = target.get("bbox_xyxy")
    if not box or len(box) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in box]
    except Exception:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    if area_a <= 1.0:
        return 0.0
    risk = 0.0
    try:
        target_index = int(target.get("index", -1))
    except Exception:
        target_index = -1
    for other in all_targets or []:
        try:
            if int(other.get("index", -2)) == target_index:
                continue
            bx1, by1, bx2, by2 = [float(v) for v in other.get("bbox_xyxy") or []]
        except Exception:
            continue
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter > 0:
            risk += inter / area_a
    return float(risk)


def _higher_occluder_risk(target, all_targets, cfg):
    box = target.get("bbox_xyxy")
    if not box or len(box) != 4:
        return 0.0, None
    target_depth = _target_depth(target)
    if target_depth is None:
        return 0.0, None
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in box]
    except Exception:
        return 0.0, None
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    if area_a <= 1.0:
        return 0.0, None
    margin = float(cfg.get("unstack_occlusion_depth_margin_mm", 0.6))
    risk = 0.0
    occluder = None
    try:
        target_index = int(target.get("index", -1))
    except Exception:
        target_index = -1
    for other in all_targets or []:
        try:
            if int(other.get("index", -2)) == target_index:
                continue
            bx1, by1, bx2, by2 = [float(v) for v in other.get("bbox_xyxy") or []]
        except Exception:
            continue
        other_depth = _target_depth(other)
        if other_depth is None or other_depth > target_depth - margin:
            continue
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            continue
        overlap = max(inter / area_a, _circle_overlap_fraction(target, other))
        if overlap > risk:
            risk = overlap
            occluder = other
    return float(risk), occluder


def _center_contains_other_count(target, all_targets, cfg):
    box = target.get("bbox_xyxy")
    if not box or len(box) != 4:
        return 0, None
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in box]
        target_index = int(target.get("index", -1))
    except Exception:
        return 0, None
    target_depth = _target_depth(target)
    margin = float(cfg.get("unstack_occlusion_depth_margin_mm", 0.6))
    pad = float(cfg.get("unstack_relation_center_pad_px", 4.0))
    count = 0
    first = None
    for other in all_targets or []:
        try:
            if int(other.get("index", -2)) == target_index:
                continue
            cx = float(other.get("quality_x_px"))
            cy = float(other.get("quality_y_px"))
        except Exception:
            continue
        if (ax1 - pad) <= cx <= (ax2 + pad) and (ay1 - pad) <= cy <= (ay2 + pad):
            other_depth = _target_depth(other)
            if target_depth is not None and other_depth is not None and other_depth > target_depth - margin:
                continue
            count += 1
            if first is None:
                first = other
    return count, first


def _center_occluded_by_higher(target, all_targets, cfg):
    target_depth = _target_depth(target)
    if target_depth is None:
        return False, None, 0.0
    try:
        cx = float(target.get("quality_x_px"))
        cy = float(target.get("quality_y_px"))
    except Exception:
        return False, None, 0.0
    margin = float(cfg.get("unstack_occlusion_depth_margin_mm", 0.6))
    pad = float(cfg.get("unstack_suction_center_pad_px", 8.0))
    try:
        target_index = int(target.get("index", -1))
    except Exception:
        target_index = -1
    for other in all_targets or []:
        try:
            if int(other.get("index", -2)) == target_index:
                continue
            other_depth = _target_depth(other)
            if other_depth is None or other_depth > target_depth - margin:
                continue
            bx1, by1, bx2, by2 = [float(v) for v in other.get("bbox_xyxy") or []]
        except Exception:
            continue
        if (bx1 - pad) <= cx <= (bx2 + pad) and (by1 - pad) <= cy <= (by2 + pad):
            return True, other, float(target_depth - other_depth)
    return False, None, 0.0


def _round_fallback_sort_key(target):
    visible = float(target.get("visible_score", target.get("top_visible_score")) or 0.0)
    score = float(target.get("top_coin_score") or 0.0)
    try:
        axis_ratio = float(target.get("axis_ratio") or 0.0)
    except Exception:
        axis_ratio = 0.0
    depth = _target_depth(target)
    return (
        -axis_ratio,
        -visible,
        -score,
        depth if depth is not None else 9999.0,
        int(target.get("index", 9999)),
    )


def _is_round_fallback_target(target, cfg):
    try:
        axis_ratio = float(target.get("axis_ratio") or 0.0)
    except Exception:
        axis_ratio = 0.0
    min_axis_ratio = float(cfg.get("unstack_min_complete_axis_ratio", 0.85))
    if axis_ratio < min_axis_ratio:
        return False, f"not_round:axis={axis_ratio:.2f} < {min_axis_ratio:.2f}"
    if target.get("diameter_mm") is None:
        return False, "no_complete_circle_diameter"
    ok_diam, diam_reason = _diameter_matches_class(target, cfg, "unstack_min_class_diameter_ratio", 0.70)
    if not ok_diam:
        return False, diam_reason
    if _target_depth(target) is None:
        return False, "no_depth"
    return True, f"round_fallback axis={axis_ratio:.2f}"


def _is_round_fallback_target_visible(target, all_targets, cfg):
    ok, reason = _is_round_fallback_target(target, cfg)
    if not ok:
        return False, reason
    if _cfg_bool(cfg, "unstack_reject_occluded_by_higher", True):
        higher_risk, occluder = _higher_occluder_risk(target, all_targets, cfg)
        min_higher_overlap = float(cfg.get("unstack_occlusion_overlap_min", 0.03))
        if higher_risk >= min_higher_overlap:
            occluder_id = "?" if occluder is None else occluder.get("index", "?")
            return False, f"fallback_covered_by_higher:Q{occluder_id} overlap={higher_risk:.2f}"
    if _cfg_bool(cfg, "unstack_reject_center_occluded", True):
        center_blocked, occluder, depth_gap = _center_occluded_by_higher(target, all_targets, cfg)
        if center_blocked:
            occluder_id = "?" if occluder is None else occluder.get("index", "?")
            return False, f"fallback_center_covered_by_higher:Q{occluder_id} dz={depth_gap:.1f}mm"
    visible = float(target.get("visible_score", target.get("top_visible_score")) or 0.0)
    min_visible = float(cfg.get("unstack_fallback_min_visible_score", 0.88))
    if visible < min_visible:
        return False, f"fallback_visible_low:{visible:.2f} < {min_visible:.2f}"
    contains_count, contains_other = _center_contains_other_count(target, all_targets, cfg)
    max_contains = int(cfg.get("unstack_fallback_max_contained_centers", 0))
    if contains_count > max_contains:
        other_id = "?" if contains_other is None else contains_other.get("index", "?")
        return False, f"fallback_contains_other_center:Q{other_id} count={contains_count} > {max_contains}"
    overlap = _bbox_overlap_risk(target, all_targets)
    if _cfg_bool(cfg, "unstack_fallback_reject_generic_overlap", False):
        max_overlap = float(cfg.get("unstack_fallback_max_overlap", 0.12))
        if overlap > max_overlap:
            return False, f"fallback_overlap_high:{overlap:.2f} > {max_overlap:.2f}"
    height = _stack_height_mm(target, all_targets, float(cfg.get("unstack_stack_neighbor_distance_mm", 60.0)))
    min_height = float(cfg.get("unstack_fallback_min_height_mm", 0.4))
    if height < min_height:
        return False, f"fallback_height_low:{height:.2f} < {min_height:.2f}"
    return True, reason


def _is_last_chance_unstack_target(target, all_targets, cfg):
    ok, reason = _is_round_fallback_target(target, cfg)
    if not ok:
        return False, reason
    depth = _target_depth(target)
    if depth is None:
        return False, "no_depth"
    height = _stack_height_mm(target, all_targets, float(cfg.get("unstack_stack_neighbor_distance_mm", 60.0)))
    min_height = float(cfg.get("unstack_last_chance_min_height_mm", 0.3))
    if height < min_height:
        return False, f"last_chance_height_low:{height:.2f} < {min_height:.2f}"
    if _cfg_bool(cfg, "unstack_reject_occluded_by_higher", True):
        higher_risk, occluder = _higher_occluder_risk(target, all_targets, cfg)
        min_higher_overlap = float(cfg.get("unstack_occlusion_overlap_min", 0.03))
        if higher_risk >= min_higher_overlap:
            occluder_id = "?" if occluder is None else occluder.get("index", "?")
            return False, f"last_chance_covered_by_higher:Q{occluder_id} overlap={higher_risk:.2f}"
    overlap = _bbox_overlap_risk(target, all_targets)
    if _cfg_bool(cfg, "unstack_last_chance_reject_generic_overlap", False):
        max_overlap = float(cfg.get("unstack_last_chance_max_overlap", 0.38))
        if overlap > max_overlap:
            return False, f"last_chance_overlap_high:{overlap:.2f} > {max_overlap:.2f}"
    contains_count, contains_other = _center_contains_other_count(target, all_targets, cfg)
    max_contains = int(cfg.get("unstack_last_chance_max_contained_centers", 1))
    if contains_count > max_contains:
        other_id = "?" if contains_other is None else contains_other.get("index", "?")
        return False, f"last_chance_contains_other_center:Q{other_id} count={contains_count} > {max_contains}"
    if _cfg_bool(cfg, "unstack_reject_center_occluded", True):
        center_blocked, occluder, depth_gap = _center_occluded_by_higher(target, all_targets, cfg)
        if center_blocked:
            occluder_id = "?" if occluder is None else occluder.get("index", "?")
            return False, f"last_chance_center_covered_by_higher:Q{occluder_id} dz={depth_gap:.1f}mm"
    visible = float(target.get("visible_score", target.get("top_visible_score")) or 0.0)
    min_visible = float(cfg.get("unstack_last_chance_min_visible_score", 0.75))
    if visible < min_visible:
        return False, f"last_chance_visible_low:{visible:.2f} < {min_visible:.2f}"
    return True, f"last_chance height={height:.2f} overlap={overlap:.2f}"


def _last_chance_sort_key(target, all_targets, cfg):
    height = _stack_height_mm(target, all_targets, float(cfg.get("unstack_stack_neighbor_distance_mm", 60.0)))
    overlap = _bbox_overlap_risk(target, all_targets)
    visible = float(target.get("visible_score", target.get("top_visible_score")) or 0.0)
    score = float(target.get("top_coin_score") or 0.0)
    try:
        axis_ratio = float(target.get("axis_ratio") or 0.0)
    except Exception:
        axis_ratio = 0.0
    depth = _target_depth(target)
    return (
        overlap,
        -visible,
        -axis_ratio,
        -height,
        -score,
        depth if depth is not None else 9999.0,
        int(target.get("index", 9999)),
    )


def _nearby_coin_count(target, all_targets, radius_mm):
    try:
        tx = float(target["robot_x_mm"])
        ty = float(target["robot_y_mm"])
        t_index = int(target.get("index", -1))
    except Exception:
        return 0
    count = 0
    for other in all_targets:
        try:
            if int(other.get("index", -2)) == t_index:
                continue
            ox = float(other["robot_x_mm"])
            oy = float(other["robot_y_mm"])
        except Exception:
            continue
        if _dist_xy((tx, ty), (ox, oy)) <= float(radius_mm):
            count += 1
    return count


def _diameter_matches_class(target, cfg, key="unstack_min_class_diameter_ratio", default=0.70):
    try:
        measured = float(target.get("diameter_mm"))
    except Exception:
        return False, "no_complete_circle_diameter"
    label = str(target.get("label") or target.get("label_name") or "").strip()
    diameters = cfg.get("coin_class_diameters_mm", {}) or {}
    aliases = {
        "1NT": "1yuan",
        "5NT": "5yuan",
        "10NT": "10yuan",
        "50NT": "50yuan",
        "1": "1yuan",
        "5": "5yuan",
        "10": "10yuan",
        "50": "50yuan",
    }
    class_key = aliases.get(label, label)
    expected = diameters.get(class_key)
    if expected is None:
        return True, "class_diameter_unknown"
    min_ratio = float(cfg.get(key, default))
    ratio = measured / float(expected)
    if ratio < min_ratio:
        return False, f"partial_class_diameter:{measured:.1f}/{float(expected):.1f}={ratio:.2f} < {min_ratio:.2f}"
    return True, f"class_diameter_ratio={ratio:.2f}"


def _is_near_processed_source(target, processed_sources, radius_mm):
    if not processed_sources:
        return False
    try:
        x = float(target["robot_x_mm"])
        y = float(target["robot_y_mm"])
    except Exception:
        return False
    for source in processed_sources:
        try:
            sx = float(source[0])
            sy = float(source[1])
            radius = float(source[2]) if len(source) >= 3 else float(radius_mm)
        except Exception:
            continue
        if _dist_xy((x, y), (sx, sy)) <= radius:
            return True
    return False


def is_stacked_top_target(target, all_targets, cfg):
    min_height = float(cfg.get("unstack_min_stack_height_mm", 4.0))
    min_visible = float(cfg.get("unstack_min_top_visible_score", 0.72))
    min_score = float(cfg.get("unstack_min_top_coin_score", 0.70))
    min_axis_ratio = float(cfg.get("unstack_min_complete_axis_ratio", 0.82))
    max_reasonable_height = float(cfg.get("unstack_max_reasonable_height_mm", 25.0))
    require_circle = _cfg_bool(cfg, "unstack_require_complete_circle", True)
    require_nearby = _cfg_bool(cfg, "unstack_require_nearby_under_coin", True)
    nearby_max = float(cfg.get("unstack_stack_neighbor_distance_mm", 60.0))
    min_neighbors = int(cfg.get("unstack_min_stack_neighbors", 1))

    height = _stack_height_mm(target, all_targets, nearby_max)
    visible_raw = target.get("visible_score", target.get("top_visible_score"))
    visible = None if visible_raw in (None, "") else float(visible_raw)
    top_score = float(target.get("top_coin_score") or 0.0)
    try:
        axis_ratio = float(target.get("axis_ratio") or 0.0)
    except Exception:
        axis_ratio = 0.0
    nearest = _nearby_coin_distance_mm(target, all_targets)
    nearby_count = _nearby_coin_count(target, all_targets, nearby_max)
    reference_count = _nearby_reference_count(target, all_targets, nearby_max)
    if _cfg_bool(cfg, "unstack_require_actual_table_height", True) and not _has_actual_table_height(target):
        return False, "no_height_above_table"
    if height > max_reasonable_height:
        return False, f"bad_depth_height:{height:.2f}mm > {max_reasonable_height:.2f}mm"
    if height <= min_height:
        return False, f"flat_plane:height={height:.2f}mm <= {min_height:.2f}mm"
    if require_nearby and (nearest is None or nearest > nearby_max or nearby_count < min_neighbors):
        nearest_text = "none" if nearest is None else f"{nearest:.1f}mm"
        return False, (
            f"not_stacked:no_nearby_cluster nearest={nearest_text} "
            f"neighbors={nearby_count} need={min_neighbors} radius={nearby_max:.1f}mm"
        )
    has_cluster = (not require_nearby) or (
        nearest is not None and nearest <= nearby_max and nearby_count >= min_neighbors
    )
    if visible is not None and visible < min_visible:
        return False, f"incomplete_circle:visible={visible:.2f} < {min_visible:.2f}"
    if axis_ratio < min_axis_ratio:
        return False, f"incomplete_circle:axis={axis_ratio:.2f} < {min_axis_ratio:.2f}"
    if top_score < min_score:
        return False, f"low_top_score:{top_score:.2f} < {min_score:.2f}"
    if require_circle and target.get("diameter_mm") is None:
        return False, "no_complete_circle_diameter"
    ok_diam, diam_reason = _diameter_matches_class(target, cfg, "unstack_min_class_diameter_ratio", 0.70)
    if not ok_diam:
        return False, diam_reason
    contains_count, contains_other = _center_contains_other_count(target, all_targets, cfg)
    max_contains = int(cfg.get("unstack_max_contained_centers", 0))
    if contains_count > max_contains:
        other_id = "?" if contains_other is None else contains_other.get("index", "?")
        return False, f"supports_or_contains_other_center:Q{other_id} count={contains_count} > {max_contains}"
    if _cfg_bool(cfg, "unstack_reject_center_occluded", True):
        center_blocked, occluder, depth_gap = _center_occluded_by_higher(target, all_targets, cfg)
        if center_blocked:
            occluder_id = "?" if occluder is None else occluder.get("index", "?")
            return False, f"suction_center_covered_by_higher:Q{occluder_id} dz={depth_gap:.1f}mm"
    if _cfg_bool(cfg, "unstack_reject_occluded_by_higher", True):
        higher_risk, occluder = _higher_occluder_risk(target, all_targets, cfg)
        min_higher_overlap = float(cfg.get("unstack_occlusion_overlap_min", 0.03))
        if higher_risk >= min_higher_overlap:
            occluder_id = "?" if occluder is None else occluder.get("index", "?")
            return False, f"covered_by_higher:Q{occluder_id} overlap={higher_risk:.2f}"
    overlap = _bbox_overlap_risk(target, all_targets)
    if _cfg_bool(cfg, "unstack_reject_generic_overlap", False):
        max_top_overlap = float(cfg.get("unstack_top_max_overlap", 0.24))
        if overlap > max_top_overlap:
            return False, f"overlap_too_high:{overlap:.2f} > {max_top_overlap:.2f}"
    nearest_text = "none" if nearest is None else f"{nearest:.1f}mm"
    visible_text = "NA" if visible is None else f"{visible:.2f}"
    return True, (
        f"stack_height={height:.2f}mm nearest={nearest_text} "
        f"neighbors={nearby_count} plane_refs={reference_count} visible={visible_text} "
        f"axis={axis_ratio:.2f} score={top_score:.2f}"
    )


def select_top_target(cfg, processed_sources=None):
    if not TARGETS_FILE.exists():
        raise RuntimeError("找不到 robot_targets.json，請先重新辨識")
    data = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    skip_radius = float(cfg.get("unstack_skip_picked_source_radius_mm", 24.0))
    targets = [
        t for t in data.get("targets", [])
        if t.get("robot_x_mm") is not None
        and t.get("robot_y_mm") is not None
        and t.get("diameter_mm") is not None
        and is_pick_range_target(t, cfg)
        and is_safe_xy(t["robot_x_mm"], t["robot_y_mm"], cfg)
        and (skip_radius <= 0 or not _is_near_processed_source(t, processed_sources or [], skip_radius))
    ]
    if _cfg_bool(cfg, "unstack_require_classified_pick", True):
        targets = [
            t for t in targets
            if t.get("label") not in (None, "?")
            and t.get("label_name") not in (None, "?")
            and t.get("diameter_mm") is not None
        ]
    if not targets:
        raise RuntimeError("目前沒有在夾取安全範圍內且可完整辨識、且尚未處理過的上層硬幣")
    priority = str(cfg.get("unstack_pick_priority", "stacked_highest")).lower()
    if priority in ("top_visible_highest", "top_visible", "complete_top", "visible_highest"):
        checked = []
        for target in targets:
            ok, reason = is_stacked_top_target(target, data.get("targets", []), cfg)
            checked.append((target, ok, reason))
            print(f"[unstack] top-visible check Q{target.get('index')}: {reason}")
        candidates = [(target, reason) for target, ok, reason in checked if ok and _target_depth(target) is not None]
        if not candidates:
            if _cfg_bool(cfg, "unstack_fallback_pick_any_round", True):
                fallback = []
                for target, _ok, _reason in checked:
                    fb_ok, fb_reason = _is_round_fallback_target_visible(target, data.get("targets", []), cfg)
                    print(f"[unstack] round fallback check Q{target.get('index')}: {fb_reason}")
                    if fb_ok:
                        fallback.append((target, fb_reason))
                if fallback:
                    fallback.sort(key=lambda item: _round_fallback_sort_key(item[0]))
                    candidate, reason = fallback[0]
                    print(
                        f"[unstack] selected Q{candidate.get('index')}: {reason} "
                        f"visible={float(candidate.get('visible_score', candidate.get('top_visible_score')) or 0.0):.2f} "
                        f"score={float(candidate.get('top_coin_score') or 0.0):.2f}"
                    )
                    return data, candidate
            if _cfg_bool(cfg, "unstack_last_chance_enabled", True):
                last_chance = []
                for target, _ok, _reason in checked:
                    lc_ok, lc_reason = _is_last_chance_unstack_target(target, data.get("targets", []), cfg)
                    print(f"[unstack] last-chance check Q{target.get('index')}: {lc_reason}")
                    if lc_ok:
                        last_chance.append((target, lc_reason))
                if last_chance:
                    last_chance.sort(key=lambda item: _last_chance_sort_key(item[0], data.get("targets", []), cfg))
                    candidate, reason = last_chance[0]
                    print(
                        f"[unstack] selected Q{candidate.get('index')}: {reason} "
                        f"overlap={_bbox_overlap_risk(candidate, data.get('targets', [])):.2f} "
                        f"height={_stack_height_mm(candidate, data.get('targets', []), float(cfg.get('unstack_stack_neighbor_distance_mm', 60.0))):.2f}mm"
                    )
                    return data, candidate
            reasons = "; ".join(f"Q{target.get('index')} {reason}" for target, _ok, reason in checked[:8])
            raise RuntimeError(f"目前沒有完整且高於平面的上層硬幣：{reasons}")
        candidates.sort(key=lambda item: _top_visible_sort_key(item[0], data.get("targets", []), cfg))
        for rank, (rank_target, rank_reason) in enumerate(candidates[:8], 1):
            print(
                f"[unstack] top-visible rank {rank}: Q{rank_target.get('index')} "
                f"visible={float(rank_target.get('visible_score', rank_target.get('top_visible_score')) or 0.0):.2f} "
                f"overlap={_bbox_overlap_risk(rank_target, data.get('targets', [])):.2f} "
                f"axis={float(rank_target.get('axis_ratio') or 0.0):.2f} "
                f"height={_stack_height_mm(rank_target, data.get('targets', []), float(cfg.get('unstack_stack_neighbor_distance_mm', 60.0))):.2f}mm "
                f"score={float(rank_target.get('top_coin_score') or 0.0):.2f} reason={rank_reason}"
            )
        candidate, reason = candidates[0]
        print(
            f"[unstack] selected Q{candidate.get('index')}: top_visible "
            f"height={_stack_height_mm(candidate, data.get('targets', []), float(cfg.get('unstack_stack_neighbor_distance_mm', 60.0))):.2f}mm "
            f"depth={_target_depth(candidate):.1f} visible={float(candidate.get('visible_score', candidate.get('top_visible_score')) or 0.0):.2f} "
            f"overlap={_bbox_overlap_risk(candidate, data.get('targets', [])):.2f} "
            f"axis={float(candidate.get('axis_ratio') or 0.0):.2f} score={float(candidate.get('top_coin_score') or 0.0):.2f}"
        )
        return data, candidate
    if priority in ("q_index_above_plane", "q_index", "quality_index"):
        checked = []
        for target in targets:
            ok, reason = is_stacked_top_target(target, data.get("targets", []), cfg)
            checked.append((target, ok, reason))
            print(f"[unstack] Q-priority check Q{target.get('index')}: {reason}")
        candidates = [(target, reason) for target, ok, reason in checked if ok and _target_depth(target) is not None]
        if not candidates:
            reasons = "; ".join(f"Q{target.get('index')} {reason}" for target, _ok, reason in checked[:8])
            raise RuntimeError(f"目前沒有高於平面門檻的圓形堆疊硬幣：{reasons}")
        candidates.sort(key=lambda item: int(item[0].get("index", 9999)))
        candidate, reason = candidates[0]
        print(
            f"[unstack] selected Q{candidate.get('index')}: q_index "
            f"height={_stack_height_mm(candidate, data.get('targets', []), float(cfg.get('unstack_stack_neighbor_distance_mm', 60.0))):.2f}mm "
            f"depth={_target_depth(candidate):.1f} visible={float(candidate.get('visible_score', candidate.get('top_visible_score')) or 0.0):.2f} "
            f"score={float(candidate.get('top_coin_score') or 0.0):.2f}"
        )
        return data, candidate
    if priority in ("round_highest_above_plane", "round_highest", "above_plane"):
        checked = []
        for target in targets:
            ok, reason = is_stacked_top_target(target, data.get("targets", []), cfg)
            checked.append((target, ok, reason))
            print(f"[unstack] above-plane check Q{target.get('index')}: {reason}")
        candidates = [(target, reason) for target, ok, reason in checked if ok and _target_depth(target) is not None]
        if not candidates:
            reasons = "; ".join(f"Q{target.get('index')} {reason}" for target, _ok, reason in checked[:8])
            raise RuntimeError(f"目前沒有高於平面門檻的圓形堆疊硬幣：{reasons}")
        candidates.sort(
            key=lambda item: (
                -_stack_height_mm(
                    item[0],
                    data.get("targets", []),
                    float(cfg.get("unstack_stack_neighbor_distance_mm", 60.0)),
                ),
                _target_depth(item[0]) if _target_depth(item[0]) is not None else 9999.0,
                -float(item[0].get("visible_score", item[0].get("top_visible_score")) or 0.0),
                -float(item[0].get("top_coin_score") or 0.0),
            )
        )
        candidate, reason = candidates[0]
        print(
            f"[unstack] selected Q{candidate.get('index')}: round_highest "
            f"height={_stack_height_mm(candidate, data.get('targets', []), float(cfg.get('unstack_stack_neighbor_distance_mm', 60.0))):.2f}mm "
            f"depth={_target_depth(candidate):.1f} visible={float(candidate.get('visible_score', candidate.get('top_visible_score')) or 0.0):.2f} "
            f"score={float(candidate.get('top_coin_score') or 0.0):.2f}"
        )
        return data, candidate
    if priority in ("highest_depth", "highest", "depth"):
        depth_targets = [t for t in targets if _target_depth(t) is not None]
        if not depth_targets:
            raise RuntimeError("目前沒有可用深度的硬幣，無法依最高優先夾取")
        depth_targets.sort(
            key=lambda t: (
                _target_depth(t) if _target_depth(t) is not None else 9999.0,
                -float(t.get("top_coin_score") or 0.0),
                -float(t.get("visible_score", t.get("top_visible_score")) or 0.0),
            )
        )
        for target in depth_targets[:8]:
            print(
                f"[unstack] highest-depth candidate Q{target.get('index')}: "
                f"depth={_target_depth(target):.1f} score={float(target.get('top_coin_score') or 0.0):.2f} "
                f"visible={float(target.get('visible_score', target.get('top_visible_score')) or 0.0):.2f}"
            )
        candidate = depth_targets[0]
        print(f"[unstack] selected Q{candidate.get('index')}: highest_depth={_target_depth(candidate):.1f}mm")
        return data, candidate
    checked = []
    for target in targets:
        ok, reason = is_stacked_top_target(target, data.get("targets", []), cfg)
        checked.append((target, ok, reason))
        print(f"[unstack] stacked check Q{target.get('index')}: {reason}")
    stacked = [(target, reason) for target, ok, reason in checked if ok]
    if not stacked:
        reasons = "; ".join(f"Q{target.get('index')} {reason}" for target, _ok, reason in checked[:8])
        raise RuntimeError(f"目前沒有需要拆堆的上層硬幣：{reasons}")
    stacked.sort(
        key=lambda item: (
            _target_depth(item[0]) if _target_depth(item[0]) is not None else 9999.0,
            -_stack_height_mm(
                item[0],
                data.get("targets", []),
                float(cfg.get("unstack_stack_neighbor_distance_mm", 60.0)),
            ),
            -float(item[0].get("top_coin_score") or 0.0),
            -float(item[0].get("visible_score", item[0].get("top_visible_score")) or 0.0),
        )
    )
    candidate, reason = stacked[0]
    print(f"[unstack] selected Q{candidate.get('index')}: {reason}")
    return data, candidate


def move_to_target(robot, target, travel_z, safe_z):
    x = float(target["robot_x_mm"])
    y = float(target["robot_y_mm"])
    check_bounds(x, y, travel_z)
    check_bounds(x, y, safe_z)
    print(f"[unstack] move above Q{target.get('index')} X={x:.2f} Y={y:.2f}")
    if not move_to_pose(robot, x, y, travel_z):
        return False
    if abs(travel_z - safe_z) > 0.5:
        return robot.movl(x, y, safe_z)
    return True


def lower_pick_lift(robot, target, pick_z, safe_z, lower_speed, move_speed, do_index, dry_run, cfg=None):
    x = float(target["robot_x_mm"])
    y = float(target["robot_y_mm"])
    cfg = cfg or {}
    z_mode = str(cfg.get("unstack_pick_z_mode", "height_above_table")).lower()
    z = float(pick_z)
    if z_mode in ("height_above_table", "table_height", "dynamic_height"):
        height_values = []
        for key in ("height_above_table_mm", "top_surface_height_mm"):
            value = target.get(key)
            if value not in (None, ""):
                try:
                    height_values.append(float(value))
                except Exception:
                    pass
        max_reasonable_height = float(cfg.get("unstack_max_reasonable_height_mm", 25.0))
        height_values = [h for h in height_values if 0.0 <= h <= max_reasonable_height]
        height = min(height_values) if height_values else None
        clamp_height = cfg.get("unstack_pick_height_clamp_mm")
        if height is not None and clamp_height not in (None, ""):
            height = min(height, max(0.0, float(clamp_height)))
        if height not in (None, ""):
            z = (
                float(cfg.get("robot_table_z_mm", -160.0))
                + float(height)
                + float(cfg.get("unstack_pick_touch_offset_mm", -0.8))
            )
        elif _cfg_bool(cfg, "unstack_use_target_robot_z", False) and target.get("z_offset_ready") and target.get("robot_z_mm") is not None:
            z = float(target["robot_z_mm"])
    elif _cfg_bool(cfg, "unstack_use_target_robot_z", False) and target.get("z_offset_ready") and target.get("robot_z_mm") is not None:
        z = float(target["robot_z_mm"])
    check_lower_bounds(z)
    print(
        f"[unstack] lower pick Z={z:.2f} mode={z_mode} "
        f"height={target.get('height_above_table_mm')} top_surface={target.get('top_surface_height_mm')} "
        f"touch_offset={cfg.get('unstack_pick_touch_offset_mm', -0.8)} "
        f"dry_run={dry_run}"
    )
    robot.set_speed(lower_speed)
    pick_tol = max(0.1, float(cfg.get("unstack_pick_reach_tol_mm", 0.35)))
    if not robot.movl(
        x,
        y,
        z,
        timeout_s=float(cfg.get("unstack_pick_lower_timeout_s", 20.0)),
        tol_mm=pick_tol,
        speed_l=int(cfg.get("robot_lower_speed_l_pct", lower_speed)),
        acc_l=int(cfg.get("robot_lower_acc_l_pct", 25)),
    ):
        return False
    settle_sec = max(0.0, float(cfg.get("unstack_pick_settle_sec", 0.15)))
    if settle_sec:
        time.sleep(settle_sec)
    if dry_run:
        print("[unstack] dry-run: skip DO ON")
    else:
        print(f"[unstack] DO{do_index}=ON")
        robot.set_do(do_index, 1)
        time.sleep(max(0.0, float(cfg.get("unstack_vacuum_dwell_sec", 0.45))))
    robot.set_speed(move_speed)
    return robot.movl(x, y, safe_z)


def suction_off(robot, do_index):
    if do_index is None:
        return
    try:
        print(f"[unstack] DO{do_index}=OFF (safety)")
        robot.set_do(do_index, 0)
    except Exception as exc:
        print(f"[unstack] DO{do_index}=OFF failed: {exc}")


def place_to_slot(robot, slot, travel_z, safe_z, move_speed, lower_speed, do_index, dry_run, cfg=None):
    cfg = cfg or {}
    x = float(slot["x"])
    y = float(slot["y"])
    z = float(slot["z"])
    check_bounds(x, y, travel_z)
    check_bounds(x, y, safe_z)
    check_lower_bounds(z)
    print(f"[unstack] place {slot['name']} X={x:.2f} Y={y:.2f} Z={z:.2f}")
    robot.set_speed(move_speed)
    if not move_to_pose(robot, x, y, travel_z):
        return False
    if abs(travel_z - safe_z) > 0.5 and not robot.movl(x, y, safe_z):
        return False
    robot.set_speed(lower_speed)
    if not robot.movl(
        x,
        y,
        z,
        speed_l=int(cfg.get("robot_lower_speed_l_pct", lower_speed)),
        acc_l=int(cfg.get("robot_lower_acc_l_pct", 25)),
    ):
        return False
    if dry_run:
        print("[unstack] dry-run: skip DO OFF")
    else:
        print(f"[unstack] DO{do_index}=OFF")
        robot.set_do(do_index, 0)
        time.sleep(0.25)
    robot.set_speed(move_speed)
    return robot.movl(x, y, safe_z)


def refresh_targets_with_retry(attempt_label, retries=4, settle_sec=1.2):
    last_exc = None
    for attempt in range(1, int(retries) + 1):
        try:
            release_vision_processes()
            time.sleep(float(settle_sec))
            refresh_targets_after_start()
            return
        except Exception as exc:
            last_exc = exc
            print(f"[unstack] {attempt_label} 辨識失敗 {attempt}/{retries}: {exc}")
            release_vision_processes()
            time.sleep(float(settle_sec) * attempt)
    raise RuntimeError(f"{attempt_label} 辨識重試失敗：{last_exc}")


class PersistentYoloTargetRefresher:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.detect_cfg = dict(cfg)
        self.source_roi_locked = False
        from dual_camera_live import (
            Gemini2Camera,
            apply_color_controls,
            attach_gemini_depth_to_quality_ellipses,
            attach_quality_robot_table_diameter_fallback,
            attach_robot_coords_to_quality_ellipses,
            average_depth_stack,
            blur_score,
            classify_quality_ellipses,
            detect_quality_ellipses,
            filter_quality_diameter_candidates,
            load_calib,
            load_quality_to_gemini_homography,
            load_robot_calibration,
            load_stereo_extrinsics,
            median_quality_ellipses,
            open_quality_camera,
            rank_top_quality_coins,
            set_gemini_stream_env,
            summarize_quality_coins,
            write_robot_targets,
        )

        self._api = {
            "apply_color_controls": apply_color_controls,
            "attach_gemini_depth_to_quality_ellipses": attach_gemini_depth_to_quality_ellipses,
            "attach_quality_robot_table_diameter_fallback": attach_quality_robot_table_diameter_fallback,
            "attach_robot_coords_to_quality_ellipses": attach_robot_coords_to_quality_ellipses,
            "average_depth_stack": average_depth_stack,
            "blur_score": blur_score,
            "classify_quality_ellipses": classify_quality_ellipses,
            "detect_quality_ellipses": detect_quality_ellipses,
            "filter_quality_diameter_candidates": filter_quality_diameter_candidates,
            "median_quality_ellipses": median_quality_ellipses,
            "rank_top_quality_coins": rank_top_quality_coins,
            "summarize_quality_coins": summarize_quality_coins,
            "write_robot_targets": write_robot_targets,
        }
        self.calib = load_calib(self.cfg)
        self.quality_to_gemini_h = load_quality_to_gemini_homography()
        self.stereo_extrinsics = load_stereo_extrinsics()
        self.robot_calib = load_robot_calibration(self.cfg)
        set_gemini_stream_env(self.cfg)
        self.qcap, self.q_status = open_quality_camera(self.cfg)
        if self.qcap is None:
            raise RuntimeError(f"畫質相機常駐開啟失敗：{self.q_status}")
        self.camera = Gemini2Camera(align_depth_to_color=True)
        self.camera.open()
        apply_color_controls(self.camera, self.cfg)
        self.intr = self.camera.intrinsics
        print("[unstack-yolo] 常駐 YOLO/相機已啟動")

    def close(self):
        try:
            if self.qcap is not None:
                self.qcap.release()
        except Exception:
            pass
        try:
            if self.camera is not None:
                self.camera.close()
        except Exception:
            pass

    def lock_source_roi_from_targets(self, targets):
        if self.source_roi_locked or not _cfg_bool(self.cfg, "unstack_lock_source_roi_enabled", True):
            return
        points = []
        for target in targets or []:
            try:
                qx = float(target["quality_x_px"])
                qy = float(target["quality_y_px"])
            except Exception:
                continue
            points.append((qx, qy))
        if len(points) < 2:
            print("[unstack-yolo] source ROI not locked: targets too few")
            return
        margin = float(self.cfg.get("unstack_source_roi_margin_px", 140.0))
        width = float(self.cfg.get("quality_width", 3840))
        height = float(self.cfg.get("quality_height", 2160))
        if self.cfg.get("quality_roi"):
            base_x1, base_y1, base_x2, base_y2 = [float(v) for v in self.cfg["quality_roi"]]
        else:
            base_x1, base_y1, base_x2, base_y2 = 0.0, 0.0, width, height
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        roi = [
            int(max(base_x1, min(xs) - margin)),
            int(max(base_y1, min(ys) - margin)),
            int(min(base_x2, max(xs) + margin)),
            int(min(base_y2, max(ys) + margin)),
        ]
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            print(f"[unstack-yolo] source ROI not locked: invalid {roi}")
            return
        self.detect_cfg["quality_roi"] = roi
        self.source_roi_locked = True
        print(f"[unstack-yolo] source ROI locked {roi} from {len(points)} targets")

    def unlock_source_roi(self, reason):
        if not self.source_roi_locked:
            return
        self.detect_cfg["quality_roi"] = self.cfg.get("quality_roi")
        self.source_roi_locked = False
        print(f"[unstack-yolo] source ROI unlocked: {reason}")

    def refresh(self, attempt_label):
        t0 = time.perf_counter()
        settle_sec = max(0.0, float(self.cfg.get("unstack_yolo_persistent_settle_sec", 0.35)))
        if settle_sec:
            time.sleep(settle_sec)

        n_depth = max(1, int(self.cfg.get("quality_yolo_fast_depth_frames", 3)))
        n_quality = max(1, int(self.cfg.get("quality_yolo_fast_save_once_frames", 1)))
        depth_frames = []
        q_frames = []
        max_reads = max(n_depth, n_quality) + 8
        for _ in range(max_reads):
            if len(depth_frames) < n_depth:
                color, depth = self.camera.get_frames(timeout_ms=600)
                if color is not None and depth is not None:
                    depth_frames.append(depth.astype(np.float32))
            if len(q_frames) < n_quality:
                ok, q_read = self.qcap.read()
                if ok and q_read is not None:
                    q_frames.append(q_read.copy())
            if len(depth_frames) >= n_depth and len(q_frames) >= n_quality:
                break
            time.sleep(0.02)

        if not depth_frames:
            raise RuntimeError(f"{attempt_label} 常駐 YOLO：Gemini 沒有深度影像")
        if not q_frames:
            raise RuntimeError(f"{attempt_label} 常駐 YOLO：畫質相機沒有影像")

        depth_avg = self._api["average_depth_stack"](depth_frames)
        q_score, _q_frame = max(
            ((self._api["blur_score"](frame, self.detect_cfg.get("quality_roi")), frame) for frame in q_frames),
            key=lambda item: item[0],
        )
        q_runs = [self._api["detect_quality_ellipses"](frame, self.detect_cfg) for frame in q_frames]
        q_ellipses = self._api["median_quality_ellipses"](q_runs, min_hits=1, cfg=self.detect_cfg)
        if self.source_roi_locked and not q_ellipses:
            self.unlock_source_roi("locked ROI returned 0 targets; retry full ROI")
            q_score, _q_frame = max(
                ((self._api["blur_score"](frame, self.detect_cfg.get("quality_roi")), frame) for frame in q_frames),
                key=lambda item: item[0],
            )
            q_runs = [self._api["detect_quality_ellipses"](frame, self.detect_cfg) for frame in q_frames]
            q_ellipses = self._api["median_quality_ellipses"](q_runs, min_hits=1, cfg=self.detect_cfg)
        q_ellipses = self._api["attach_gemini_depth_to_quality_ellipses"](
            q_ellipses,
            self.quality_to_gemini_h,
            depth_avg,
            self.intr,
            self.stereo_extrinsics,
        )
        q_ellipses = self._api["attach_quality_robot_table_diameter_fallback"](q_ellipses, self.robot_calib)
        q_ellipses = self._api["filter_quality_diameter_candidates"](q_ellipses, self.detect_cfg)
        q_ellipses = self._api["classify_quality_ellipses"](q_ellipses, self.calib, self.detect_cfg)
        q_ellipses = self._api["attach_robot_coords_to_quality_ellipses"](q_ellipses, self.intr, self.robot_calib)
        q_ellipses = self._api["rank_top_quality_coins"](q_ellipses, self.detect_cfg)
        counts, total_value, _ = self._api["summarize_quality_coins"](q_ellipses, self.detect_cfg)
        self._api["write_robot_targets"](q_ellipses, counts, total_value, self.cfg)
        elapsed = time.perf_counter() - t0
        print(
            f"[unstack-yolo] {attempt_label}: targets={len(q_ellipses)} "
            f"quality_blur={q_score:.1f} time={elapsed:.2f}s"
        )


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-picks", type=int, default=int(cfg.get("unstack_max_picks", 12)))
    ap.add_argument("--safe-z", type=float, default=float(cfg.get("unstack_safe_z_mm", DEFAULT_SAFE_Z)))
    ap.add_argument("--travel-z", type=float, default=float(cfg.get("unstack_travel_z_mm", DEFAULT_TRAVEL_Z)))
    ap.add_argument("--pick-z", type=float, default=float(cfg.get("unstack_pick_z_mm", DEFAULT_PICK_Z)))
    ap.add_argument("--move-speed", type=int, default=int(cfg.get("ui_move_speed", 40)))
    ap.add_argument("--lower-speed", type=int, default=int(cfg.get("ui_lower_speed", 25)))
    ap.add_argument("--move-speed-j", type=int, default=int(cfg.get("robot_move_speed_j_pct", 80)))
    ap.add_argument("--move-acc-j", type=int, default=int(cfg.get("robot_move_acc_j_pct", 70)))
    ap.add_argument("--move-speed-l", type=int, default=int(cfg.get("robot_move_speed_l_pct", 70)))
    ap.add_argument("--move-acc-l", type=int, default=int(cfg.get("robot_move_acc_l_pct", 60)))
    ap.add_argument("--lower-speed-l", type=int, default=int(cfg.get("robot_lower_speed_l_pct", 35)))
    ap.add_argument("--lower-acc-l", type=int, default=int(cfg.get("robot_lower_acc_l_pct", 25)))
    ap.add_argument("--start-pose", type=parse_xyz, default=default_start_pose(cfg))
    ap.add_argument("--real-pick", action="store_true", help="啟用真空/夾爪 DO；需 config 明確允許")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    max_picks = max(1, int(args.max_picks))
    dry_run = not args.real_pick
    if dry_run:
        max_picks = min(max_picks, max(1, int(cfg.get("unstack_dry_run_max_picks", 1))))
    do_index = validate_real_pick_config(cfg, args.real_pick)
    use_auto_empty = _cfg_bool(cfg, "unstack_auto_empty_place_enabled", True)
    slots = [] if use_auto_empty else fixed_staging_slots(cfg, max_picks)
    robot_to_gemini_h = None
    gemini_to_quality_h = None
    if use_auto_empty:
        robot_to_gemini_h, gemini_to_quality_h = load_homographies(cfg)

    bounds = safe_bounds(cfg)
    if float(args.travel_z) > bounds["z_max"]:
        print(f"[unstack] travel_z={float(args.travel_z):.2f} 超過安全上限 {bounds['z_max']:.2f}，改用安全上限")
        args.travel_z = bounds["z_max"]
    if float(args.safe_z) > bounds["z_max"]:
        print(f"[unstack] safe_z={float(args.safe_z):.2f} 超過安全上限 {bounds['z_max']:.2f}，改用安全上限")
        args.safe_z = bounds["z_max"]
    print(f"[unstack] safe bounds={bounds}")
    print(f"[unstack] max_picks={max_picks} dry_run={dry_run}")
    if dry_run:
        print("[unstack] 目前是 dry-run：只會走位/下降，不切 DO，不會真的搬硬幣")
    if not args.yes:
        ans = input("確認開始拆堆？輸入 y 後 Enter: ").strip().lower()
        if ans not in ("y", "yes"):
            print("[unstack] 已取消")
            return

    robot = MG400()
    last_target = None
    last_attempt = None
    picked = 0
    placed_slots = []
    processed_sources = []
    blocked_place_slots = []
    stop_reason = None
    persistent_vision = None
    try:
        release_vision_processes()
        robot.connect()
        robot.enable()
        robot.set_speed(max(1, min(100, int(args.move_speed))))
        cfg["robot_move_speed_j_pct"] = max(1, min(100, int(args.move_speed_j)))
        cfg["robot_move_acc_j_pct"] = max(1, min(100, int(args.move_acc_j)))
        cfg["robot_move_speed_l_pct"] = max(1, min(100, int(args.move_speed_l)))
        cfg["robot_move_acc_l_pct"] = max(1, min(100, int(args.move_acc_l)))
        cfg["robot_lower_speed_l_pct"] = max(1, min(100, int(args.lower_speed_l)))
        cfg["robot_lower_acc_l_pct"] = max(1, min(100, int(args.lower_acc_l)))
        robot.set_motion_profile(
            speed_j=cfg["robot_move_speed_j_pct"],
            acc_j=cfg["robot_move_acc_j_pct"],
            speed_l=cfg["robot_move_speed_l_pct"],
            acc_l=cfg["robot_move_acc_l_pct"],
        )
        print(
            "[robot-speed] "
            f"MovJ SpeedJ={cfg['robot_move_speed_j_pct']} AccJ={cfg['robot_move_acc_j_pct']} "
            f"MovL SpeedL={cfg['robot_move_speed_l_pct']} AccL={cfg['robot_move_acc_l_pct']} "
            f"Lower SpeedL={cfg['robot_lower_speed_l_pct']} AccL={cfg['robot_lower_acc_l_pct']}"
        )
        sx, sy, sz = [float(v) for v in args.start_pose]
        if not move_to_pose(robot, sx, sy, sz):
            raise RuntimeError("移動到相機避讓位置失敗")
        method = str(cfg.get("quality_detection_method", "")).lower()
        use_persistent_yolo = (
            method in ("yolo", "yolov8", "yolov8m")
            and _cfg_bool(cfg, "unstack_yolo_persistent_enabled", True)
        )
        if use_persistent_yolo:
            try:
                persistent_vision = PersistentYoloTargetRefresher(cfg)
            except Exception as exc:
                persistent_vision = None
                print(f"[unstack-yolo] 常駐辨識啟動失敗，改用舊式重新辨識：{exc}")

        def refresh_current_targets(label):
            if persistent_vision is not None:
                persistent_vision.refresh(label)
            else:
                refresh_targets_with_retry(label)

        for i in range(max_picks):
            write_action_status("running", None, f"unstack refresh {i + 1}/{max_picks}")
            refresh_current_targets(f"unstack refresh {i + 1}/{max_picks}")
            if i == 0 and persistent_vision is not None:
                try:
                    first_data, _first_valid = load_targets()
                    persistent_vision.lock_source_roi_from_targets(first_data.get("targets", []))
                except Exception as exc:
                    print(f"[unstack-yolo] source ROI lock failed: {exc}")
            try:
                data, target = select_top_target(cfg, processed_sources)
            except RuntimeError as exc:
                last_select_exc = exc
                recovered = False
                if processed_sources and _cfg_bool(cfg, "unstack_retry_ignore_processed_sources", True):
                    print(f"[unstack] no candidate with processed-source skip; retry without source skip: {last_select_exc}")
                    try:
                        data, target = select_top_target(cfg, [])
                        recovered = True
                        processed_sources = []
                    except RuntimeError as retry_exc:
                        last_select_exc = retry_exc
                if (
                    not recovered
                    and
                    persistent_vision is not None
                    and getattr(persistent_vision, "source_roi_locked", False)
                    and _cfg_bool(cfg, "unstack_unlock_source_roi_on_no_candidate", True)
                ):
                    print(f"[unstack-yolo] no candidate in locked source ROI; unlock and retry full ROI: {last_select_exc}")
                    persistent_vision.unlock_source_roi("no candidate in locked ROI")
                    refresh_current_targets("unstack no-candidate full-ROI retry")
                    try:
                        data, target = select_top_target(cfg, processed_sources)
                        recovered = True
                    except RuntimeError as retry_exc:
                        last_select_exc = retry_exc
                if recovered:
                    pass
                else:
                    retries = max(0, int(cfg.get("unstack_no_candidate_retries", 2)))
                    retry_delay = max(0.0, float(cfg.get("unstack_no_candidate_retry_delay_sec", 0.8)))
                    for retry in range(1, retries + 1):
                        print(f"[unstack] 暫時沒有候選，重新確認 {retry}/{retries}: {last_select_exc}")
                        if retry_delay:
                            time.sleep(retry_delay)
                        refresh_current_targets(f"unstack no-candidate confirm {retry}/{retries}")
                        try:
                            data, target = select_top_target(cfg, processed_sources)
                            recovered = True
                            break
                        except RuntimeError as retry_exc:
                            last_select_exc = retry_exc
                if not recovered:
                    stop_reason = f"沒有更多上層可取硬幣：{last_select_exc}"
                    print(f"[unstack] {stop_reason}")
                    break
            failed_slots = []
            if use_auto_empty:
                try:
                    slot = next_empty_slot(cfg, data, placed_slots, blocked_place_slots + failed_slots, robot_to_gemini_h, gemini_to_quality_h)
                except RuntimeError as exc:
                    stop_reason = f"沒有更多畫質相機可見空位：{exc}"
                    print(f"[unstack] {stop_reason}")
                    break
                print(
                    f"[unstack] auto empty slot {slot['name']} X={slot['x']:.2f} Y={slot['y']:.2f} "
                    f"Q=({slot.get('quality_x_px')},{slot.get('quality_y_px')}) "
                    f"nearest={slot['nearest_occupied_mm']:.1f}mm"
                )
            else:
                slot = slots[i]
            last_target = target
            write_action_status("travel", target, f"unstack pick {i + 1}/{max_picks}")
            last_attempt = (float(target["robot_x_mm"]), float(target["robot_y_mm"]), float(args.travel_z))
            if not move_to_target(robot, target, float(args.travel_z), float(args.safe_z)):
                raise RuntimeError("移到目標上方失敗")
            if not lower_pick_lift(
                robot,
                target,
                float(args.pick_z),
                float(args.safe_z),
                max(1, min(100, int(args.lower_speed))),
                max(1, min(100, int(args.move_speed))),
                do_index,
                dry_run,
                cfg,
            ):
                raise RuntimeError("下降吸取/抬起失敗")
            place_ok = False
            max_place_attempts = max(1, int(cfg.get("unstack_place_retry_slots", 12)))
            for place_attempt in range(1, max_place_attempts + 1):
                last_attempt = (float(slot["x"]), float(slot["y"]), float(args.travel_z))
                write_action_status("place", target, f"unstack place {i + 1}/{max_picks} try {place_attempt}/{max_place_attempts}")
                place_ok = place_to_slot(
                    robot,
                    slot,
                    float(args.travel_z),
                    float(args.safe_z),
                    max(1, min(100, int(args.move_speed))),
                    max(1, min(100, int(args.lower_speed))),
                    do_index,
                    dry_run,
                    cfg,
                )
                if place_ok:
                    break
                failed_slots.append(slot)
                blocked_place_slots.append(slot)
                print(
                    f"[unstack] slot {slot['name']} X={slot['x']:.2f} Y={slot['y']:.2f} 不可達，改找下一個空位"
                )
                try:
                    robot.clear_error()
                    robot.enable()
                    robot.set_speed(max(1, min(100, int(args.move_speed))))
                except Exception as exc:
                    print(f"[unstack] 清除錯誤後無法繼續找空位：{exc}")
                    break
                if not use_auto_empty:
                    break
                try:
                    slot = next_empty_slot(cfg, data, placed_slots, blocked_place_slots + failed_slots, robot_to_gemini_h, gemini_to_quality_h)
                    print(
                        f"[unstack] next empty slot {slot['name']} X={slot['x']:.2f} Y={slot['y']:.2f} "
                        f"Q=({slot.get('quality_x_px')},{slot.get('quality_y_px')}) "
                        f"nearest={slot['nearest_occupied_mm']:.1f}mm"
                    )
                except RuntimeError as exc:
                    print(f"[unstack] 找不到下一個空位：{exc}")
                    break
            if not place_ok:
                if not dry_run:
                    suction_off(robot, do_index)
                raise RuntimeError("移到暫放區/釋放失敗")
            picked += 1
            placed_slots.append(slot)
            processed_sources.append((
                float(target["robot_x_mm"]),
                float(target["robot_y_mm"]),
                float(cfg.get("unstack_skip_picked_source_radius_mm", 8.0)),
            ))
            processed_sources.append((
                float(slot["x"]),
                float(slot["y"]),
                float(cfg.get("unstack_skip_placed_slot_radius_mm", 32.0)),
            ))
            print(
                f"[unstack] mark source Q{target.get('index')} as processed; "
                f"source_skip={float(cfg.get('unstack_skip_picked_source_radius_mm', 8.0)):.1f}mm "
                f"placed_skip={float(cfg.get('unstack_skip_placed_slot_radius_mm', 32.0)):.1f}mm"
            )
            if _cfg_bool(cfg, "unstack_return_to_start_each_pick", False):
                if not move_to_pose(robot, sx, sy, sz):
                    raise RuntimeError("返回相機避讓位置失敗")
            else:
                print("[unstack] skip return-to-start; next recognition uses locked source ROI")

        write_action_status("running", None, "final recognition after unstack")
        refresh_current_targets("final recognition after unstack")
        done_message = f"unstack completed, picked={picked}"
        if stop_reason:
            done_message += f"; stop_reason={stop_reason}"
        write_action_status("done", None, done_message)
        print(f"[unstack] 完成，已處理 {picked} 顆；已重新辨識暫放區")
        if stop_reason:
            print(f"[unstack] 停止原因：{stop_reason}")
    except Exception as exc:
        error_code = None
        try:
            errs = getattr(robot, "last_errors", None) or robot.get_errors()
            error_code = errs[0] if errs else None
        except Exception:
            error_code = None
        write_action_status(
            "failed",
            last_target,
            str(exc),
            error_code=error_code,
            robot_xyz=last_attempt,
            controller_response=getattr(robot, "last_response", str(exc)),
        )
        print(f"[unstack] 失敗：{exc}")
        sys.exit(2)
    finally:
        if persistent_vision is not None:
            try:
                persistent_vision.close()
            except Exception:
                pass
        if do_index is not None:
            try:
                robot.set_do(do_index, 0)
            except Exception:
                pass
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
