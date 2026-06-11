# -*- coding: utf-8 -*-
"""
Interactive pick-offset calibration helper for the MG400 UI.

The UI calls this script for small, safe operations:
  start: move above a selected detected coin
  jog:   nudge the robot at the current Z
  teach: save the current pose as a pick-offset sample
  reset: clear samples and reset offset to zero
"""

import argparse
import datetime
import json
import sys
from pathlib import Path


HERE = Path(__file__).parent
TARGETS_FILE = HERE / "robot_targets.json"
CONFIG_FILE = HERE / "dual_camera_config.json"
SAMPLES_FILE = HERE / "robot_pick_offset_samples.json"

SAFE_X_MIN = 100.0
SAFE_X_MAX = 390.0
SAFE_Y_MIN = -310.0
SAFE_Y_MAX = 220.0
SAFE_Z_MIN = -162.0
SAFE_Z_MAX = 180.0
LIFT_X_MIN = -50.0
LIFT_X_MAX = 450.0
LIFT_Y_MIN = -330.0
LIFT_Y_MAX = 330.0
DEFAULT_SAFE_Z = -155.0
DEFAULT_PRE_XY_LIFT_MM = 20.0


from core.robot import MG400


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config():
    return load_json(CONFIG_FILE, {})


def save_config(cfg):
    write_json(CONFIG_FILE, cfg)


def safe_bounds(cfg=None):
    cfg = cfg or load_config()
    return {
        "x_min": float(cfg.get("robot_safe_x_min_mm", SAFE_X_MIN)),
        "x_max": float(cfg.get("robot_safe_x_max_mm", SAFE_X_MAX)),
        "y_min": float(cfg.get("robot_safe_y_min_mm", SAFE_Y_MIN)),
        "y_max": float(cfg.get("robot_safe_y_max_mm", SAFE_Y_MAX)),
        "z_min": float(cfg.get("robot_safe_z_min_mm", SAFE_Z_MIN)),
        "z_max": float(cfg.get("robot_safe_z_max_mm", SAFE_Z_MAX)),
    }


def lift_bounds(cfg=None):
    cfg = cfg or load_config()
    return {
        "x_min": float(cfg.get("robot_lift_x_min_mm", cfg.get("robot_auto_x_min_mm", LIFT_X_MIN))),
        "x_max": float(cfg.get("robot_lift_x_max_mm", cfg.get("robot_auto_x_max_mm", LIFT_X_MAX))),
        "y_min": float(cfg.get("robot_lift_y_min_mm", cfg.get("robot_auto_y_min_mm", LIFT_Y_MIN))),
        "y_max": float(cfg.get("robot_lift_y_max_mm", cfg.get("robot_auto_y_max_mm", LIFT_Y_MAX))),
    }


def load_targets():
    data = load_json(TARGETS_FILE, {})
    targets = data.get("targets", [])
    if not targets:
        raise RuntimeError("目前沒有硬幣目標，請先重新辨識。")
    return data, targets


def selected_target(index=None):
    _data, targets = load_targets()
    if index is not None:
        for t in targets:
            if int(t.get("index", -1)) == int(index):
                return t
        raise RuntimeError(f"找不到 Q{index}，請先重新辨識。")
    for t in targets:
        if t.get("valid_for_pick"):
            return t
    raise RuntimeError("目前沒有 OK 可取硬幣，請換一顆或重新辨識。")


def current_offset(cfg):
    return (
        float(cfg.get("robot_target_offset_x_mm", 0.0)),
        float(cfg.get("robot_target_offset_y_mm", 0.0)),
        float(cfg.get("robot_target_offset_z_mm", 0.0)),
    )


def load_offset_samples():
    data = load_json(SAMPLES_FILE, {"samples": []})
    samples = []
    for sample in data.get("samples", []):
        try:
            item = {
                "raw_x": float(sample["raw_robot_x_mm"]),
                "raw_y": float(sample["raw_robot_y_mm"]),
                "offset_x": float(sample["offset_x_mm"]),
                "offset_y": float(sample["offset_y_mm"]),
            }
            if "raw_robot_z_mm" in sample and "offset_z_mm" in sample:
                item["raw_z"] = float(sample["raw_robot_z_mm"])
                item["offset_z"] = float(sample["offset_z_mm"])
            samples.append(item)
        except Exception:
            continue
    return samples


def offset_for_xyz(raw_x, raw_y, raw_z, cfg, samples):
    global_x, global_y, global_z = current_offset(cfg)
    if not cfg.get("robot_local_offset_enabled", True) or not samples:
        return global_x, global_y, global_z, "global", None
    radius = max(1.0, float(cfg.get("robot_local_offset_radius_mm", 180.0)))
    neighbors = max(1, int(cfg.get("robot_local_offset_neighbors", 5)))
    ranked = []
    for sample in samples:
        dx = float(raw_x) - sample["raw_x"]
        dy = float(raw_y) - sample["raw_y"]
        dist = (dx * dx + dy * dy) ** 0.5
        ranked.append((dist, sample))
    ranked.sort(key=lambda item: item[0])
    if not ranked or ranked[0][0] > radius:
        return global_x, global_y, global_z, "global-far", None if not ranked else round(ranked[0][0], 3)
    picked = [item for item in ranked if item[0] <= radius][:neighbors]
    if picked[0][0] <= 0.5:
        sample = picked[0][1]
        return sample["offset_x"], sample["offset_y"], sample.get("offset_z", global_z), "local-exact", round(picked[0][0], 3)
    bias_mm = 18.0
    total_w = 0.0
    sum_x = 0.0
    sum_y = 0.0
    total_w_z = 0.0
    sum_z = 0.0
    for dist, sample in picked:
        w = 1.0 / ((dist + bias_mm) ** 2)
        total_w += w
        sum_x += w * sample["offset_x"]
        sum_y += w * sample["offset_y"]
        if "offset_z" in sample:
            total_w_z += w
            sum_z += w * sample["offset_z"]
    if total_w <= 0:
        return global_x, global_y, global_z, "global", None
    off_z = sum_z / total_w_z if total_w_z > 0 else global_z
    return sum_x / total_w, sum_y / total_w, off_z, "local", round(ranked[0][0], 3)


def update_current_targets(new_x, new_y, old_x, old_y):
    if not TARGETS_FILE.exists():
        return
    cfg = load_config()
    samples = load_offset_samples()
    data = load_json(TARGETS_FILE, {})
    for t in data.get("targets", []):
        raw_x = t.get("raw_robot_x_mm")
        raw_y = t.get("raw_robot_y_mm")
        if raw_x is None and t.get("robot_x_mm") is not None:
            raw_x = float(t["robot_x_mm"]) - float(old_x)
            t["raw_robot_x_mm"] = round(raw_x, 3)
        if raw_y is None and t.get("robot_y_mm") is not None:
            raw_y = float(t["robot_y_mm"]) - float(old_y)
            t["raw_robot_y_mm"] = round(raw_y, 3)
        if raw_x is not None and raw_y is not None:
            raw_z = t.get("raw_robot_z_mm")
            if raw_z is None and t.get("robot_z_mm") is not None:
                raw_z = float(t["robot_z_mm"]) - float(cfg.get("robot_target_offset_z_mm", 0.0))
                t["raw_robot_z_mm"] = round(raw_z, 3)
            off_x, off_y, off_z, method, dist = offset_for_xyz(raw_x, raw_y, raw_z, cfg, samples)
            t["robot_x_mm"] = round(float(raw_x) + float(off_x), 3)
            t["robot_y_mm"] = round(float(raw_y) + float(off_y), 3)
            if raw_z is not None:
                t["robot_z_mm"] = round(float(raw_z) + float(off_z), 3)
            t["applied_offset_x_mm"] = round(float(off_x), 3)
            t["applied_offset_y_mm"] = round(float(off_y), 3)
            t["applied_offset_z_mm"] = round(float(off_z), 3)
            t["z_offset_ready"] = any("offset_z" in s for s in samples)
            t["offset_method"] = method
            t["nearest_offset_sample_dist_mm"] = dist
    write_json(TARGETS_FILE, data)


def set_offset(new_x, new_y, new_z=None):
    cfg = load_config()
    old_x, old_y, old_z = current_offset(cfg)
    cfg["robot_target_offset_x_mm"] = round(float(new_x), 3)
    cfg["robot_target_offset_y_mm"] = round(float(new_y), 3)
    if new_z is not None:
        cfg["robot_target_offset_z_mm"] = round(float(new_z), 3)
    save_config(cfg)
    update_current_targets(float(new_x), float(new_y), old_x, old_y)
    return old_x, old_y, old_z


def check_xyz(x, y, z):
    bounds = safe_bounds()
    if not (bounds["x_min"] <= x <= bounds["x_max"]):
        raise RuntimeError(f"X={x:.2f} 超出安全範圍 {bounds['x_min']}..{bounds['x_max']}")
    if not (bounds["y_min"] <= y <= bounds["y_max"]):
        raise RuntimeError(f"Y={y:.2f} 超出安全範圍 {bounds['y_min']}..{bounds['y_max']}")
    if not (bounds["z_min"] <= z <= bounds["z_max"]):
        raise RuntimeError(f"Z={z:.2f} 超出安全範圍 {bounds['z_min']}..{bounds['z_max']}")


def lift_before_xy(robot, min_transfer_z):
    cfg = load_config()
    bounds = safe_bounds(cfg)
    lift = lift_bounds(cfg)
    pose = robot.get_pose()
    if pose is None:
        return True
    x, y, z, r = pose
    lift_z = min(max(float(z), float(min_transfer_z)), bounds["z_max"])
    if lift_z <= float(z) + 0.5:
        return True
    if not (lift["x_min"] <= float(x) <= lift["x_max"]):
        raise RuntimeError(f"目前 X={float(x):.2f} 超出抬高安全範圍 {lift['x_min']}..{lift['x_max']}")
    if not (lift["y_min"] <= float(y) <= lift["y_max"]):
        raise RuntimeError(f"目前 Y={float(y):.2f} 超出抬高安全範圍 {lift['y_min']}..{lift['y_max']}")
    if not (bounds["z_min"] <= lift_z <= bounds["z_max"]):
        raise RuntimeError(f"抬高 Z={lift_z:.2f} 超出安全範圍 {bounds['z_min']}..{bounds['z_max']}")
    print(f"換目標前先抬高：Z={float(z):.2f} -> {lift_z:.2f}")
    return robot.movl(float(x), float(y), lift_z, float(r))


def move_to_pose(robot, x, y, z):
    if hasattr(robot, "movj"):
        ok = robot.movj(float(x), float(y), float(z))
        if ok:
            return True
        print("MovJ 到高位失敗，改用 MovL 重試")
    return robot.movl(float(x), float(y), float(z))


def start(index, safe_z, speed, pre_xy_lift):
    target = selected_target(index)
    cfg = load_config()
    bounds = safe_bounds(cfg)
    x = float(target["robot_x_mm"])
    y = float(target["robot_y_mm"])
    z = float(safe_z)
    check_xyz(x, y, z)
    with MG400() as robot:
        robot.enable()
        robot.set_speed(speed)
        transfer_z = float(cfg.get("offset_calib_travel_z_mm", bounds["z_max"]))
        transfer_z = min(max(transfer_z, bounds["z_min"]), bounds["z_max"])
        if transfer_z < z + max(0.0, pre_xy_lift):
            transfer_z = min(max(z + max(0.0, pre_xy_lift), bounds["z_min"]), bounds["z_max"])
        if not lift_before_xy(robot, transfer_z):
            raise RuntimeError("移動到下一顆前抬高失敗，請檢查警報或目前姿態。")
        pose = robot.get_pose()
        travel_z = transfer_z
        if pose is not None and abs(float(pose[2]) - travel_z) > 0.5:
            print(f"先在目前 XY 回到轉移高度：Z={float(pose[2]):.2f} -> {travel_z:.2f}")
            ok = robot.movl(float(pose[0]), float(pose[1]), travel_z, float(pose[3]))
            if not ok:
                raise RuntimeError("回到轉移高度失敗，請檢查警報或目前姿態。")
        check_xyz(x, y, travel_z)
        print(f"先平移到 Q{target.get('index')} 上方高位：X={x:.2f} Y={y:.2f} Z={travel_z:.2f}")
        ok = move_to_pose(robot, x, y, travel_z)
        if ok and abs(travel_z - z) > 0.5:
            print(f"再垂直下降到校正高度：Z={z:.2f}")
            ok = robot.movl(x, y, z)
    if not ok:
        raise RuntimeError("手臂移到硬幣上方失敗，請檢查警報或選擇較中央的硬幣。")
    print(f"已移到 Q{target.get('index')} 上方：X={x:.2f} Y={y:.2f} Z={z:.2f}")


def jog(dx, dy, dz, step, speed):
    dx *= step
    dy *= step
    dz *= step
    with MG400() as robot:
        robot.enable()
        robot.set_speed(speed)
        pose = robot.get_pose()
        if pose is None:
            raise RuntimeError("讀取目前手臂座標失敗。")
        x, y, z, r = pose
        nx, ny, nz = x + dx, y + dy, z + dz
        check_xyz(nx, ny, nz)
        ok = robot.movl(nx, ny, nz, r)
    if not ok:
        raise RuntimeError("手臂微調失敗，請檢查是否接近邊界或有警報。")
    print(f"微調完成：X={nx:.2f} Y={ny:.2f} Z={nz:.2f}")


def teach(index, replace, taught_x=None, taught_y=None, taught_z=None):
    target = selected_target(index)
    cfg = load_config()
    off_x, off_y, off_z = current_offset(cfg)
    target_x = float(target["robot_x_mm"])
    target_y = float(target["robot_y_mm"])
    target_z = float(target.get("robot_z_mm", 0.0))
    raw_x = float(target.get("raw_robot_x_mm", target_x - off_x))
    raw_y = float(target.get("raw_robot_y_mm", target_y - off_y))
    raw_z = float(target.get("raw_robot_z_mm", target_z - off_z))
    if taught_x is not None and taught_y is not None and taught_z is not None:
        pose = (float(taught_x), float(taught_y), float(taught_z), 0.0)
    else:
        with MG400() as robot:
            robot.enable()
            pose = robot.get_pose()
        if pose is None:
            raise RuntimeError("讀取目前手臂座標失敗，無法教點。")
    sample = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "target_index": int(target.get("index", 0)),
        "target_label": target.get("label_name", "?"),
        "raw_robot_x_mm": round(raw_x, 3),
        "raw_robot_y_mm": round(raw_y, 3),
        "raw_robot_z_mm": round(raw_z, 3),
        "taught_robot_x_mm": round(float(pose[0]), 3),
        "taught_robot_y_mm": round(float(pose[1]), 3),
        "taught_robot_z_mm": round(float(pose[2]), 3),
        "offset_x_mm": round(float(pose[0]) - raw_x, 3),
        "offset_y_mm": round(float(pose[1]) - raw_y, 3),
        "offset_z_mm": round(float(pose[2]) - raw_z, 3),
    }
    data = {"samples": []} if replace else load_json(SAMPLES_FILE, {"samples": []})
    data.setdefault("samples", []).append(sample)
    samples = data["samples"]
    avg_x = sum(float(s["offset_x_mm"]) for s in samples) / len(samples)
    avg_y = sum(float(s["offset_y_mm"]) for s in samples) / len(samples)
    z_samples = [float(s["offset_z_mm"]) for s in samples if "offset_z_mm" in s]
    avg_z = sum(z_samples) / len(z_samples) if z_samples else 0.0
    data["average_offset_x_mm"] = round(avg_x, 3)
    data["average_offset_y_mm"] = round(avg_y, 3)
    data["average_offset_z_mm"] = round(avg_z, 3)
    data["sample_count"] = len(samples)
    write_json(SAMPLES_FILE, data)
    set_offset(avg_x, avg_y, avg_z)
    print(f"教點完成：新增 Q{target.get('index')} 樣本，目前 {len(samples)} 筆平均 X={avg_x:+.2f} Y={avg_y:+.2f} Z={avg_z:+.2f}")


def reset():
    write_json(SAMPLES_FILE, {"samples": [], "sample_count": 0, "average_offset_x_mm": 0.0, "average_offset_y_mm": 0.0, "average_offset_z_mm": 0.0})
    set_offset(0.0, 0.0, 0.0)
    print("已清空公差樣本，夾取偏移歸零。")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("start")
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--safe-z", type=float, default=DEFAULT_SAFE_Z)
    p.add_argument("--speed", type=int, default=25)
    p.add_argument("--pre-xy-lift", type=float, default=DEFAULT_PRE_XY_LIFT_MM)

    p = sub.add_parser("jog")
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dz", type=float, default=0.0)
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--speed", type=int, default=15)

    p = sub.add_parser("teach")
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--taught-x", type=float, default=None)
    p.add_argument("--taught-y", type=float, default=None)
    p.add_argument("--taught-z", type=float, default=None)

    sub.add_parser("reset")
    args = ap.parse_args()

    if args.cmd == "start":
        start(args.index, args.safe_z, max(1, min(50, int(args.speed))), max(0.0, float(args.pre_xy_lift)))
    elif args.cmd == "jog":
        jog(float(args.dx), float(args.dy), float(args.dz), float(args.step), max(1, min(30, int(args.speed))))
    elif args.cmd == "teach":
        teach(args.index, bool(args.replace), args.taught_x, args.taught_y, args.taught_z)
    elif args.cmd == "reset":
        reset()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"失敗：{exc}", file=sys.stderr)
        sys.exit(1)
