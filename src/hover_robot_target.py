# -*- coding: utf-8 -*-
"""
MG400 safe hover test.

Reads robot_targets.json and moves the robot above a selected valid coin.
Optionally performs a dry lower test. It never toggles DO/vacuum.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).parent
TARGETS_FILE = HERE / "robot_targets.json"
ACTION_STATUS_FILE = HERE / "robot_action_status.json"
CONFIG_FILE = HERE / "dual_camera_config.json"
GEMINI_LIBS = Path(r"C:\Users\user\Desktop\sam3+座標轉換與夾取")

if str(GEMINI_LIBS) not in sys.path:
    sys.path.append(str(GEMINI_LIBS))

from core.robot import MG400


SAFE_X_MIN = 100.0
SAFE_X_MAX = 390.0
AUTO_X_MIN = 120.0
AUTO_X_MAX = 380.0
SAFE_Y_MIN = -310.0
SAFE_Y_MAX = 220.0
AUTO_Y_MIN = -250.0
AUTO_Y_MAX = 190.0
SAFE_Z_MIN = 60.0
SAFE_Z_MAX = 180.0
SAFE_LOWER_Z_MIN = -162.0
SAFE_LOWER_Z_MAX = 180.0
DEFAULT_START_POSE = (30.0, 280.0, 150.0)
MOVE_SPEED_PCT = 40
LOWER_SPEED_PCT = 25
ERROR_HINTS = {
    23: {
        "meaning": "Motion interrupted / path rejected",
        "likely_cause": "Unreachable straight-line MovL path, soft-limit, collision detection, or unsafe approach path.",
        "suggested_action": "Check obstacles, raise travel Z, reduce speed, clear MG400 error, then retry from a safe pose.",
    },
    32: {
        "meaning": "Controller motion state abnormal",
        "likely_cause": "Previous motion/alarm may not be fully cleared, the current pose may be hard to plan from, or the configured camera-clear pose is not a good start pose.",
        "suggested_action": "Clear and enable the robot, move to a central safe pose, then retry.",
    },
    96: {
        "meaning": "Robot not ready / motion rejected",
        "likely_cause": "EnableRobot may not have completed successfully or the controller is not ready to accept motion commands.",
        "suggested_action": "Confirm DobotStudio shows enabled with no alarms, then retry from a safe pose.",
    },
    98: {
        "meaning": "Robot controller not ready / alarm state",
        "likely_cause": "The robot may still be disabled after Emergency Stop, or an alarm/error has not been cleared before MovL.",
        "suggested_action": "Press Clear + Enable MG400, confirm the robot is enabled in DobotStudio, then retry from the camera-clear pose.",
    },
    2: {
        "meaning": "Controller alarm / motion paused",
        "likely_cause": "The MG400 controller entered an error/pause state during motion. This often happens after path planning fails, a limit is approached, collision detection triggers, or a previous alarm was not fully cleared.",
        "suggested_action": "Stop the sequence, press Clear + Enable MG400, move to the camera-clear pose, reduce speed or skip the edge target, then retry.",
    },
    17: {
        "meaning": "MG400 rejected an edge or unreachable travel target",
        "likely_cause": "The selected coin is close to the calibrated/workspace boundary, or the robot cannot safely plan the high-Z transfer to that XY.",
        "suggested_action": "Clear the alarm, skip this edge target, choose a central coin, and keep targets inside the conservative workspace.",
    },
    18: {
        "meaning": "MG400 rejected the automatic path or low-Z motion",
        "likely_cause": "The high-Z transfer pose may be hard to plan, the dry-lower target may be too low, XY may still have local calibration error, collision detection triggered, or the point is near a workspace boundary.",
        "suggested_action": "Clear the alarm, verify this coin with hover only, then retry. If manual jogging reaches the point, the program will try MovL after MovJ fails.",
    },
    66: {
        "meaning": "MG400 rejected the camera-clear/start pose",
        "likely_cause": "The configured return/start pose is unreachable or too low for a safe transfer, or the controller still has an uncleared alarm.",
        "suggested_action": "Clear and enable the robot, use a higher camera-clear Z, then retry from a safe pose.",
    },
}


def _has_robot_xyz(target):
    try:
        float(target["robot_x_mm"])
        float(target["robot_y_mm"])
        float(target.get("robot_z_mm", target.get("raw_robot_z_mm")))
    except Exception:
        return False
    return True


def load_targets(include_non_top_pickable=False):
    if not TARGETS_FILE.exists():
        raise FileNotFoundError(f"找不到 {TARGETS_FILE}")
    data = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    targets = []
    for t in data.get("targets", []):
        if t.get("valid_for_pick"):
            targets.append(t)
            continue
        if include_non_top_pickable and t.get("pick_check_reason") in ("not_top_coin", "unknown_class") and _has_robot_xyz(t):
            targets.append(t)
    if not targets:
        if include_non_top_pickable:
            raise RuntimeError("robot_targets.json 裡沒有可逐顆下降的硬幣")
        raise RuntimeError("robot_targets.json 裡沒有 valid_for_pick=true 的硬幣")
    return data, targets


def load_config():
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def default_start_pose(cfg=None):
    cfg = cfg or load_config()
    return (
        float(cfg.get("robot_return_x_mm", DEFAULT_START_POSE[0])),
        float(cfg.get("robot_return_y_mm", DEFAULT_START_POSE[1])),
        float(cfg.get("robot_return_z_mm", DEFAULT_START_POSE[2])),
    )


def targets_are_fresh(max_age_sec):
    if max_age_sec <= 0 or not TARGETS_FILE.exists():
        return False
    age = time.time() - TARGETS_FILE.stat().st_mtime
    if age > max_age_sec:
        return False
    try:
        _data, targets = load_targets()
    except Exception:
        return False
    print(f"[hover] 沿用 {age:.1f} 秒前已鎖定的辨識座標，跳過重新辨識 ({len(targets)} targets)")
    return True


def write_action_status(state, target=None, message="", error_code=None, robot_xyz=None, controller_response=None):
    payload = {"state": state, "message": message}
    if controller_response:
        payload["controller_response"] = str(controller_response)
    if error_code is not None:
        payload["error_code"] = error_code
        payload["error_hint"] = ERROR_HINTS.get(error_code, {
            "meaning": "Unknown MG400 error",
            "likely_cause": "Controller returned an unmapped error code.",
            "suggested_action": "Check DobotStudio/MG400 alarm details and clear errors before retrying.",
        })
        payload["requires_human_intervention"] = True
    if robot_xyz is not None:
        payload["attempted_robot_xyz_mm"] = [float(v) for v in robot_xyz]
    if target is not None:
        payload["target"] = {
            "index": target.get("index"),
            "label_name": target.get("label_name"),
            "robot_x_mm": target.get("robot_x_mm"),
            "robot_y_mm": target.get("robot_y_mm"),
            "robot_z_mm": target.get("robot_z_mm"),
        }
    try:
        ACTION_STATUS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def refresh_targets_after_start():
    print("[hover] 手臂已離開相機視野，開始重新影像辨識...")
    release_camera_processes()
    cmd = [sys.executable, str(HERE / "dual_camera_live.py"), "--save-once", "--fast", "--quality-only"]
    result = None
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    for attempt in range(1, 4):
        time.sleep(0.8)
        result = subprocess.run(cmd, cwd=str(HERE), text=True, capture_output=True, encoding="utf-8", errors="replace", env=env)
        output = (result.stdout or "") + (result.stderr or "")
        if output.strip():
            print(output[-3000:])
        if result.returncode == 0:
            break
        print(f"[hover] 影像辨識第 {attempt} 次失敗，釋放相機後重試")
        release_camera_processes()
    if result is None or result.returncode != 0:
        output = "" if result is None else ((result.stdout or "") + (result.stderr or ""))
        detail = output[-1200:].strip()
        raise RuntimeError(
            f"影像辨識失敗 returncode={None if result is None else result.returncode}"
            + (f"\n{detail}" if detail else "")
        )
    print("[hover] 影像辨識完成，讀取最新 robot_targets.json")


def release_camera_processes():
    ps = (
        "$self=$PID; "
        "$pattern='dual_camera_live|camera_preview_once|select_quality_roi|calibrate_robot_tabletop_homography|"
        "tune_gemini_display_roi|tune_gemini_exposure|capture_stereo|capture_one_stereo|"
        "capture_quality_calib|capture_tabletop|capture'; "
        "$procs=Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -match 'coin_classifier' -and $_.CommandLine -match $pattern "
        "}; "
        "foreach ($p in $procs) { if ($p.ProcessId -ne $self) { "
        "try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } catch {} "
        "} }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], cwd=str(HERE), text=True, capture_output=True, encoding="utf-8", errors="replace")


def check_bounds(x, y, z):
    bounds = safe_bounds()
    if not (bounds["x_min"] <= x <= bounds["x_max"]):
        raise RuntimeError(f"X={x:.2f} 超出安全範圍 {bounds['x_min']}..{bounds['x_max']}")
    if not (bounds["y_min"] <= y <= bounds["y_max"]):
        raise RuntimeError(f"Y={y:.2f} 超出安全範圍 {bounds['y_min']}..{bounds['y_max']}")
    if not (bounds["z_min"] <= z <= bounds["z_max"]):
        raise RuntimeError(f"Z={z:.2f} 超出安全範圍 {bounds['z_min']}..{bounds['z_max']}")


def is_auto_safe_target(target, cfg=None):
    cfg = cfg or {}
    x = float(target["robot_x_mm"])
    y = float(target["robot_y_mm"])
    x_min = float(cfg.get("robot_auto_x_min_mm", cfg.get("robot_pick_x_min_mm", AUTO_X_MIN)))
    x_max = float(cfg.get("robot_auto_x_max_mm", min(float(cfg.get("robot_pick_x_max_mm", AUTO_X_MAX)), AUTO_X_MAX)))
    y_min = float(cfg.get("robot_auto_y_min_mm", cfg.get("robot_pick_y_min_mm", AUTO_Y_MIN)))
    y_max = float(cfg.get("robot_auto_y_max_mm", cfg.get("robot_pick_y_max_mm", AUTO_Y_MAX)))
    return x_min <= x <= x_max and y_min <= y <= y_max


def is_within_motion_bounds(target, z, cfg=None):
    bounds = safe_bounds(cfg)
    x = float(target["robot_x_mm"])
    y = float(target["robot_y_mm"])
    return (
        bounds["x_min"] <= x <= bounds["x_max"]
        and bounds["y_min"] <= y <= bounds["y_max"]
        and bounds["z_min"] <= float(z) <= bounds["z_max"]
    )


def check_lower_bounds(z):
    if not (SAFE_LOWER_Z_MIN <= z <= SAFE_LOWER_Z_MAX):
        raise RuntimeError(f"lower Z={z:.2f} 超出保守安全範圍 {SAFE_LOWER_Z_MIN}..{SAFE_LOWER_Z_MAX}")


def parse_xyz(text):
    parts = str(text).replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("格式需為 X,Y,Z，例如 30,280,150")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("X/Y/Z 必須是數字") from e


def move_to_pose(robot, x, y, z):
    if hasattr(robot, "movj"):
        ok = robot.movj(x, y, z)
        if ok:
            return True
        print("[hover] MovJ failed, retry with MovL")
    return robot.movl(x, y, z)


def lift_before_xy(robot, lift_mm, max_z):
    if lift_mm <= 0:
        return True, None
    pose = robot.get_pose()
    if pose is None:
        print("[hover] 無法讀取目前 Z，略過下一顆前抬高")
        return True, None
    x, y, z, r = pose
    lift_z = min(float(z) + float(lift_mm), float(max_z))
    if lift_z <= float(z) + 0.5:
        return True, (x, y, z)
    print(f"[hover] 下一顆前先原地抬高 Z {z:.2f} -> {lift_z:.2f}")
    ok = robot.movl(float(x), float(y), lift_z, float(r))
    return ok, (float(x), float(y), lift_z)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None, help="robot_targets.json 裡的硬幣 index；不填則取第一顆 valid")
    ap.add_argument("--all", action="store_true", help="逐顆走過所有 valid_for_pick=true 的硬幣")
    ap.add_argument("--include-non-top-pickable", action="store_true", help="搭配 --all：dry-run 逐顆下降時包含非最上層但座標有效的硬幣")
    ap.add_argument("--fallback-first-valid", action="store_true", help="指定 index 重新辨識後不可取時，改用最新第一顆可取目標")
    ap.add_argument("--safe-z", type=float, default=100.0, help="只移到硬幣上方的安全 Z，高度 mm")
    ap.add_argument("--travel-z", type=float, default=150.0, help="XY 轉移時使用的高空 Z")
    ap.add_argument("--between-target-lift", type=float, default=20.0, help="逐顆移到下一顆前，先在目前 XY 原地抬高的 Z 距離 mm")
    ap.add_argument("--lower-z", type=float, default=None, help="可選：乾跑下降到此 Z；不開真空、不切 DO")
    ap.add_argument("--use-target-lower-z", action="store_true", help="乾跑下降使用每顆 target 的 robot_z_mm；若缺少則退回 --lower-z")
    ap.add_argument("--move-speed", type=int, default=MOVE_SPEED_PCT, help="一般移動速度百分比")
    ap.add_argument("--lower-speed", type=int, default=LOWER_SPEED_PCT, help="下降速度百分比")
    ap.add_argument("--move-speed-j", type=int, default=int(cfg.get("robot_move_speed_j_pct", 80)), help="MovJ 速度百分比")
    ap.add_argument("--move-acc-j", type=int, default=int(cfg.get("robot_move_acc_j_pct", 70)), help="MovJ 加速度百分比")
    ap.add_argument("--move-speed-l", type=int, default=int(cfg.get("robot_move_speed_l_pct", 70)), help="MovL 移動速度百分比")
    ap.add_argument("--move-acc-l", type=int, default=int(cfg.get("robot_move_acc_l_pct", 60)), help="MovL 移動加速度百分比")
    ap.add_argument("--lower-speed-l", type=int, default=int(cfg.get("robot_lower_speed_l_pct", 35)), help="下降 MovL 速度百分比")
    ap.add_argument("--lower-acc-l", type=int, default=int(cfg.get("robot_lower_acc_l_pct", 25)), help="下降 MovL 加速度百分比")
    ap.add_argument("--start-pose", type=parse_xyz, default=default_start_pose(cfg), help="開始前先移到此 X,Y,Z 避開相機")
    ap.add_argument("--no-start-pose", action="store_true", help="不要先移到開始位置")
    ap.add_argument("--start-only", action="store_true", help="只移到開始位置，不讀硬幣目標")
    ap.add_argument("--refresh-after-start", action="store_true", help="先移到開始位置，再重新跑一次影像辨識，最後才去硬幣")
    ap.add_argument("--refresh-max-age-sec", type=float, default=0.0, help="若 robot_targets.json 比此秒數更新，沿用已鎖定座標以加快動作")
    ap.add_argument("--skip-start-if-close", action="store_true", help="若目前已接近開始位置，就不再先移動一次")
    ap.add_argument("--no-return-start", action="store_true", help="動作結束後不要回開始位置")
    ap.add_argument("--yes", action="store_true", help="不等待確認，直接移動")
    args = ap.parse_args()

    start_pose = None if args.no_start_pose else tuple(float(v) for v in args.start_pose)
    if start_pose is not None:
        sx, sy, sz = start_pose
        if not (-200.0 <= sx <= 500.0 and -400.0 <= sy <= 400.0 and -250.0 <= sz <= 250.0):
            raise RuntimeError(f"start pose X={sx:.2f} Y={sy:.2f} Z={sz:.2f} 超出安全範圍")
    if args.lower_z is not None:
        check_lower_bounds(float(args.lower_z))
    travel_z = float(args.travel_z)
    if not (SAFE_Z_MIN <= travel_z <= SAFE_Z_MAX):
        raise RuntimeError(f"travel Z={travel_z:.2f} 超出安全範圍 {SAFE_Z_MIN}..{SAFE_Z_MAX}")

    print("[hover] 安全定位測試，不開真空、不切 DO")
    if start_pose is not None:
        print(f"[hover] first move to start pose X={start_pose[0]:.2f} Y={start_pose[1]:.2f} Z={start_pose[2]:.2f}")
    if args.start_only:
        print("[hover] start-only mode：只回避開相機位置")
    if args.use_target_lower_z:
        print("[hover] dry-lower target Z=each target robot_z_mm")
    elif args.lower_z is not None:
        print(f"[hover] dry-lower target Z={float(args.lower_z):.2f}")

    if not args.yes:
        ans = input("確認開始？會先回避開相機位置，必要時重新辨識，再去硬幣。輸入 y 後 Enter: ").strip().lower()
        if ans not in ("y", "yes"):
            print("[hover] 已取消")
            return

    last_target = None
    last_attempt = None
    robot = MG400()
    try:
        robot.connect()
        robot.enable()
        move_speed = max(1, min(100, int(args.move_speed)))
        lower_speed = max(1, min(100, int(args.lower_speed)))
        robot.set_speed(move_speed)
        move_speed_j = max(1, min(100, int(args.move_speed_j)))
        move_acc_j = max(1, min(100, int(args.move_acc_j)))
        move_speed_l = max(1, min(100, int(args.move_speed_l)))
        move_acc_l = max(1, min(100, int(args.move_acc_l)))
        lower_speed_l = max(1, min(100, int(args.lower_speed_l)))
        lower_acc_l = max(1, min(100, int(args.lower_acc_l)))
        robot.set_motion_profile(speed_j=move_speed_j, acc_j=move_acc_j, speed_l=move_speed_l, acc_l=move_acc_l)
        print(
            "[robot-speed] "
            f"MovJ SpeedJ={move_speed_j} AccJ={move_acc_j} "
            f"MovL SpeedL={move_speed_l} AccL={move_acc_l} "
            f"Lower SpeedL={lower_speed_l} AccL={lower_acc_l}"
        )
        if start_pose is not None:
            need_start_move = True
            if args.skip_start_if_close:
                pose = robot.get_pose()
                if pose is not None:
                    dist = ((pose[0] - start_pose[0]) ** 2 + (pose[1] - start_pose[1]) ** 2 + (pose[2] - start_pose[2]) ** 2) ** 0.5
                    need_start_move = dist > 3.0
                    if not need_start_move:
                        print("[hover] 已在開始位置附近，略過回避移動")
            if need_start_move:
                if not move_to_pose(robot, start_pose[0], start_pose[1], start_pose[2]):
                    print("[hover] 移動到開始位置失敗")
                    write_action_status("failed", None, "move to start pose failed")
                    sys.exit(2)
        if args.start_only:
            print("[hover] 已到開始位置")
            return
        if args.refresh_after_start and not targets_are_fresh(float(args.refresh_max_age_sec)):
            refresh_targets_after_start()

        data, targets = load_targets(include_non_top_pickable=args.all and args.include_non_top_pickable)
        if args.all:
            target = targets[0]
        elif args.index is None:
            target = targets[0]
        else:
            matched = [t for t in targets if int(t.get("index", -1)) == args.index]
            if not matched:
                if args.fallback_first_valid:
                    target = targets[0]
                    print(
                        f"[hover] Q{args.index} 重新辨識後已不是可取目標，"
                        f"改用最新第一顆可取目標 Q{target.get('index')}"
                    )
                else:
                    raise RuntimeError(
                        f"Q{args.index} 重新辨識後不是可取目標。"
                        "可能原因：硬幣太靠近工作邊界、深度/座標無效，或重新辨識後排序改變。"
                    )
            else:
                target = matched[0]

        z = float(args.safe_z)
        if args.all:
            skipped = [t for t in targets if not is_auto_safe_target(t, cfg) or not is_within_motion_bounds(t, z, cfg)]
            if skipped:
                print("[hover] 以下目標靠近工作邊界，略過自動 ALL，請人工確認後單顆測試：")
                print(
                    "[hover] 自動範圍 "
                    f"X={float(cfg.get('robot_auto_x_min_mm', cfg.get('robot_pick_x_min_mm', AUTO_X_MIN))):.1f}"
                    f"..{float(cfg.get('robot_auto_x_max_mm', min(float(cfg.get('robot_pick_x_max_mm', AUTO_X_MAX)), AUTO_X_MAX))):.1f} "
                    f"Y={float(cfg.get('robot_auto_y_min_mm', cfg.get('robot_pick_y_min_mm', AUTO_Y_MIN))):.1f}"
                    f"..{float(cfg.get('robot_auto_y_max_mm', cfg.get('robot_pick_y_max_mm', AUTO_Y_MAX))):.1f}"
                )
                for t in skipped:
                    print(f"  Q{t.get('index')} {t.get('label_name')} X={float(t['robot_x_mm']):.2f} Y={float(t['robot_y_mm']):.2f}")
            run_targets = [t for t in targets if is_auto_safe_target(t, cfg) and is_within_motion_bounds(t, z, cfg)]
            if not run_targets:
                raise RuntimeError("所有 valid 目標都靠近邊界，已停止自動 ALL")
        else:
            run_targets = [target]
        ok = True
        failed_targets = []
        for i, target in enumerate(run_targets, 1):
            last_target = target
            x = float(target["robot_x_mm"])
            y = float(target["robot_y_mm"])
            check_bounds(x, y, z)
            print(f"[hover] target {i}/{len(run_targets)} Q{target['index']} {target.get('label_name', '?')}  d={target.get('diameter_mm')}mm")
            if i > 1:
                write_action_status("lift", target, f"{i}/{len(run_targets)} lift before next target")
                ok, lifted_attempt = lift_before_xy(robot, float(args.between_target_lift), travel_z)
                if lifted_attempt is not None:
                    last_attempt = lifted_attempt
                if not ok:
                    break
            print(f"[hover] move MG400 to X={x:.2f}  Y={y:.2f}  travel Z={travel_z:.2f}")
            last_attempt = (x, y, travel_z)
            write_action_status("travel", target, f"{i}/{len(run_targets)} high travel above target")
            ok = move_to_pose(robot, x, y, travel_z)
            if ok and abs(travel_z - z) > 0.5:
                print(f"[hover] lower to safe Z={z:.2f}")
                last_attempt = (x, y, z)
                write_action_status("hover", target, f"{i}/{len(run_targets)} lower to safe hover")
                ok = robot.movl(x, y, z)
            target_lower_z = None
            if args.use_target_lower_z and target.get("z_offset_ready") and target.get("robot_z_mm") is not None:
                target_lower_z = float(target["robot_z_mm"])
                check_lower_bounds(target_lower_z)
            elif args.lower_z is not None:
                target_lower_z = float(args.lower_z)
            if ok and target_lower_z is not None:
                print(f"[hover] dry-lower to Z={target_lower_z:.2f}, then return to Z={z:.2f}")
                write_action_status("lower", target, f"{i}/{len(run_targets)} dry lower")
                robot.set_speed(lower_speed)
                last_attempt = (x, y, target_lower_z)
                ok = robot.movl(x, y, target_lower_z, speed_l=lower_speed_l, acc_l=lower_acc_l)
                if ok:
                    write_action_status("return", target, f"{i}/{len(run_targets)} return safe height")
                    last_attempt = (x, y, z)
                    ok = robot.movl(x, y, z)
                    robot.set_speed(move_speed)
            if not ok:
                if args.all:
                    try:
                        errs = getattr(robot, "last_errors", None) or robot.get_errors()
                        error_code = errs[0] if errs else None
                    except Exception:
                        error_code = None
                    controller_response = getattr(robot, "last_response", None)
                    failed_targets.append((target, error_code, controller_response, last_attempt))
                    print(
                        f"[hover] Q{target.get('index')} 移動失敗，略過此目標並繼續下一顆 "
                        f"(error={error_code})"
                    )
                    write_action_status(
                        "skipped",
                        target,
                        f"{i}/{len(run_targets)} skipped after robot move failed",
                        error_code=error_code,
                        robot_xyz=last_attempt,
                        controller_response=controller_response,
                    )
                    try:
                        robot.clear_error()
                        robot.enable()
                        robot.set_speed(move_speed)
                    except Exception as exc:
                        print(f"[hover] 清除錯誤後無法繼續：{exc}")
                        ok = False
                        break
                    ok = True
                    continue
                break
        error_code = None
        controller_response = None
        if not ok:
            try:
                errs = getattr(robot, "last_errors", None) or robot.get_errors()
                error_code = errs[0] if errs else None
            except Exception:
                error_code = None
            controller_response = getattr(robot, "last_response", None)
        write_action_status(
            "done" if ok else "failed",
            None if ok else last_target,
            (
                f"all targets completed, skipped={len(failed_targets)}"
                if ok and failed_targets
                else "all targets completed"
                if ok
                else "robot move failed - human intervention required"
            ),
            error_code=error_code,
            robot_xyz=last_attempt if not ok else None,
            controller_response=controller_response,
        )
        if ok and start_pose is not None and not args.no_return_start:
            write_action_status("return_start", None, "return to camera-clear pose")
            robot.set_speed(move_speed)
            ok = move_to_pose(robot, start_pose[0], start_pose[1], start_pose[2])
            write_action_status("done" if ok else "failed", None, "returned to camera-clear pose" if ok else "return to start failed")
        if failed_targets:
            print("[hover] 以下目標移動失敗已跳過：" + ", ".join(f"Q{t.get('index')}" for t, _e, _r, _a in failed_targets))
        print("[hover] 移動完成" if ok else "[hover] 移動失敗")
        if not ok:
            sys.exit(2)
    except (ConnectionAbortedError, ConnectionResetError, OSError) as exc:
        print(f"[hover] 動作中止或連線中斷：{exc}")
        write_action_status(
            "failed",
            last_target,
            "robot connection interrupted or emergency stop pressed",
            robot_xyz=last_attempt,
            controller_response=str(exc),
        )
        sys.exit(2)
    except RuntimeError as exc:
        print(f"[hover] 動作失敗：{exc}")
        write_action_status(
            "failed",
            last_target,
            str(exc),
            robot_xyz=last_attempt,
            controller_response=str(exc),
        )
        sys.exit(2)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
