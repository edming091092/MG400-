# -*- coding: utf-8 -*-
"""
MG400 safe hover test.

Reads robot_targets.json and moves the robot above a selected valid coin.
Optionally performs a dry lower test. It never toggles DO/vacuum.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).parent
TARGETS_FILE = HERE / "robot_targets.json"
ACTION_STATUS_FILE = HERE / "robot_action_status.json"
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
SAFE_LOWER_Z_MIN = -158.0
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
        "meaning": "MG400 alarm during low-Z motion",
        "likely_cause": "The dry-lower target is too low, the XY target is offset, collision detection triggered, or the selected point is outside the reliably calibrated area.",
        "suggested_action": "Stop low-Z tests, clear the alarm in DobotStudio/UI, verify XY with hover only, then retry with a higher lower-Z such as -145 before returning to -158.",
    },
}


def load_targets():
    if not TARGETS_FILE.exists():
        raise FileNotFoundError(f"找不到 {TARGETS_FILE}")
    data = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    targets = [t for t in data.get("targets", []) if t.get("valid_for_pick")]
    if not targets:
        raise RuntimeError("robot_targets.json 裡沒有 valid_for_pick=true 的硬幣")
    return data, targets


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
    for attempt in range(1, 4):
        time.sleep(0.35)
        result = subprocess.run(cmd, cwd=str(HERE), text=True, encoding="utf-8", errors="replace")
        if result.returncode == 0:
            break
        print(f"[hover] 影像辨識第 {attempt} 次失敗，釋放相機後重試")
        release_camera_processes()
    if result is None or result.returncode != 0:
        raise RuntimeError(f"影像辨識失敗 returncode={None if result is None else result.returncode}")
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
    if not (SAFE_X_MIN <= x <= SAFE_X_MAX):
        raise RuntimeError(f"X={x:.2f} 超出安全範圍 {SAFE_X_MIN}..{SAFE_X_MAX}")
    if not (SAFE_Y_MIN <= y <= SAFE_Y_MAX):
        raise RuntimeError(f"Y={y:.2f} 超出安全範圍 {SAFE_Y_MIN}..{SAFE_Y_MAX}")
    if not (SAFE_Z_MIN <= z <= SAFE_Z_MAX):
        raise RuntimeError(f"Z={z:.2f} 超出安全範圍 {SAFE_Z_MIN}..{SAFE_Z_MAX}")


def is_auto_safe_target(target):
    x = float(target["robot_x_mm"])
    y = float(target["robot_y_mm"])
    return AUTO_X_MIN <= x <= AUTO_X_MAX and AUTO_Y_MIN <= y <= AUTO_Y_MAX


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None, help="robot_targets.json 裡的硬幣 index；不填則取第一顆 valid")
    ap.add_argument("--all", action="store_true", help="逐顆走過所有 valid_for_pick=true 的硬幣")
    ap.add_argument("--fallback-first-valid", action="store_true", help="指定 index 重新辨識後不可取時，改用最新第一顆可取目標")
    ap.add_argument("--safe-z", type=float, default=100.0, help="只移到硬幣上方的安全 Z，高度 mm")
    ap.add_argument("--travel-z", type=float, default=150.0, help="XY 轉移時使用的高空 Z")
    ap.add_argument("--lower-z", type=float, default=None, help="可選：乾跑下降到此 Z；不開真空、不切 DO")
    ap.add_argument("--move-speed", type=int, default=MOVE_SPEED_PCT, help="一般移動速度百分比")
    ap.add_argument("--lower-speed", type=int, default=LOWER_SPEED_PCT, help="下降速度百分比")
    ap.add_argument("--start-pose", type=parse_xyz, default=DEFAULT_START_POSE, help="開始前先移到此 X,Y,Z 避開相機")
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
        if not (-50.0 <= sx <= 450.0 and -330.0 <= sy <= 330.0 and 60.0 <= sz <= 220.0):
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
    if args.lower_z is not None:
        print(f"[hover] dry-lower target Z={float(args.lower_z):.2f}")

    if not args.yes:
        ans = input("確認開始？會先回避開相機位置，必要時重新辨識，再去硬幣。輸入 y 後 Enter: ").strip().lower()
        if ans not in ("y", "yes"):
            print("[hover] 已取消")
            return

    robot = MG400()
    try:
        robot.connect()
        robot.enable()
        move_speed = max(1, min(100, int(args.move_speed)))
        lower_speed = max(1, min(100, int(args.lower_speed)))
        robot.set_speed(move_speed)
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

        data, targets = load_targets()
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
            skipped = [t for t in targets if not is_auto_safe_target(t)]
            if skipped:
                print("[hover] 以下目標靠近工作邊界，略過自動 ALL，請人工確認後單顆測試：")
                for t in skipped:
                    print(f"  Q{t.get('index')} {t.get('label_name')} X={float(t['robot_x_mm']):.2f} Y={float(t['robot_y_mm']):.2f}")
            run_targets = [t for t in targets if is_auto_safe_target(t)]
            if not run_targets:
                raise RuntimeError("所有 valid 目標都靠近邊界，已停止自動 ALL")
        else:
            run_targets = [target]
        ok = True
        last_target = None
        last_attempt = None
        for i, target in enumerate(run_targets, 1):
            last_target = target
            x = float(target["robot_x_mm"])
            y = float(target["robot_y_mm"])
            check_bounds(x, y, z)
            print(f"[hover] target {i}/{len(run_targets)} Q{target['index']} {target.get('label_name', '?')}  d={target.get('diameter_mm')}mm")
            print(f"[hover] move MG400 to X={x:.2f}  Y={y:.2f}  travel Z={travel_z:.2f}")
            last_attempt = (x, y, travel_z)
            write_action_status("travel", target, f"{i}/{len(run_targets)} high travel above target")
            ok = robot.movj(x, y, travel_z) if hasattr(robot, "movj") else robot.movl(x, y, travel_z)
            if ok and abs(travel_z - z) > 0.5:
                print(f"[hover] lower to safe Z={z:.2f}")
                last_attempt = (x, y, z)
                write_action_status("hover", target, f"{i}/{len(run_targets)} lower to safe hover")
                ok = robot.movl(x, y, z)
            if ok and args.lower_z is not None:
                print(f"[hover] dry-lower to Z={float(args.lower_z):.2f}, then return to Z={z:.2f}")
                write_action_status("lower", target, f"{i}/{len(run_targets)} dry lower")
                robot.set_speed(lower_speed)
                last_attempt = (x, y, float(args.lower_z))
                ok = robot.movl(x, y, float(args.lower_z))
                if ok:
                    write_action_status("return", target, f"{i}/{len(run_targets)} return safe height")
                    last_attempt = (x, y, z)
                    ok = robot.movl(x, y, z)
                    robot.set_speed(move_speed)
            if not ok:
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
            "all targets completed" if ok else "robot move failed - human intervention required",
            error_code=error_code,
            robot_xyz=last_attempt if not ok else None,
            controller_response=controller_response,
        )
        if ok and start_pose is not None and not args.no_return_start:
            write_action_status("return_start", None, "return to camera-clear pose")
            robot.set_speed(move_speed)
            ok = move_to_pose(robot, start_pose[0], start_pose[1], start_pose[2])
            write_action_status("done" if ok else "failed", None, "returned to camera-clear pose" if ok else "return to start failed")
        print("[hover] 移動完成" if ok else "[hover] 移動失敗")
        if not ok:
            sys.exit(2)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
