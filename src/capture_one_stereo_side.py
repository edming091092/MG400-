# capture_one_stereo_side.py
# -*- coding: utf-8 -*-

"""分開擷取 Gemini 或側相機的雙相機標定照片。"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

from calibrate_camera import find_chessboard_corners, resize_for_show
from calibration_session import archive_dir, choose_session_mode
from capture_stereo_calib_pairs import blur_score, draw_status
from dual_camera_live import load_config, open_quality_camera
from gemini_controls import apply_color_controls, set_gemini_stream_env


HERE = Path(__file__).parent
GEMINI_LIBS = Path(r"C:\Users\user\Desktop\sam3+座標轉換與夾取")
OUT_DIR = HERE / "stereo_calib_pairs_7x9"

if str(GEMINI_LIBS) not in sys.path:
    sys.path.append(str(GEMINI_LIBS))

from core.camera import Gemini2Camera


def detect_any(gray, board_w, board_h):
    for pattern in ((board_w, board_h), (board_h, board_w)):
        found, corners = find_chessboard_corners(gray, pattern)
        if found and corners is not None:
            return True, pattern, corners
    return False, (board_w, board_h), None


def next_index(side_dir, side):
    existing = sorted(side_dir.glob(f"pair_*_{side}.jpg"))
    if not existing:
        return 1
    indexes = []
    for path in existing:
        parts = path.name.split("_")
        if len(parts) >= 2 and parts[0] == "pair":
            try:
                indexes.append(int(parts[1]))
            except ValueError:
                pass
    return (max(indexes) + 1) if indexes else (len(existing) + 1)


def save_metadata(out_dir, board_w, board_h):
    metadata_path = out_dir / "metadata_separate.json"
    data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "separate_camera_capture",
        "board": {
            "inner_corners_w": board_w,
            "inner_corners_h": board_h,
        },
        "note": "Gemini and Quality images are paired by pair index after both sides are captured.",
    }
    metadata_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="分開擷取單一相機的雙相機標定照片")
    parser.add_argument("--side", choices=("gemini", "quality"), required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--board-w", type=int, default=9)
    parser.add_argument("--board-h", type=int, default=6)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--reset", action="store_true", help="備份這一側舊照片並重新開始")
    mode_group.add_argument("--append", action="store_true", help="保留這一側舊照片並繼續新增")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else HERE / args.out_dir
    side_dir = out_dir / args.side
    preview_dir = out_dir / "single_preview"
    existing_count = len(list(side_dir.glob(f"pair_*_{args.side}.jpg"))) if side_dir.exists() else 0
    if args.reset:
        session_mode = "reset"
    elif args.append:
        session_mode = "append"
    else:
        session_mode = choose_session_mode(f"單相機標定照片擷取：{args.side}", existing_count, default="reset")
    if session_mode == "reset":
        backup = archive_dir(side_dir)
        if backup is not None:
            print(f"[資料] 舊資料已備份：{backup}")
    side_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(out_dir, args.board_w, args.board_h)

    pattern_size = (args.board_w, args.board_h)
    camera = None
    cap = None

    try:
        if args.side == "gemini":
            cfg = load_config()
            set_gemini_stream_env(cfg)
            camera = Gemini2Camera(align_depth_to_color=True)
            camera.open()
            apply_color_controls(camera, cfg)
            title = "Gemini color"
        else:
            cfg = load_config()
            cap, status = open_quality_camera(cfg)
            if cap is None:
                raise RuntimeError(f"側相機開啟失敗：{status}")
            title = "Quality side"

        print("=" * 60)
        print(f"單相機標定照片擷取：{title}")
        print("=" * 60)
        print(f"輸出資料夾：{side_dir}")
        print(f"資料模式：{'保留舊資料繼續新增' if session_mode == 'append' else '清空舊資料重新開始'}")
        print(f"棋盤格內角點：{args.board_w} x {args.board_h}")
        print("SPACE 拍照，C 檢查棋盤偵測，Q/ESC 離開")
        print("請用同一個 pair 編號對應另一台相機的照片。")

        win = f"Single Capture - {title}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        found = False
        corners = None
        detected_pattern = pattern_size
        blur = None

        while True:
            if args.side == "gemini":
                frame, _ = camera.get_frames(timeout_ms=1000)
                ok = frame is not None
            else:
                ok, frame = cap.read()

            if not ok or frame is None:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            blur = blur_score(frame)
            vis = frame.copy()
            if found and corners is not None:
                cv2.drawChessboardCorners(vis, detected_pattern, corners, found)
            vis = draw_status(
                vis,
                title,
                found,
                next_index(side_dir, args.side) - 1,
                blur,
                "single camera mode",
                photo_mode=False,
                session_mode=session_mode,
            )
            cv2.imshow(win, resize_for_show(vis, 1280, 720))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print("[檢查] 正在偵測棋盤...")
                found, detected_pattern, corners = detect_any(gray, args.board_w, args.board_h)
                print(f"[檢查] {title}={found} pattern={detected_pattern} blur={blur:.1f}")
                continue
            if key == 32:
                pair_idx = next_index(side_dir, args.side)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = f"pair_{pair_idx:03d}_{ts}_separate"
                img_path = side_dir / f"{stem}_{args.side}.jpg"
                preview_path = preview_dir / f"{stem}_{args.side}_preview.jpg"
                cv2.imwrite(str(img_path), frame)
                cv2.imwrite(str(preview_path), vis)
                print(f"[拍照] {title} pair {pair_idx:03d} 已儲存：{img_path.name}")

    finally:
        if cap is not None:
            cap.release()
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
