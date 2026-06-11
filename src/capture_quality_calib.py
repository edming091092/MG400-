# capture_quality_calib.py
# -*- coding: utf-8 -*-

"""
側相機棋盤格標定照片擷取工具

操作：
- SPACE：儲存目前畫面
- Q / ESC：離開

建議拍 15~30 張，讓棋盤格出現在畫面中央、四角、邊緣，並有不同傾角。
拍完後執行 calibrate_camera.py，指定本工具的輸出資料夾即可產生側相機內參。
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2

from calibrate_camera import find_chessboard_corners
from calibration_session import archive_dir, choose_session_mode


HERE = Path(__file__).parent
CONFIG_FILE = HERE / "dual_camera_config.json"
OUT_DIR = HERE / "quality_calib_images"
PREVIEW_DIR = HERE / "quality_calib_preview"


def load_dual_camera_config():
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def _camera_backend(name):
    return {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }.get(str(name).lower(), cv2.CAP_DSHOW)


def apply_camera_props(cap, cfg, width, height, fps):
    fourcc = str(cfg.get("quality_fourcc", "MJPG")).strip().upper()
    if fourcc and fourcc not in ("NONE", "DEFAULT", "0"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4].ljust(4)))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(cfg.get("quality_buffer_size", 1)))
    except Exception:
        pass


def open_camera(index, width, height, fps, cfg):
    backend_name = str(cfg.get("quality_camera_backend", "dshow")).lower()
    backend_names = [backend_name, "dshow"] if backend_name != "msmf" else ["msmf", "dshow"]
    backend_names = list(dict.fromkeys(backend_names))
    cap = None
    for name in backend_names:
        candidate = cv2.VideoCapture(index, _camera_backend(name))
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()
    if cap is None:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    apply_camera_props(cap, cfg, width, height, fps)
    return cap


def draw_chessboard_visible(frame, pattern_size, corners, scale):
    if corners is None:
        return frame
    pts = corners.reshape(-1, 2) * float(scale)
    cols, rows = pattern_size
    radius = max(4, int(round(5 / max(scale, 0.2))))
    thickness = max(2, int(round(3 / max(scale, 0.2))))
    for r in range(rows):
        row = pts[r * cols:(r + 1) * cols].astype(int)
        color = (0, 220, 255) if r % 2 == 0 else (255, 120, 0)
        cv2.polylines(frame, [row.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    for p in pts.astype(int):
        cv2.circle(frame, tuple(p), radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(p), radius, (0, 255, 120), max(1, thickness - 1), cv2.LINE_AA)
    return frame


def resize_for_capture_show(frame, max_w=1600, max_h=900):
    h, w = frame.shape[:2]
    scale = min(float(max_w) / max(float(w), 1.0), float(max_h) / max(float(h), 1.0), 1.0)
    if scale >= 0.999:
        return frame.copy(), 1.0
    out = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def draw_status(frame, found, saved_count, camera_index, board_w, board_h):
    out = frame.copy()
    color = (0, 220, 80) if found else (0, 80, 255)
    text = "CHESSBOARD OK" if found else "NO CHESSBOARD"
    scale = max(0.75, min(out.shape[1] / 1280.0, out.shape[0] / 720.0))
    font1 = 0.68 * scale
    font2 = 0.48 * scale
    thick = max(1, int(round(2 * scale)))
    bar_h = max(96, int(round(104 * scale)))
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (20, 20, 20), -1)
    cv2.putText(out, f"Quality cam #{camera_index}  {text}", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, font1, color, thick, cv2.LINE_AA)
    cv2.putText(out, f"board={board_w}x{board_h} inner corners  saved={saved_count}  SPACE=save  Q=quit",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, font2, (190, 190, 190), max(1, thick - 1), cv2.LINE_AA)
    return out


def find_chessboard_on_preview(frame, pattern_size, max_detect_w=1600):
    h, w = frame.shape[:2]
    scale = min(float(max_detect_w) / max(float(w), 1.0), 1.0)
    if scale < 0.999:
        small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    found, corners = find_chessboard_corners(gray, pattern_size)
    if found and corners is not None and scale < 0.999:
        corners = corners / scale
    return found, corners


def count_images(path):
    return len(list(Path(path).glob("*.jpg"))) if Path(path).exists() else 0


def main():
    cfg = load_dual_camera_config()

    parser = argparse.ArgumentParser(description="側相機標定照片擷取")
    parser.add_argument("--camera-index", type=int, default=int(cfg.get("quality_camera_index", 0)))
    parser.add_argument("--width", type=int, default=int(cfg.get("quality_width", 3840)))
    parser.add_argument("--height", type=int, default=int(cfg.get("quality_height", 2160)))
    parser.add_argument("--fps", type=int, default=int(cfg.get("quality_fps", 15)))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR)
    parser.add_argument("--board-w", type=int, default=9)
    parser.add_argument("--board-h", type=int, default=6)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--reset", action="store_true", help="備份舊資料並重新開始")
    mode_group.add_argument("--append", action="store_true", help="保留舊資料並繼續新增")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else HERE / args.out_dir
    preview_dir = args.preview_dir if args.preview_dir.is_absolute() else HERE / args.preview_dir
    existing_count = count_images(out_dir)
    if args.reset:
        session_mode = "reset"
    elif args.append:
        session_mode = "append"
    else:
        session_mode = choose_session_mode("畫質相機內參棋盤拍攝", existing_count, default="reset")
    if session_mode == "reset":
        backup = archive_dir(out_dir)
        archive_dir(preview_dir)
        if backup is not None:
            print(f"[資料] 舊資料已備份：{backup}")
    out_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)

    cap = open_camera(args.camera_index, args.width, args.height, args.fps, cfg)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟側相機 index={args.camera_index}")

    print("=" * 60)
    print("側相機標定照片擷取")
    print("=" * 60)
    print(f"相機 index：{args.camera_index}")
    print(f"解析度要求：{args.width} x {args.height} @ {args.fps}fps")
    print(f"輸出資料夾：{out_dir}")
    print("SPACE 儲存，Q/ESC 離開")

    saved_count = len(list(out_dir.glob("*.jpg")))
    pattern_size = (args.board_w, args.board_h)
    win = "Quality Camera Calibration Capture"

    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[警告] 側相機沒有回傳畫面")
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            found, corners = find_chessboard_on_preview(frame, pattern_size)

            debug, show_scale = resize_for_capture_show(frame)
            if found and corners is not None:
                debug = draw_chessboard_visible(debug, pattern_size, corners, show_scale)
            debug = draw_status(debug, found, saved_count, args.camera_index, args.board_w, args.board_h)
            cv2.putText(debug, f"mode={'APPEND' if session_mode == 'append' else 'RESET'}",
                        (14, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, debug)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == 32:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                suffix = "ok" if found else "raw"
                out_path = out_dir / f"quality_calib_{ts}_{saved_count + 1:02d}_{suffix}.jpg"
                cv2.imwrite(str(out_path), frame)
                preview_path = preview_dir / f"preview_{out_path.stem}.jpg"
                cv2.imwrite(str(preview_path), debug)
                saved_count += 1
                print(f"[儲存] {out_path.name}  found={found}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n完成，已儲存 {saved_count} 張到：{out_dir}")
    print("下一步執行：")
    print("python calibrate_camera.py --img-dir quality_calib_images --out-json quality_camera_calib.json --preview-dir quality_calib_preview")


if __name__ == "__main__":
    main()
