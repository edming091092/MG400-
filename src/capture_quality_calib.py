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

from calibrate_camera import find_chessboard_corners, resize_for_show


HERE = Path(__file__).parent
CONFIG_FILE = HERE / "dual_camera_config.json"
OUT_DIR = HERE / "quality_calib_images"
PREVIEW_DIR = HERE / "quality_calib_preview"


def load_dual_camera_config():
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    return cap


def draw_status(frame, found, saved_count, camera_index, board_w, board_h):
    out = frame.copy()
    color = (0, 220, 80) if found else (0, 80, 255)
    text = "CHESSBOARD OK" if found else "NO CHESSBOARD"
    cv2.rectangle(out, (0, 0), (out.shape[1], 74), (20, 20, 20), -1)
    cv2.putText(out, f"Quality cam #{camera_index}  {text}", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
    cv2.putText(out, f"board={board_w}x{board_h} inner corners  saved={saved_count}  SPACE=save  Q=quit",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)
    return out


def main():
    cfg = load_dual_camera_config()

    parser = argparse.ArgumentParser(description="側相機標定照片擷取")
    parser.add_argument("--camera-index", type=int, default=int(cfg.get("quality_camera_index", 0)))
    parser.add_argument("--width", type=int, default=int(cfg.get("quality_width", 1280)))
    parser.add_argument("--height", type=int, default=int(cfg.get("quality_height", 720)))
    parser.add_argument("--fps", type=int, default=int(cfg.get("quality_fps", 30)))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR)
    parser.add_argument("--board-w", type=int, default=9)
    parser.add_argument("--board-h", type=int, default=6)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else HERE / args.out_dir
    preview_dir = args.preview_dir if args.preview_dir.is_absolute() else HERE / args.preview_dir
    out_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)

    cap = open_camera(args.camera_index, args.width, args.height, args.fps)
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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_chessboard_corners(gray, pattern_size)

            debug = frame.copy()
            if found and corners is not None:
                cv2.drawChessboardCorners(debug, pattern_size, corners, found)
            debug = draw_status(debug, found, saved_count, args.camera_index, args.board_w, args.board_h)
            cv2.imshow(win, resize_for_show(debug, 1280, 720))

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
