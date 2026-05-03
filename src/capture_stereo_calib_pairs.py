# capture_stereo_calib_pairs.py
# -*- coding: utf-8 -*-

"""
Gemini 深度相機 + 側相機外參標定配對照片擷取。

操作：
- SPACE：同步模式下同時儲存一組配對照片
- 分開拍模式：G 拍 Gemini，H 拍 Quality，兩邊都拍到後自動存成一組配對，SPACE 不會拍照
- C：對目前畫面跑一次完整慢速棋盤偵測，只用來檢查
- Q / ESC：離開

建議拍 12~20 組，棋盤格必須同時完整出現在 Gemini 彩色畫面與側相機畫面。
分開拍時，拍完 Gemini 與 Quality 之前不要移動棋盤格。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from calibrate_camera import find_chessboard_corners, resize_for_show
from dual_camera_live import load_config, open_quality_camera
from gemini_controls import apply_color_controls, set_gemini_stream_env


HERE = Path(__file__).parent
GEMINI_LIBS = Path(r"C:\Users\user\Desktop\sam3+座標轉換與夾取")
OUT_DIR = HERE / "stereo_calib_pairs"

if str(GEMINI_LIBS) not in sys.path:
    sys.path.append(str(GEMINI_LIBS))

from core.camera import Gemini2Camera


def intrinsics_to_dict(intr):
    return {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.cx),
        "cy": float(intr.cy),
        "width": int(intr.width),
        "height": int(intr.height),
        "dist_coeffs": np.asarray(intr.dist_coeffs, dtype=np.float64).reshape(-1).tolist(),
        "camera_matrix": intr.K.tolist(),
    }


def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fast_find_chessboard_corners(gray, pattern_size, scale=0.5):
    if scale != 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(small, pattern_size, flags)
    if not found or corners is None:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
    corners = cv2.cornerSubPix(small, corners, (7, 7), (-1, -1), criteria)
    if scale != 1.0:
        corners = corners / scale
    return True, corners.astype(np.float32)


def detect_chessboard(gray, pattern_size, mode="fast"):
    patterns = [pattern_size, (pattern_size[1], pattern_size[0])]
    for pat in patterns:
        if mode == "full":
            found, corners = find_chessboard_corners(gray, pat)
        else:
            found, corners = fast_find_chessboard_corners(gray, pat)
        if found and corners is not None:
            return True, corners, pat
    return False, None, pattern_size


def draw_status(frame, title, found, count, blur=None, detect_note="", photo_mode=True, separate_mode=False):
    out = frame.copy()
    color = (0, 220, 255) if photo_mode else ((0, 220, 80) if found else (0, 80, 255))
    msg = "PHOTO MODE" if photo_mode else ("CHESSBOARD OK" if found else "NO CHESSBOARD")
    blur_text = "" if blur is None else f"  blur={blur:.1f}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 72), (20, 20, 20), -1)
    cv2.putText(out, f"{title}: {msg}{blur_text}", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
    keys = "G=Gemini  H=Quality  C=check  Q=quit" if separate_mode else "SPACE=both  C=check  Q=quit"
    cv2.putText(out, f"saved={count}  {keys} {detect_note}",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1, cv2.LINE_AA)
    return out


def fit_panel(img, width=960, height=540):
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh))
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    panel[y:y + nh, x:x + nw] = resized
    return panel


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="雙相機外參標定配對照片擷取")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--board-w", type=int, default=9)
    parser.add_argument("--board-h", type=int, default=6)
    parser.add_argument("--live-detect", action="store_true", help="開啟即時棋盤偵測；預設關閉，避免卡頓")
    parser.add_argument("--detect-interval", type=int, default=15, help="即時偵測模式下每幾幀跑一次棋盤偵測")
    parser.add_argument("--separate-capture", action="store_true", help="分開按 G/H 拍 Gemini 與側相機，再自動組成一組配對")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else HERE / args.out_dir
    gemini_dir = out_dir / "gemini"
    quality_dir = out_dir / "quality"
    preview_dir = out_dir / "preview"
    for d in (gemini_dir, quality_dir, preview_dir):
        d.mkdir(parents=True, exist_ok=True)

    existing = sorted(gemini_dir.glob("pair_*_gemini.jpg"))
    saved_count = len(existing)
    metadata_path = out_dir / "metadata.json"
    pattern_size = (args.board_w, args.board_h)

    camera = Gemini2Camera(align_depth_to_color=True)
    qcap = None
    try:
        set_gemini_stream_env(cfg)
        camera.open()
        apply_color_controls(camera, cfg)
        qcap, q_status = open_quality_camera(cfg)
        if qcap is None:
            raise RuntimeError(f"側相機開啟失敗：{q_status}")

        metadata = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "board": {
                "inner_corners_w": args.board_w,
                "inner_corners_h": args.board_h,
            },
            "gemini_intrinsics": intrinsics_to_dict(camera.intrinsics),
            "quality_camera_index": int(cfg["quality_camera_index"]),
            "quality_resolution": [int(cfg["quality_width"]), int(cfg["quality_height"])],
            "pairs": [],
        }
        if metadata_path.exists():
            try:
                old = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["pairs"] = old.get("pairs", [])
            except Exception:
                pass

        print("=" * 60)
        print("雙相機外參標定配對拍照")
        print("=" * 60)
        print(f"輸出資料夾：{out_dir}")
        print(f"棋盤格內角點：{args.board_w} x {args.board_h}")
        if args.separate_capture:
            print("模式：分開拍照，不做即時辨識")
            print("G 拍 Gemini，H 拍側相機；兩邊都拍完自動存成一組。拍完兩邊前不要移動棋盤。")
        else:
            print("模式：同步拍照，不做即時辨識")
            print("SPACE 拍一組 Gemini+側相機照片，C 跑一次完整慢速偵測檢查，Q/ESC 離開")

        win = "Stereo Calibration Pair Capture"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1920, 540)

        for _ in range(5):
            camera.get_frames(timeout_ms=1000)
            qcap.read()

        frame_idx = 0
        g_found, q_found = False, False
        g_corners, q_corners = None, None
        g_pattern, q_pattern = pattern_size, pattern_size
        g_blur, q_blur = None, None
        pending_gemini = None
        pending_quality = None

        def save_pair(gemini_frame, quality_frame, tag="photo"):
            nonlocal saved_count
            pair_idx = saved_count + 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = f"pair_{pair_idx:03d}_{ts}_{tag}"
            gemini_path = gemini_dir / f"{stem}_gemini.jpg"
            quality_path = quality_dir / f"{stem}_quality.jpg"
            preview_path = preview_dir / f"{stem}_preview.jpg"
            preview = np.hstack([
                fit_panel(draw_status(gemini_frame, "Gemini color", False, saved_count, blur_score(gemini_frame), "saved pair preview", True, args.separate_capture)),
                fit_panel(draw_status(quality_frame, "Quality side", False, saved_count, blur_score(quality_frame), "saved pair preview", True, args.separate_capture)),
            ])
            cv2.imwrite(str(gemini_path), gemini_frame)
            cv2.imwrite(str(quality_path), quality_frame)
            cv2.imwrite(str(preview_path), preview)
            metadata["pairs"].append({
                "index": pair_idx,
                "gemini": str(gemini_path.relative_to(out_dir)),
                "quality": str(quality_path.relative_to(out_dir)),
                "preview": str(preview_path.relative_to(out_dir)),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            saved_count += 1
            print(f"[拍照] pair {pair_idx:03d} 已儲存，稍後用 run_stereo_check.bat / run_stereo_calib.bat 離線偵測")

        while True:
            frame_idx += 1
            gemini_bgr, _ = camera.get_frames(timeout_ms=1000)
            q_ok, quality_bgr = qcap.read()
            if gemini_bgr is None or not q_ok or quality_bgr is None:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            should_detect = args.live_detect and frame_idx % max(1, args.detect_interval) == 1
            if should_detect:
                gemini_gray = cv2.cvtColor(gemini_bgr, cv2.COLOR_BGR2GRAY)
                quality_gray = cv2.cvtColor(quality_bgr, cv2.COLOR_BGR2GRAY)
                g_found, g_corners, g_pattern = detect_chessboard(gemini_gray, pattern_size, "fast")
                q_found, q_corners, q_pattern = detect_chessboard(quality_gray, pattern_size, "fast")
                g_blur = blur_score(gemini_bgr)
                q_blur = blur_score(quality_bgr)
            elif frame_idx % 20 == 1:
                g_blur = blur_score(gemini_bgr)
                q_blur = blur_score(quality_bgr)

            g_vis = gemini_bgr.copy()
            q_vis = quality_bgr.copy()
            if g_found and g_corners is not None:
                cv2.drawChessboardCorners(g_vis, g_pattern, g_corners, g_found)
            if q_found and q_corners is not None:
                cv2.drawChessboardCorners(q_vis, q_pattern, q_corners, q_found)
            detect_note = f"live detect every {max(1, args.detect_interval)}f" if args.live_detect else "offline detect after shooting"
            if args.separate_capture:
                detect_note = f"separate mode  pending Gemini={pending_gemini is not None} Quality={pending_quality is not None}"
            g_vis = draw_status(g_vis, "Gemini color", g_found, saved_count, g_blur, detect_note, photo_mode=not args.live_detect, separate_mode=args.separate_capture)
            q_vis = draw_status(q_vis, "Quality side", q_found, saved_count, q_blur, detect_note, photo_mode=not args.live_detect, separate_mode=args.separate_capture)
            combined = np.hstack([fit_panel(g_vis), fit_panel(q_vis)])
            cv2.imshow(win, resize_for_show(combined, 1920, 720))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                gemini_gray = cv2.cvtColor(gemini_bgr, cv2.COLOR_BGR2GRAY)
                quality_gray = cv2.cvtColor(quality_bgr, cv2.COLOR_BGR2GRAY)
                print("[檢查] 正在跑完整慢速棋盤偵測...")
                g_found, g_corners, g_pattern = detect_chessboard(gemini_gray, pattern_size, "full")
                q_found, q_corners, q_pattern = detect_chessboard(quality_gray, pattern_size, "full")
                g_blur = blur_score(gemini_bgr)
                q_blur = blur_score(quality_bgr)
                print(f"[檢查] Gemini={g_found} pattern={g_pattern} blur={g_blur:.1f} | Quality={q_found} pattern={q_pattern} blur={q_blur:.1f}")
                continue
            if args.separate_capture and key in (ord("g"), ord("G")):
                pending_gemini = gemini_bgr.copy()
                print("[分開拍] 已拍 Gemini，請不要移動棋盤，接著按 H 拍側相機")
                if pending_quality is not None:
                    save_pair(pending_gemini, pending_quality, "separate")
                    pending_gemini = None
                    pending_quality = None
                continue
            if args.separate_capture and key in (ord("h"), ord("H")):
                pending_quality = quality_bgr.copy()
                print("[分開拍] 已拍側相機，請不要移動棋盤，接著按 G 拍 Gemini")
                if pending_gemini is not None:
                    save_pair(pending_gemini, pending_quality, "separate")
                    pending_gemini = None
                    pending_quality = None
                continue
            if key == 32:
                if args.separate_capture:
                    print("[分開拍] SPACE 已停用；請按 G 拍 Gemini、按 H 拍側相機")
                    continue
                save_pair(gemini_bgr, quality_bgr, "photo")
    finally:
        if qcap is not None:
            qcap.release()
        camera.close()
        cv2.destroyAllWindows()

    print(f"\n完成，配對照片數：{saved_count}")
    print("下一步執行：")
    print("python calibrate_stereo_extrinsics.py")


if __name__ == "__main__":
    main()
