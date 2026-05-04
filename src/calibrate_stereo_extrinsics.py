# calibrate_stereo_extrinsics.py
# -*- coding: utf-8 -*-

"""
使用 Gemini 彩色圖 + 側相機配對棋盤格照片，計算雙相機外參。

輸出 stereo_extrinsics.json：
  P_quality = R_gemini_to_quality @ P_gemini + T_gemini_to_quality
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from calibrate_camera import find_chessboard_corners


HERE = Path(__file__).parent
PAIR_DIR = HERE / "stereo_calib_pairs"
QUALITY_CALIB = HERE / "quality_camera_calib.json"
OUT_JSON = HERE / "stereo_extrinsics.json"
PREVIEW_DIR = HERE / "stereo_calib_preview"


def load_quality_calib(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        np.array(data["camera_matrix"], dtype=np.float64),
        np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
        tuple(data["img_size"]),
        data,
    )


def load_gemini_intrinsics(pair_dir):
    metadata_path = pair_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到 {metadata_path}，請先執行 capture_stereo_calib_pairs.py")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    intr = meta["gemini_intrinsics"]
    k = np.array(intr["camera_matrix"], dtype=np.float64)
    d = np.array(intr.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)
    image_size = (int(intr["width"]), int(intr["height"]))
    return k, d, image_size, meta


def collect_pair_paths(pair_dir):
    metadata_path = pair_dir / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        pairs = []
        for item in meta.get("pairs", []):
            g = pair_dir / item["gemini"]
            q = pair_dir / item["quality"]
            if g.exists() and q.exists():
                pairs.append((g, q))
        if pairs:
            return pairs

    gemini_paths = sorted((pair_dir / "gemini").glob("pair_*_gemini.jpg"))
    quality_paths = sorted((pair_dir / "quality").glob("pair_*_quality.jpg"))
    return list(zip(gemini_paths, quality_paths))


def make_object_points(board_w, board_h, square_mm):
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    objp *= float(square_mm)
    return objp


def detect_chessboard_any(gray, board_w, board_h):
    for pattern_size in ((board_w, board_h), (board_h, board_w)):
        found, corners = find_chessboard_corners(gray, pattern_size)
        if found and corners is not None:
            return True, corners.astype(np.float32), pattern_size
    return False, None, (board_w, board_h)


def corner_order_candidates(corners, pattern):
    w, h = pattern
    pts = corners.reshape(-1, 2)
    grid = pts.reshape(h, w, 2)
    yield "normal", grid.reshape(-1, 2)
    yield "reverse_all", grid[::-1, ::-1].reshape(-1, 2)
    yield "flip_rows", grid[::-1, :].reshape(-1, 2)
    yield "flip_cols", grid[:, ::-1].reshape(-1, 2)


def align_quality_corners_to_gemini(g_corners, q_corners, pattern):
    """棋盤格沒有方向標記時，兩台相機可能回傳相反角點順序；用 2D homography 選最佳順序。"""
    g_pts = g_corners.reshape(-1, 2).astype(np.float32)
    best = None
    for order_name, q_pts in corner_order_candidates(q_corners, pattern):
        h_mat, _ = cv2.findHomography(q_pts.astype(np.float32), g_pts, 0)
        if h_mat is None:
            continue
        projected = cv2.perspectiveTransform(q_pts.reshape(-1, 1, 2).astype(np.float32), h_mat).reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(projected - g_pts, axis=1)))
        if best is None or err < best[0]:
            best = (err, order_name, q_pts.astype(np.float32))
    if best is None:
        return q_corners, "unknown", None
    return best[2].reshape(-1, 1, 2), best[1], best[0]


def draw_projection_preview(quality_img, quality_corners, projected, out_path):
    vis = quality_img.copy()
    q = quality_corners.reshape(-1, 2)
    p = projected.reshape(-1, 2)
    for actual, pred in zip(q[::8], p[::8]):
        ax, ay = int(round(actual[0])), int(round(actual[1]))
        px, py = int(round(pred[0])), int(round(pred[1]))
        cv2.circle(vis, (ax, ay), 4, (0, 255, 0), -1)
        cv2.circle(vis, (px, py), 4, (0, 0, 255), 1)
        cv2.line(vis, (ax, ay), (px, py), (0, 200, 255), 1)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(vis, "green=actual quality corners  red=projected from Gemini pose",
                (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def print_stereo_quality(rms, mean_err, max_err, used_count):
    print("\n" + "=" * 60)
    print("雙相機外參品質判斷")
    print("=" * 60)
    check_mean = mean_err if mean_err is not None else rms
    check_max = max_err if max_err is not None else rms
    if rms <= 2.0 and check_mean <= 3.0 and check_max <= 20.0:
        print("結果：可以使用。外參與投影檢查誤差都低。")
    elif rms <= 5.0 and check_mean <= 8.0:
        print("結果：勉強可用，建議先做預覽檢查，不要直接拿來精密抓取。")
    else:
        print("結果：不建議使用，誤差偏大。")
        print("可能原因：")
        print("1. Gemini 與畫質相機照片不是同一個棋盤姿態")
        print("2. 新舊棋盤資料混在同一個資料夾")
        print("3. 內參品質差，或棋盤格尺寸 square-mm 設錯")
        print("4. 棋盤太斜、模糊、反光，角點順序或位置錯")
        print("改善方法：清空舊資料，重新拍 10~20 組同步棋盤，確認兩邊都是同一張棋盤且完整入鏡。")
    if used_count < 10:
        print("提醒：有效配對少於 10 組，建議多拍幾組分散位置。")


def main():
    parser = argparse.ArgumentParser(description="雙相機外參標定")
    parser.add_argument("--pair-dir", type=Path, default=PAIR_DIR)
    parser.add_argument("--quality-calib", type=Path, default=QUALITY_CALIB)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR)
    parser.add_argument("--board-w", type=int, default=9)
    parser.add_argument("--board-h", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=26.0)
    parser.add_argument("--min-pairs", type=int, default=8)
    parser.add_argument("--zero-gemini-dist", action="store_true", help="將 Gemini 彩色畸變係數視為 0；SDK 畸變模型不相容時可改善外參")
    parser.add_argument("--zero-quality-dist", action="store_true", help="將側相機畸變係數視為 0")
    args = parser.parse_args()

    pair_dir = args.pair_dir if args.pair_dir.is_absolute() else HERE / args.pair_dir
    out_json = args.out_json if args.out_json.is_absolute() else HERE / args.out_json
    preview_dir = args.preview_dir if args.preview_dir.is_absolute() else HERE / args.preview_dir
    preview_dir.mkdir(exist_ok=True)

    gemini_k, gemini_d, gemini_size, meta = load_gemini_intrinsics(pair_dir)
    quality_k, quality_d, quality_size, quality_data = load_quality_calib(
        args.quality_calib if args.quality_calib.is_absolute() else HERE / args.quality_calib
    )
    if args.zero_gemini_dist:
        gemini_d = np.zeros_like(gemini_d)
        print("[設定] Gemini dist_coeffs = 0")
    if args.zero_quality_dist:
        quality_d = np.zeros_like(quality_d)
        print("[設定] Quality dist_coeffs = 0")

    pairs = collect_pair_paths(pair_dir)

    obj_points = []
    gemini_points = []
    quality_points = []
    used_pairs = []

    print("=" * 60)
    print("雙相機外參標定")
    print("=" * 60)
    print(f"配對資料夾：{pair_dir}")
    print(f"找到配對：{len(pairs)}")

    for g_path, q_path in pairs:
        g_img = cv2.imread(str(g_path))
        q_img = cv2.imread(str(q_path))
        if g_img is None or q_img is None:
            print(f"  {g_path.name}: 讀圖失敗")
            continue

        g_found, g_corners, g_pattern = detect_chessboard_any(
            cv2.cvtColor(g_img, cv2.COLOR_BGR2GRAY), args.board_w, args.board_h
        )
        q_found, q_corners, q_pattern = detect_chessboard_any(
            cv2.cvtColor(q_img, cv2.COLOR_BGR2GRAY), args.board_w, args.board_h
        )
        if not (g_found and q_found):
            print(f"  {g_path.name}: 角點不足 Gemini={g_found} Quality={q_found}")
            continue
        if g_pattern != q_pattern:
            print(f"  {g_path.name}: 兩邊棋盤方向不同 Gemini={g_pattern} Quality={q_pattern}，略過")
            continue

        q_corners, q_order, order_err = align_quality_corners_to_gemini(g_corners, q_corners, g_pattern)

        objp = make_object_points(g_pattern[0], g_pattern[1], args.square_mm)
        obj_points.append(objp.copy())
        gemini_points.append(g_corners.astype(np.float32))
        quality_points.append(q_corners.astype(np.float32))
        used_pairs.append((g_path, q_path))
        print(f"  {g_path.name}: OK pattern={g_pattern} q_order={q_order} order_err={order_err:.2f}px")

    print(f"\n有效配對數：{len(used_pairs)} / {len(pairs)}")
    if len(used_pairs) < args.min_pairs:
        raise RuntimeError(f"有效配對不足，需要至少 {args.min_pairs} 組；請再拍幾組兩邊都完整看到棋盤格的照片")

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, k1, d1, k2, d2, r, t, e, f = cv2.stereoCalibrate(
        obj_points,
        gemini_points,
        quality_points,
        gemini_k,
        gemini_d,
        quality_k,
        quality_d,
        gemini_size,
        criteria=criteria,
        flags=flags,
    )

    # 交叉驗證：Gemini solvePnP 的棋盤姿態轉到側相機，再投影回側相機畫面。
    errors = []
    sample_previews = 0
    for idx, ((g_path, q_path), obj, gp, qp) in enumerate(zip(used_pairs, obj_points, gemini_points, quality_points), start=1):
        ok, rvec_g, tvec_g = cv2.solvePnP(obj, gp, gemini_k, gemini_d, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        rg, _ = cv2.Rodrigues(rvec_g)
        rq = r @ rg
        tq = r @ tvec_g + t
        rvec_q, _ = cv2.Rodrigues(rq)
        projected, _ = cv2.projectPoints(obj, rvec_q, tq, quality_k, quality_d)
        err = np.linalg.norm(projected.reshape(-1, 2) - qp.reshape(-1, 2), axis=1)
        errors.extend(err.tolist())
        if sample_previews < 5:
            q_img = cv2.imread(str(q_path))
            draw_projection_preview(q_img, qp, projected, preview_dir / f"projection_check_{idx:02d}.jpg")
            sample_previews += 1

    mean_err = float(np.mean(errors)) if errors else None
    max_err = float(np.max(errors)) if errors else None

    result = {
        "mode": "stereo_chessboard_extrinsics",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "convention": "P_quality = R_gemini_to_quality @ P_gemini + T_gemini_to_quality",
        "rms_px": float(rms),
        "projection_check_mean_px": mean_err,
        "projection_check_max_px": max_err,
        "n_pairs": len(used_pairs),
        "board": {
            "inner_corners_w": args.board_w,
            "inner_corners_h": args.board_h,
            "square_mm": args.square_mm,
        },
        "gemini_camera_matrix": gemini_k.tolist(),
        "gemini_dist_coeffs": gemini_d.reshape(-1).tolist(),
        "gemini_img_size": list(gemini_size),
        "quality_camera_matrix": quality_k.tolist(),
        "quality_dist_coeffs": quality_d.reshape(-1).tolist(),
        "quality_img_size": list(quality_size),
        "R_gemini_to_quality": r.tolist(),
        "T_gemini_to_quality_mm": t.reshape(3).tolist(),
        "essential_matrix": e.tolist(),
        "fundamental_matrix": f.tolist(),
        "used_pairs": [
            {
                "gemini": str(g.relative_to(pair_dir)),
                "quality": str(q.relative_to(pair_dir)),
            }
            for g, q in used_pairs
        ],
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("外參標定完成")
    print("=" * 60)
    print(f"stereo RMS：{rms:.4f} px")
    if mean_err is not None:
        print(f"投影檢查：mean={mean_err:.3f}px  max={max_err:.3f}px")
    print_stereo_quality(float(rms), mean_err, max_err, len(used_pairs))
    print(f"T Gemini→Quality：{t.reshape(3)} mm")
    print(f"已輸出：{out_json}")
    print(f"投影預覽：{preview_dir}")


if __name__ == "__main__":
    main()
