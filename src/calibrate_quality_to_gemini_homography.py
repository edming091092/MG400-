# calibrate_quality_to_gemini_homography.py
# -*- coding: utf-8 -*-

"""用棋盤格建立 Quality 影像座標到 Gemini 影像座標的 2D 對位。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).parent
PAIR_DIR = HERE / "stereo_calib_pairs_7x9"
OUT_JSON = HERE / "quality_to_gemini_homography.json"
PREVIEW_DIR = HERE / "homography_preview"


def detect_any(gray, board_w, board_h):
    for pattern in ((board_w, board_h), (board_h, board_w)):
        found, corners = find_chessboard_corners_quick(gray, pattern)
        if found and corners is not None:
            return True, pattern, corners.reshape(-1, 2).astype(np.float32)
    return False, (board_w, board_h), None


def find_chessboard_corners_quick(gray, pattern):
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if found and corners is not None:
        return True, corners

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE if hasattr(cv2, "CALIB_CB_NORMALIZE_IMAGE") else 0
        found, corners = cv2.findChessboardCornersSB(gray, pattern, sb_flags)
        if found and corners is not None:
            return True, corners

    return False, None


def corner_order_candidates(corners, pattern):
    w, h = pattern
    grid = corners.reshape(h, w, 2)
    yield "normal", grid.reshape(-1, 2)
    yield "reverse_all", grid[::-1, ::-1].reshape(-1, 2)
    yield "flip_rows", grid[::-1, :].reshape(-1, 2)
    yield "flip_cols", grid[:, ::-1].reshape(-1, 2)


def fit_best_homography(quality_corners, gemini_corners, pattern):
    best = None
    for order_name, q_ordered in corner_order_candidates(quality_corners, pattern):
        h_mat, inliers = cv2.findHomography(q_ordered, gemini_corners, cv2.RANSAC, 4.0)
        if h_mat is None:
            continue
        projected = cv2.perspectiveTransform(q_ordered.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
        err = np.linalg.norm(projected - gemini_corners, axis=1)
        mean_err = float(np.mean(err))
        inlier_count = int(inliers.sum()) if inliers is not None else len(q_ordered)
        score = (mean_err, -inlier_count)
        if best is None or score < best["score"]:
            best = {
                "homography": h_mat,
                "order": order_name,
                "mean_error_px": mean_err,
                "max_error_px": float(np.max(err)),
                "inliers": inlier_count,
                "quality_corners": q_ordered,
                "gemini_corners": gemini_corners,
                "score": score,
            }
    return best


def fit_all_order_homographies(quality_corners, gemini_corners, pattern):
    candidates = []
    g_candidates = list(corner_order_candidates(gemini_corners, pattern))
    q_candidates = list(corner_order_candidates(quality_corners, pattern))
    for g_order, g_ordered in g_candidates:
        for q_order, q_ordered in q_candidates:
            h_mat, inliers = cv2.findHomography(q_ordered, g_ordered, 0)
            if h_mat is None:
                continue
            projected = cv2.perspectiveTransform(q_ordered.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
            err = np.linalg.norm(projected - g_ordered, axis=1)
            candidates.append({
                "homography": h_mat,
                "g_order": g_order,
                "q_order": q_order,
                "mean_error_px": float(np.mean(err)),
                "quality_corners": q_ordered.astype(np.float32),
                "gemini_corners": g_ordered.astype(np.float32),
            })
    candidates.sort(key=lambda x: x["mean_error_px"])
    return candidates


def choose_consistent_correspondences(pair_candidates):
    """Choose globally consistent checkerboard orientations and drop outliers.

    A plain chessboard has 180-degree / row / column ambiguities.  A single pair
    can be fitted almost perfectly with several wrong corner orders, so a greedy
    "best pair_err" choice can poison the whole calibration.  This routine first
    searches all per-pair homography candidates as possible global seeds, then
    iteratively re-selects the best orientation for each pair under the current
    global H and rejects pairs that do not fit the same camera-to-camera mapping.
    """

    def candidate_error(h_mat, cand):
        projected = cv2.perspectiveTransform(
            cand["quality_corners"].reshape(-1, 1, 2), h_mat
        ).reshape(-1, 2)
        err = np.linalg.norm(projected - cand["gemini_corners"], axis=1)
        return float(np.mean(err)), float(np.max(err)), float(np.median(err))

    def fit_global(selected):
        src = np.vstack([cand["quality_corners"] for _, cand in selected]).astype(np.float32)
        dst = np.vstack([cand["gemini_corners"] for _, cand in selected]).astype(np.float32)
        h_mat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        return h_mat

    best_seed = None
    for seed_idx, candidates in pair_candidates:
        for seed in candidates:
            seed_h = seed["homography"]
            selected = []
            errs = []
            for idx, pair_opts in pair_candidates:
                scored = [(candidate_error(seed_h, cand)[0], cand) for cand in pair_opts]
                scored.sort(key=lambda x: x[0])
                if scored and scored[0][0] <= 12.0:
                    selected.append((idx, scored[0][1]))
                    errs.append(scored[0][0])
            if len(selected) < 4:
                continue
            score = (-len(selected), float(np.median(errs)), float(np.mean(errs)), seed_idx)
            if best_seed is None or score < best_seed["score"]:
                best_seed = {"score": score, "selected": selected}

    if best_seed is None:
        return []

    selected = best_seed["selected"]
    for _ in range(10):
        h_mat = fit_global(selected)
        if h_mat is None:
            break
        next_selected = []
        for idx, pair_opts in pair_candidates:
            scored = []
            for cand in pair_opts:
                mean_err, max_err, _ = candidate_error(h_mat, cand)
                scored.append((mean_err, max_err, cand))
            scored.sort(key=lambda x: x[0])
            if scored and scored[0][0] <= 8.0 and scored[0][1] <= 35.0:
                next_selected.append((idx, scored[0][2]))
        if len(next_selected) < 4:
            break
        same = (
            [idx for idx, _ in next_selected] == [idx for idx, _ in selected]
            and all(
                a["q_order"] == b["q_order"] and a["g_order"] == b["g_order"]
                for (_, a), (_, b) in zip(next_selected, selected)
            )
        )
        selected = next_selected
        if same:
            break

    return selected


def draw_preview(g_img, q_img, homography, out_path):
    g_vis = g_img.copy()
    q_vis = q_img.copy()
    h, w = q_img.shape[:2]
    q_box = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    g_box = cv2.perspectiveTransform(q_box, homography).reshape(-1, 2).astype(int)
    cv2.polylines(g_vis, [g_box], True, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(g_vis, "Yellow box = Quality image projected into Gemini", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    g_panel = cv2.resize(g_vis, (960, 540), interpolation=cv2.INTER_AREA)
    q_panel = cv2.resize(q_vis, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), np.hstack([g_panel, q_panel]))


def print_homography_quality(mean_err, max_err, used_count, total_count):
    print("\n" + "=" * 60)
    print("Quality -> Gemini 對位品質判斷")
    print("=" * 60)
    if mean_err <= 3.0 and max_err <= 25.0:
        print("結果：可以使用。對位誤差低，硬幣中心映射到 Gemini depth 應該穩定。")
    elif mean_err <= 6.0 and max_err <= 50.0:
        print("結果：勉強可用，但建議重拍棋盤讓資料更乾淨。")
        print("改善方法：棋盤放桌面不同位置，拍 10~15 張，不要混入舊資料。")
    else:
        print("結果：不建議使用，誤差太大。")
        print("可能原因：")
        print("1. 新舊棋盤照片混在同一個資料夾一起算")
        print("2. 棋盤被手遮到、反光、模糊，或兩台相機有一邊沒有完整看到")
        print("3. 棋盤方向歧義造成某幾組角點順序選錯")
        print("4. 拍攝過程中相機位置動到")
        print("改善方法：重新拍攝時選「清空舊資料重新開始」，拍 10~15 張分散在桌面四角和中心，再重新計算。")
    if total_count > used_count:
        print(f"提醒：共有 {total_count} 組資料，只使用 {used_count} 組；被略過的照片請檢查是否棋盤不完整或模糊。")


def homography_is_usable(mean_err, max_err, used_count):
    return used_count >= 4 and mean_err <= 6.0 and max_err <= 50.0


def main():
    parser = argparse.ArgumentParser(description="Quality 到 Gemini 的 2D 影像對位校準")
    parser.add_argument("--pair-dir", type=Path, default=PAIR_DIR)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR)
    parser.add_argument("--board-w", type=int, default=9)
    parser.add_argument("--board-h", type=int, default=6)
    args = parser.parse_args()

    pair_dir = args.pair_dir if args.pair_dir.is_absolute() else HERE / args.pair_dir
    out_json = args.out_json if args.out_json.is_absolute() else HERE / args.out_json
    preview_dir = args.preview_dir if args.preview_dir.is_absolute() else HERE / args.preview_dir
    preview_dir.mkdir(exist_ok=True)

    gemini_paths = sorted((pair_dir / "gemini").glob("pair_*_gemini.jpg"))
    quality_paths = sorted((pair_dir / "quality").glob("pair_*_quality.jpg"))
    pairs = list(zip(gemini_paths, quality_paths))

    pair_candidates = []
    print("=" * 60)
    print("Quality -> Gemini 影像對位校準")
    print("=" * 60)
    print(f"配對數：{len(pairs)}")

    for idx, (g_path, q_path) in enumerate(pairs, start=1):
        g_img = cv2.imread(str(g_path))
        q_img = cv2.imread(str(q_path))
        if g_img is None or q_img is None:
            print(f"{idx:02d}: 讀圖失敗")
            continue

        g_ok, g_pattern, g_corners = detect_any(cv2.cvtColor(g_img, cv2.COLOR_BGR2GRAY), args.board_w, args.board_h)
        q_ok, q_pattern, q_corners = detect_any(cv2.cvtColor(q_img, cv2.COLOR_BGR2GRAY), args.board_w, args.board_h)
        if not (g_ok and q_ok):
            print(f"{idx:02d}: 略過 Gemini={g_ok} Quality={q_ok}")
            continue
        if g_pattern != q_pattern:
            print(f"{idx:02d}: 略過，兩邊棋盤方向不同 Gemini={g_pattern} Quality={q_pattern}")
            continue

        candidates = fit_all_order_homographies(q_corners, g_corners, g_pattern)
        if not candidates:
            print(f"{idx:02d}: homography 失敗")
            continue
        pair_candidates.append((idx, candidates))
        best = candidates[0]
        print(f"{idx:02d}: OK pattern={g_pattern} best={best['q_order']}->{best['g_order']} pair_err={best['mean_error_px']:.2f}px")

    selected = choose_consistent_correspondences(pair_candidates)
    all_quality = []
    all_gemini = []
    used = []
    pair_lookup = dict(pairs)
    for idx, best in selected:
        g_path, q_path = pairs[idx - 1]
        all_quality.append(best["quality_corners"])
        all_gemini.append(best["gemini_corners"])
        used.append({
            "index": idx,
            "gemini": str(g_path.relative_to(pair_dir)),
            "quality": str(q_path.relative_to(pair_dir)),
            "pattern": [args.board_w, args.board_h],
            "corner_order": f"quality:{best['q_order']} gemini:{best['g_order']}",
            "pair_mean_error_px": best["mean_error_px"],
        })

    if not used:
        raise RuntimeError("沒有任何可用配對；請先拍到至少一組兩邊都能抓到棋盤的照片")

    src = np.vstack(all_quality).astype(np.float32)
    dst = np.vstack(all_gemini).astype(np.float32)
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if homography is None:
        raise RuntimeError("整體 homography 計算失敗")

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    err = np.linalg.norm(projected - dst, axis=1)
    final_pair_errors = []
    offset = 0
    for item, pts in zip(used, all_quality):
        n = len(pts)
        pair_err = err[offset:offset + n]
        offset += n
        final_pair_errors.append({
            "index": item["index"],
            "mean_error_px": float(np.mean(pair_err)),
            "max_error_px": float(np.max(pair_err)),
        })
    result = {
        "mode": "quality_to_gemini_image_homography",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "description": "Map a Quality camera pixel [x,y] to the corresponding Gemini color/depth pixel [x,y]. No world coordinate conversion.",
        "quality_to_gemini_homography": homography.tolist(),
        "mean_error_px": float(np.mean(err)),
        "max_error_px": float(np.max(err)),
        "inliers": int(inliers.sum()) if inliers is not None else int(len(src)),
        "n_points": int(len(src)),
        "final_pair_errors": final_pair_errors,
        "excluded_pair_indexes": [idx for idx, _ in pair_candidates if idx not in {item["index"] for item in used}],
        "used_pairs": used,
    }
    usable = homography_is_usable(result["mean_error_px"], result["max_error_px"], len(used))
    save_json = out_json
    if not usable:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_json = out_json.with_name(f"{out_json.stem}_FAILED_{stamp}{out_json.suffix}")
        result["not_applied_reason"] = "Calibration error is too high; existing active homography was not overwritten."
    save_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    first = used[0]["index"] - 1
    draw_preview(cv2.imread(str(pairs[first][0])), cv2.imread(str(pairs[first][1])),
                 homography, preview_dir / "quality_to_gemini_preview.jpg")

    print("\n完成")
    print(f"使用配對：{len(used)}")
    print(f"平均對位誤差：{result['mean_error_px']:.2f}px")
    print(f"最大對位誤差：{result['max_error_px']:.2f}px")
    worst = sorted(final_pair_errors, key=lambda x: x["max_error_px"], reverse=True)[:5]
    if worst:
        print("誤差最大的配對：")
        for item in worst:
            print(f"  pair {item['index']:03d}: mean={item['mean_error_px']:.2f}px max={item['max_error_px']:.2f}px")
    print_homography_quality(result["mean_error_px"], result["max_error_px"], len(used), len(pairs))
    excluded = result["excluded_pair_indexes"]
    if excluded:
        print(f"自動排除配對：{', '.join(f'{idx:03d}' for idx in excluded)}")
    if usable:
        print(f"已輸出：{save_json}")
    else:
        print(f"未套用到正式檔，失敗診斷已輸出：{save_json}")
    print(f"預覽圖：{preview_dir / 'quality_to_gemini_preview.jpg'}")


if __name__ == "__main__":
    main()
