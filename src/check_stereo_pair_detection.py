# check_stereo_pair_detection.py
# -*- coding: utf-8 -*-

"""檢查強制儲存的雙相機配對照片是否能離線偵測棋盤格。"""

import argparse
from pathlib import Path

import cv2


HERE = Path(__file__).parent
PAIR_DIR = HERE / "stereo_calib_pairs"


def find_chessboard_corners_quick(gray, pattern_size):
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if found and corners is not None:
        return True, corners

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = 0
        if hasattr(cv2, "CALIB_CB_NORMALIZE_IMAGE"):
            sb_flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
        if found and corners is not None:
            return True, corners

    return False, None


def detect_any(gray, board_w, board_h):
    for pattern in ((board_w, board_h), (board_h, board_w)):
        found, corners = find_chessboard_corners_quick(gray, pattern)
        if found:
            return True, pattern, corners
    return False, (board_w, board_h), None


def main():
    parser = argparse.ArgumentParser(description="檢查雙相機配對圖棋盤偵測")
    parser.add_argument("--pair-dir", type=Path, default=PAIR_DIR)
    parser.add_argument("--board-w", type=int, default=9)
    parser.add_argument("--board-h", type=int, default=6)
    args = parser.parse_args()

    pair_dir = args.pair_dir if args.pair_dir.is_absolute() else HERE / args.pair_dir
    gemini_paths = sorted((pair_dir / "gemini").glob("pair_*_gemini.jpg"))
    quality_paths = sorted((pair_dir / "quality").glob("pair_*_quality.jpg"))
    pairs = list(zip(gemini_paths, quality_paths))

    print("=" * 60)
    print("雙相機配對圖棋盤偵測檢查")
    print("=" * 60)
    print(f"配對數：{len(pairs)}", flush=True)

    ok_pairs = 0
    for idx, (g_path, q_path) in enumerate(pairs, start=1):
        print(f"{idx:02d}: 檢查中... {g_path.name}", flush=True)
        g_img = cv2.imread(str(g_path))
        q_img = cv2.imread(str(q_path))
        if g_img is None or q_img is None:
            print(f"{idx:02d}: 讀圖失敗", flush=True)
            continue

        g_ok, g_pat, _ = detect_any(cv2.cvtColor(g_img, cv2.COLOR_BGR2GRAY), args.board_w, args.board_h)
        q_ok, q_pat, _ = detect_any(cv2.cvtColor(q_img, cv2.COLOR_BGR2GRAY), args.board_w, args.board_h)
        if g_ok and q_ok:
            ok_pairs += 1
        print(f"{idx:02d}: Gemini={g_ok} {g_pat} | Quality={q_ok} {q_pat} | {g_path.name}", flush=True)

    print(f"\n有效配對：{ok_pairs} / {len(pairs)}")
    if ok_pairs < 8:
        print("建議：再拍幾組，讓棋盤更大、更平、更亮，並確認整張棋盤完整入鏡。")


if __name__ == "__main__":
    main()
