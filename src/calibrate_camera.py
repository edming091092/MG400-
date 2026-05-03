# calibrate_camera.py
# -*- coding: utf-8 -*-

"""
純黑白棋盤格相機標定工具
========================

功能：
1. 只使用普通黑白棋盤格
2. 不使用 ArUco / ChArUco
3. 自動讀取 calib_images/ 裡的照片
4. 偵測棋盤格角點
5. 儲存角點 debug 圖到 calib_preview/
6. 成功後輸出 camera_calib.json
7. classify.py 可讀取 camera_calib.json 做 undistort

使用方式：
1. 印一張純黑白棋盤格
2. 如果 BOARD_W=9、BOARD_H=6，代表棋盤要有 9×6 內角點，也就是 10×7 格子
3. 拍 15~25 張不同位置、角度的照片
4. 放到 calib_images/
5. 執行：
   python calibrate_camera.py
"""

import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime


# ============================================================
# 棋盤格設定
# ============================================================

# 這裡填「內角點數」，不是格子數
# 常見標定板：9x6 內角點 = 10x7 格子
BOARD_W = 9
BOARD_H = 6

# 每一格實際邊長，單位 mm
# 你印出來後請用尺量，例如 25mm、30mm
SQUARE_MM = 26.0

# 至少有效照片數
MIN_IMAGES = 10

# 是否顯示每張角點偵測結果
SHOW_DEBUG_WINDOW = True

# True = 每張停住，按任意鍵下一張
# False = 每張顯示 300ms 自動下一張
DEBUG_WAIT_KEY = False


# ============================================================
# 路徑設定
# ============================================================

HERE = Path(__file__).parent
IMG_DIR = HERE / "calib_images"
OUT_JSON = HERE / "camera_calib.json"
PREVIEW_DIR = HERE / "calib_preview"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# 工具函式
# ============================================================

def load_images():
    IMG_DIR.mkdir(exist_ok=True)
    paths = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)

    if not paths:
        print(f"[錯誤] 找不到標定照片")
        print(f"請把棋盤格照片放到：{IMG_DIR}")
        return []

    print(f"[讀取] 找到 {len(paths)} 張照片")
    return paths


def resize_for_show(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    return img


def find_chessboard_corners(gray, pattern_size):
    """
    先用快速 findChessboardCorners，
    失敗再用新版 findChessboardCornersSB。
    針對 Gemini 這類固定焦 RGB 畫面，也會嘗試對比增強、銳化與放大偵測。
    """

    def _detect_one(src_gray):
        # 舊版方法加 FAST_CHECK，找不到時會很快返回，避免檢查腳本看起來卡死。
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )

        found, corners = cv2.findChessboardCorners(src_gray, pattern_size, flags)

        if found and corners is not None:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            corners = cv2.cornerSubPix(
                src_gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            return True, corners.astype(np.float32)

        # 新版方法比較穩，但不要開 EXHAUSTIVE/ACCURACY；
        # 那兩個旗標在找不到棋盤時可能會跑非常久。
        if hasattr(cv2, "findChessboardCornersSB"):
            flags = 0

            if hasattr(cv2, "CALIB_CB_NORMALIZE_IMAGE"):
                flags |= cv2.CALIB_CB_NORMALIZE_IMAGE

            found, corners = cv2.findChessboardCornersSB(src_gray, pattern_size, flags)

            if found and corners is not None:
                return True, corners.astype(np.float32)

        return False, None

    def _variants(src_gray):
        yield src_gray, 1.0

        eq = cv2.equalizeHist(src_gray)
        yield eq, 1.0

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(src_gray)
        yield clahe, 1.0

        blur = cv2.GaussianBlur(clahe, (0, 0), sigmaX=1.4)
        sharp = cv2.addWeighted(clahe, 1.8, blur, -0.8, 0)
        yield sharp, 1.0

        for scale in (1.5, 2.0):
            up = cv2.resize(sharp, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            yield up, scale

    for candidate, scale in _variants(gray):
        found, corners = _detect_one(candidate)
        if found and corners is not None:
            if scale != 1.0:
                corners = corners / scale
            return True, corners.astype(np.float32)

    return False, None


def _legacy_find_chessboard_corners(gray, pattern_size):
    """
    保留舊版流程供需要比對時使用；主流程請用 find_chessboard_corners。
    """

    # 新版方法，比較穩
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = 0

        if hasattr(cv2, "CALIB_CB_NORMALIZE_IMAGE"):
            flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            flags |= cv2.CALIB_CB_ACCURACY

        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)

        if found and corners is not None:
            return True, corners.astype(np.float32)

    # 舊版方法
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if found and corners is not None:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        return True, corners.astype(np.float32)

    return False, None


def draw_and_save_debug(img, pattern_size, found, corners, path):
    PREVIEW_DIR.mkdir(exist_ok=True)

    debug = img.copy()

    if found and corners is not None:
        cv2.drawChessboardCorners(debug, pattern_size, corners, found)
        text = f"OK: {pattern_size[0]}x{pattern_size[1]}"
        color = (0, 255, 0)
        filename = f"ok_{pattern_size[0]}x{pattern_size[1]}_{path.stem}.jpg"
    else:
        text = f"NO CORNERS: {pattern_size[0]}x{pattern_size[1]}"
        color = (0, 0, 255)
        filename = f"fail_{pattern_size[0]}x{pattern_size[1]}_{path.stem}.jpg"

    cv2.rectangle(debug, (0, 0), (debug.shape[1], 80), (20, 20, 20), -1)
    cv2.putText(
        debug,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
        cv2.LINE_AA
    )

    out_path = PREVIEW_DIR / filename
    cv2.imwrite(str(out_path), debug)

    if SHOW_DEBUG_WINDOW:
        show = resize_for_show(debug)
        cv2.imshow("Chessboard Debug", show)

        if DEBUG_WAIT_KEY:
            cv2.waitKey(0)
        else:
            cv2.waitKey(300)


def calc_reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0.0
    total_points = 0

    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        error = cv2.norm(img_points[i], projected, cv2.NORM_L2)

        total_error += error ** 2
        total_points += len(obj_points[i])

    mean_error = np.sqrt(total_error / total_points)
    return float(mean_error)


def save_calibration_result(camera_matrix, dist_coeffs, image_size, reproj_error, n_images):
    w, h = image_size

    data = {
        "mode": "chessboard",
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "img_size": [int(w), int(h)],
        "reprojection_error_px": round(float(reproj_error), 4),
        "n_images": int(n_images),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "chessboard": {
            "inner_corners_w": BOARD_W,
            "inner_corners_h": BOARD_H,
            "square_mm": SQUARE_MM
        }
    }

    OUT_JSON.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("標定完成")
    print("=" * 60)
    print("模式：普通黑白棋盤格")
    print(f"有效照片數：{n_images}")
    print(f"影像尺寸：{w} x {h}")
    print(f"重投影誤差：{reproj_error:.4f} px")
    print(f"fx = {camera_matrix[0, 0]:.2f}")
    print(f"fy = {camera_matrix[1, 1]:.2f}")
    print(f"cx = {camera_matrix[0, 2]:.2f}")
    print(f"cy = {camera_matrix[1, 2]:.2f}")
    print(f"dist = {dist_coeffs.ravel()}")
    print(f"已輸出：{OUT_JSON}")

    if reproj_error < 1.0:
        print("品質判斷：良好")
    elif reproj_error < 2.0:
        print("品質判斷：可用，但還能更好")
    else:
        print("品質判斷：偏高，建議重拍標定照片")


def save_undistort_preview(valid_paths, camera_matrix, dist_coeffs):
    PREVIEW_DIR.mkdir(exist_ok=True)

    for i, path in enumerate(valid_paths[:5]):
        img = cv2.imread(str(path))

        if img is None:
            continue

        h, w = img.shape[:2]

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (w, h),
            0,
            (w, h)
        )

        undistorted = cv2.undistort(
            img,
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix
        )

        before = cv2.resize(img, (w // 2, h // 2))
        after = cv2.resize(undistorted, (w // 2, h // 2))

        cv2.putText(
            before,
            "Before",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )

        cv2.putText(
            after,
            "After",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )

        preview = np.hstack([before, after])
        out_path = PREVIEW_DIR / f"undistort_preview_{i+1:02d}.jpg"
        cv2.imwrite(str(out_path), preview)

    print(f"[預覽] 校正前後對比已輸出到：{PREVIEW_DIR}")


# ============================================================
# 標定主流程
# ============================================================

def calibrate_from_images(paths):
    pattern_size = (BOARD_W, BOARD_H)

    print("\n" + "=" * 60)
    print("開始普通棋盤格標定")
    print("=" * 60)
    print(f"內角點數：{BOARD_W} x {BOARD_H}")
    print(f"每格尺寸：{SQUARE_MM} mm")

    # 準備棋盤格世界座標
    objp = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2)
    objp *= SQUARE_MM

    obj_points = []
    img_points = []
    valid_paths = []
    image_size = None

    for path in paths:
        img = cv2.imread(str(path))

        if img is None:
            print(f"  {path.name}: 讀圖失敗")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = find_chessboard_corners(gray, pattern_size)

        draw_and_save_debug(img, pattern_size, found, corners, path)

        if not found:
            print(f"  {path.name}: 找不到角點")
            continue

        obj_points.append(objp.copy())
        img_points.append(corners)
        valid_paths.append(path)

        print(f"  {path.name}: OK，角點數={len(corners)}")

    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("偵測結果")
    print("=" * 60)
    print(f"總照片數：{len(paths)}")
    print(f"有效照片數：{len(valid_paths)}")
    print(f"最低需求：{MIN_IMAGES}")

    if len(valid_paths) < MIN_IMAGES:
        print("\n[失敗] 有效照片數不足，無法標定")
        print("建議：")
        print("1. 確認你印的是純黑白棋盤格，不要有 ArUco 小圖案")
        print("2. BOARD_W / BOARD_H 要填內角點數")
        print("3. 如果棋盤是 10x7 格子，BOARD_W=9, BOARD_H=6")
        print("4. 如果棋盤是 9x6 格子，BOARD_W=8, BOARD_H=5")
        print("5. 棋盤要完整入鏡，不能被裁切")
        print("6. 拍近一點，讓棋盤佔畫面 1/3 到 1/2")
        print("7. 拍 15~25 張不同位置與角度")
        print(f"8. 打開 {PREVIEW_DIR} 查看 fail_*.jpg / ok_*.jpg")
        return None

    print("\n[標定] cv2.calibrateCamera() 執行中...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None
    )

    reproj_error = calc_reprojection_error(
        obj_points,
        img_points,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs
    )

    result = {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_size": image_size,
        "reproj_error": reproj_error,
        "valid_paths": valid_paths
    }

    return result


# ============================================================
# 主程式
# ============================================================

def main():
    global BOARD_W, BOARD_H, SQUARE_MM, MIN_IMAGES
    global SHOW_DEBUG_WINDOW, DEBUG_WAIT_KEY, IMG_DIR, OUT_JSON, PREVIEW_DIR

    parser = argparse.ArgumentParser(description="黑白棋盤格相機內參標定工具")
    parser.add_argument("--img-dir", type=Path, default=IMG_DIR, help="標定照片資料夾")
    parser.add_argument("--out-json", type=Path, default=OUT_JSON, help="輸出的相機內參 JSON")
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR, help="角點與 undistort 預覽輸出資料夾")
    parser.add_argument("--board-w", type=int, default=BOARD_W, help="棋盤格水平內角點數")
    parser.add_argument("--board-h", type=int, default=BOARD_H, help="棋盤格垂直內角點數")
    parser.add_argument("--square-mm", type=float, default=SQUARE_MM, help="每格實際邊長，單位 mm")
    parser.add_argument("--min-images", type=int, default=MIN_IMAGES, help="最低有效照片數")
    parser.add_argument("--no-window", action="store_true", help="不顯示角點 debug 視窗")
    parser.add_argument("--wait-key", action="store_true", help="每張 debug 圖停住，按任意鍵下一張")
    args = parser.parse_args()

    BOARD_W = args.board_w
    BOARD_H = args.board_h
    SQUARE_MM = args.square_mm
    MIN_IMAGES = args.min_images
    SHOW_DEBUG_WINDOW = not args.no_window
    DEBUG_WAIT_KEY = args.wait_key
    IMG_DIR = args.img_dir if args.img_dir.is_absolute() else HERE / args.img_dir
    OUT_JSON = args.out_json if args.out_json.is_absolute() else HERE / args.out_json
    PREVIEW_DIR = args.preview_dir if args.preview_dir.is_absolute() else HERE / args.preview_dir

    print("=" * 60)
    print("純黑白棋盤格相機標定工具")
    print("=" * 60)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"標定照片資料夾：{IMG_DIR}")
    print(f"輸出檔案：{OUT_JSON}")
    print(f"棋盤格內角點：{BOARD_W} x {BOARD_H}")
    print(f"每格尺寸：{SQUARE_MM} mm")
    print("=" * 60)

    paths = load_images()

    if not paths:
        if sys.stdin.isatty():
            input("按 Enter 結束...")
        return

    result = calibrate_from_images(paths)

    if result is None:
        print("\n標定失敗。")
        if sys.stdin.isatty():
            input("按 Enter 結束...")
        return

    save_calibration_result(
        camera_matrix=result["camera_matrix"],
        dist_coeffs=result["dist_coeffs"],
        image_size=result["image_size"],
        reproj_error=result["reproj_error"],
        n_images=len(result["valid_paths"])
    )

    save_undistort_preview(
        valid_paths=result["valid_paths"],
        camera_matrix=result["camera_matrix"],
        dist_coeffs=result["dist_coeffs"]
    )

    print("\n下一步：")
    print("1. 打開 calib_preview/ok_*.jpg，確認角點有沒有畫在正確交叉點上")
    print("2. 打開 calib_preview/undistort_preview_*.jpg，看校正前後是否合理")
    print("3. 如果重投影誤差 < 2 px，可以先拿去跑 classify.py")
    print("4. 如果重投影誤差太高，請重拍照片")
    print("5. classify.py 會讀取同資料夾的 camera_calib.json")

    if sys.stdin.isatty():
        input("\n按 Enter 結束...")


if __name__ == "__main__":
    main()
