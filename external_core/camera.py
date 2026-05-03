"""
Orbbec Gemini 2 相機介面
使用 pyorbbecsdk（conda env: sam3_env, Python 3.12）

執行方式：
  C:/Users/admin/miniconda3/envs/sam3_env/python.exe <script>.py
"""

import traceback
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: np.ndarray  # (k1,k2,p1,p2,k3)

    @property
    def K(self) -> np.ndarray:
        """3x3 相機內參矩陣"""
        return np.array([
            [self.fx,    0,    self.cx],
            [   0,    self.fy, self.cy],
            [   0,       0,       1  ],
        ], dtype=np.float64)


# ── pyorbbecsdk 內建工具（若可用） ─────────────────────────────────
try:
    from pyorbbecsdk.examples.utils import frame_to_bgr_image as _frame_to_bgr_image
except Exception:
    _frame_to_bgr_image = None


def _color_frame_to_bgr(color_frame) -> Optional[np.ndarray]:
    """將 Orbbec 彩色幀轉為 BGR ndarray，相容多種像素格式"""
    if color_frame is None:
        return None

    # 優先使用 SDK 提供的工具函式
    if _frame_to_bgr_image is not None:
        try:
            img = _frame_to_bgr_image(color_frame)
            if img is not None:
                return img
        except Exception:
            pass

    w   = color_frame.get_width()
    h   = color_frame.get_height()
    fmt = str(color_frame.get_format())
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    try:
        if "MJPG" in fmt or "JPEG" in fmt:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if "RGB" in fmt:
            return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
        if "BGR" in fmt:
            return data.reshape((h, w, 3)).copy()
        if "YUYV" in fmt or "YUY2" in fmt:
            return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_YUY2)
        if "UYVY" in fmt:
            return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_UYVY)
    except Exception as e:
        print(f"[camera] 彩色幀轉換失敗：{e}")
        traceback.print_exc()
        return None

    print(f"[camera] 警告：不支援的彩色格式 {fmt}")
    return None


def _depth_frame_to_mm(depth_frame) -> Optional[np.ndarray]:
    """將 Orbbec 深度幀轉為 float32 ndarray（單位 mm）"""
    if depth_frame is None:
        return None
    try:
        w     = depth_frame.get_width()
        h     = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        data  = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        if data.size != w * h:
            print(f"[camera] 深度幀大小不符：expected={w*h}, got={data.size}")
            return None
        depth = data.reshape((h, w)).copy()
        if scale is not None and float(scale) > 0:
            depth = depth.astype(np.float32) * float(scale)
        else:
            depth = depth.astype(np.float32)
        return depth
    except Exception as e:
        print(f"[camera] 深度幀轉換失敗：{e}")
        traceback.print_exc()
        return None


# ======================================================================
#  Gemini2Camera
# ======================================================================

class Gemini2Camera:
    """
    Gemini 2 深度相機包裝器。
    同時取得已對齊的彩色影像與深度圖。

    用法：
        camera = Gemini2Camera()
        camera.open()
        color_bgr, depth_mm = camera.get_frames()
        camera.close()
    """

    def __init__(self, align_depth_to_color: bool = True):
        self._pipeline     = None
        self._align_filter = None
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._align        = align_depth_to_color

    # ------------------------------------------------------------------ #
    #  開啟 / 關閉
    # ------------------------------------------------------------------ #
    def open(self):
        try:
            import pyorbbecsdk as ob
        except ImportError as e:
            raise ImportError(
                "找不到 pyorbbecsdk！\n"
                "請使用 sam3_env conda 環境執行：\n"
                "  C:/Users/admin/miniconda3/envs/sam3_env/python.exe <script>.py\n"
                f"原始錯誤：{e}"
            ) from e

        pipeline = ob.Pipeline()
        cfg      = ob.Config()

        # --- 彩色串流：依序嘗試多種格式 ---
        color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = None
        requested_profiles = []
        try:
            req_w = int(os.environ.get("GEMINI_COLOR_WIDTH", "0"))
            req_h = int(os.environ.get("GEMINI_COLOR_HEIGHT", "0"))
            req_fps = int(os.environ.get("GEMINI_COLOR_FPS", "0"))
            req_fmt_name = os.environ.get("GEMINI_COLOR_FORMAT", "").upper()
            req_fmt = getattr(ob.OBFormat, req_fmt_name, None) if req_fmt_name else None
            if req_w > 0 and req_h > 0 and req_fps > 0 and req_fmt is not None:
                requested_profiles.append((req_fmt, req_w, req_h, req_fps))
        except Exception:
            requested_profiles = []
        for fmt, w, h, fps in requested_profiles + [
            (ob.OBFormat.RGB,  1280, 720, 30),
            (ob.OBFormat.RGB,  640,  480, 30),
            (ob.OBFormat.MJPG, 1280, 720, 30),
            (ob.OBFormat.MJPG, 640,  480, 30),
        ]:
            try:
                color_profile = color_profiles.get_video_stream_profile(w, h, fmt, fps)
                print(f"[camera] 彩色串流：{w}x{h} {fmt} {fps}fps")
                break
            except Exception:
                continue
        if color_profile is None:
            color_profile = color_profiles.get_default_video_stream_profile()
            print("[camera] 彩色串流：使用預設格式")
        cfg.enable_stream(color_profile)

        # --- 深度串流（預設解析度）---
        depth_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = depth_profiles.get_video_stream_profile(0, 0, ob.OBFormat.Y16, 0)
        except Exception:
            depth_profile = depth_profiles.get_default_video_stream_profile()
        cfg.enable_stream(depth_profile)

        # --- 確保 color+depth 同時輸出 ---
        try:
            cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception as e:
            print(f"[camera] 警告：set_frame_aggregate_output_mode 失敗（{e}）")

        # --- 幀同步 ---
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            print(f"[camera] 警告：enable_frame_sync 失敗（{e}）")

        pipeline.start(cfg)
        self._pipeline = pipeline
        print("[camera] Pipeline 已啟動")

        # --- 深度對齊到彩色（AlignFilter）---
        if self._align:
            try:
                self._align_filter = ob.AlignFilter(
                    align_to_stream=ob.OBStreamType.COLOR_STREAM
                )
                print("[camera] AlignFilter 啟用")
            except Exception as e:
                self._align_filter = None
                print(f"[camera] 警告：AlignFilter 不可用（{e}），繼續無對齊模式")

        # --- 取得內參 ---
        self._intrinsics = self._read_intrinsics(pipeline)
        print(f"[camera] 開啟成功  解析度={self._intrinsics.width}x{self._intrinsics.height}")
        print(f"[camera] 內參  fx={self._intrinsics.fx:.2f}  fy={self._intrinsics.fy:.2f}"
              f"  cx={self._intrinsics.cx:.2f}  cy={self._intrinsics.cy:.2f}")

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        print("[camera] 相機已關閉")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------ #
    #  取得一幀
    # ------------------------------------------------------------------ #
    def get_frames(self, timeout_ms: int = 500
                   ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        回傳 (color_bgr, depth_mm)
        color_bgr : uint8 H×W×3 BGR
        depth_mm  : float32 H×W，單位 mm，無效點為 0
        任一失敗回 (None, None)
        """
        if self._pipeline is None:
            raise RuntimeError("相機尚未開啟，請先呼叫 open()")

        frames = self._pipeline.wait_for_frames(timeout_ms)
        if not frames:
            return None, None

        # --- 對齊 ---
        if self._align_filter is not None:
            try:
                aligned = self._align_filter.process(frames)
                if aligned:
                    try:
                        frames = aligned.as_frame_set()
                    except Exception:
                        frames = aligned
            except Exception as e:
                print(f"[camera] 警告：對齊處理失敗（{e}）")

        # --- 取出幀 ---
        try:
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
        except Exception as e:
            print(f"[camera] 警告：取幀失敗（{e}）")
            return None, None

        color_bgr = _color_frame_to_bgr(color_frame)
        depth_mm  = _depth_frame_to_mm(depth_frame)

        if color_bgr is None:
            return None, None

        # --- 深度圖若大小不同則 resize ---
        if depth_mm is not None and (
            depth_mm.shape[1] != color_bgr.shape[1] or
            depth_mm.shape[0] != color_bgr.shape[0]
        ):
            depth_mm = cv2.resize(
                depth_mm,
                (color_bgr.shape[1], color_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        return color_bgr, depth_mm

    # ------------------------------------------------------------------ #
    #  取得深度值（單點或小區域中位數）
    # ------------------------------------------------------------------ #
    @staticmethod
    def sample_depth(depth_mm: np.ndarray, u: int, v: int, radius: int = 5) -> float:
        h, w = depth_mm.shape
        u0, u1 = max(0, u - radius), min(w, u + radius + 1)
        v0, v1 = max(0, v - radius), min(h, v + radius + 1)
        patch = depth_mm[v0:v1, u0:u1]
        valid = patch[patch > 0]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  屬性
    # ------------------------------------------------------------------ #
    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("相機尚未開啟")
        return self._intrinsics

    # ------------------------------------------------------------------ #
    #  內部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_intrinsics(pipeline) -> CameraIntrinsics:
        try:
            param = pipeline.get_camera_param()
            intr  = param.rgb_intrinsic
            try:
                dist = param.rgb_distortion
                dist_coeffs = np.array(
                    [dist.k1, dist.k2, dist.p1, dist.p2, dist.k3],
                    dtype=np.float64
                )
            except Exception:
                dist_coeffs = np.zeros(5, dtype=np.float64)

            return CameraIntrinsics(
                fx=float(intr.fx), fy=float(intr.fy),
                cx=float(intr.cx), cy=float(intr.cy),
                width=int(intr.width), height=int(intr.height),
                dist_coeffs=dist_coeffs,
            )
        except Exception as e:
            print(f"[camera] 警告：無法取得內參（{e}），使用預設值")
            return CameraIntrinsics(
                fx=912.0, fy=912.0, cx=640.0, cy=360.0,
                width=1280, height=720,
                dist_coeffs=np.zeros(5, dtype=np.float64),
            )
