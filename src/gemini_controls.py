# gemini_controls.py
# -*- coding: utf-8 -*-

"""Gemini/Orbbec 相機屬性控制工具。"""

import os


def _get_ob():
    import pyorbbecsdk as ob

    return ob


def _device_from_camera(camera):
    pipeline = getattr(camera, "_pipeline", None)
    if pipeline is None:
        return None
    try:
        return pipeline.get_device()
    except Exception:
        return None


def _int_range(device, prop):
    try:
        r = device.get_int_property_range(prop)
        return int(r.min), int(r.max), int(r.step)
    except Exception:
        return None


def get_color_control_ranges(camera):
    ob = _get_ob()
    device = _device_from_camera(camera)
    if device is None:
        return {}
    ranges = {}
    for key, prop in (
        ("exposure", ob.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT),
        ("gain", ob.OBPropertyID.OB_PROP_COLOR_GAIN_INT),
        ("max_gain", ob.OBPropertyID.OB_PROP_COLOR_MAXIMAL_GAIN_INT),
        ("ae_max_exposure", ob.OBPropertyID.OB_PROP_COLOR_AE_MAX_EXPOSURE_INT),
    ):
        value = _int_range(device, prop)
        if value is not None:
            ranges[key] = value
    return ranges


def get_color_controls(camera):
    ob = _get_ob()
    device = _device_from_camera(camera)
    if device is None:
        return {}
    values = {}
    try:
        values["auto_exposure"] = bool(device.get_bool_property(ob.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL))
    except Exception:
        pass
    try:
        values["exposure"] = int(device.get_int_property(ob.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT))
    except Exception:
        pass
    try:
        values["gain"] = int(device.get_int_property(ob.OBPropertyID.OB_PROP_COLOR_GAIN_INT))
    except Exception:
        pass
    return values


def apply_color_controls(camera, cfg, verbose=True):
    """套用 dual_camera_config.json 裡的 Gemini 彩色相機曝光設定。"""
    ob = _get_ob()
    device = _device_from_camera(camera)
    if device is None:
        if verbose:
            print("[Gemini曝光] 找不到 device，略過")
        return

    auto_exposure = cfg.get("gemini_color_auto_exposure")
    exposure = cfg.get("gemini_color_exposure")
    gain = cfg.get("gemini_color_gain")

    if auto_exposure is not None:
        try:
            device.set_bool_property(ob.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, bool(auto_exposure))
            if verbose:
                print(f"[Gemini曝光] auto_exposure={bool(auto_exposure)}")
        except Exception as e:
            if verbose:
                print(f"[Gemini曝光] 設定 auto_exposure 失敗：{e}")

    if exposure is not None and not bool(auto_exposure):
        try:
            device.set_int_property(ob.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, int(exposure))
            if verbose:
                print(f"[Gemini曝光] exposure={int(exposure)}")
        except Exception as e:
            if verbose:
                print(f"[Gemini曝光] 設定 exposure 失敗：{e}")

    if gain is not None and not bool(auto_exposure):
        try:
            device.set_int_property(ob.OBPropertyID.OB_PROP_COLOR_GAIN_INT, int(gain))
            if verbose:
                print(f"[Gemini曝光] gain={int(gain)}")
        except Exception as e:
            if verbose:
                print(f"[Gemini曝光] 設定 gain 失敗：{e}")


def set_gemini_stream_env(cfg):
    width = cfg.get("gemini_color_width")
    height = cfg.get("gemini_color_height")
    fps = cfg.get("gemini_color_fps")
    fmt = cfg.get("gemini_color_format")
    if width:
        os.environ["GEMINI_COLOR_WIDTH"] = str(int(width))
    if height:
        os.environ["GEMINI_COLOR_HEIGHT"] = str(int(height))
    if fps:
        os.environ["GEMINI_COLOR_FPS"] = str(int(fps))
    if fmt:
        os.environ["GEMINI_COLOR_FORMAT"] = str(fmt).upper()
