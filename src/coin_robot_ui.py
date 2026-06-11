# -*- coding: utf-8 -*-
"""
Coin sorting control UI.

Product-style wrapper for:
- dual camera recognition snapshots
- coin counts / total value
- MG400 tabletop coordinates
- safe hover and dry-lower tests without vacuum/DO
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.robot import MG400

try:
    from unstack_coins import (
        _bbox_overlap_risk,
        _cfg_bool as _unstack_cfg_bool,
        _round_fallback_sort_key,
        _is_round_fallback_target_visible,
        _stack_height_mm,
        _target_depth,
        _top_visible_sort_key,
        is_pick_range_target,
        is_safe_xy,
        is_stacked_top_target,
        _is_round_fallback_target,
    )
except Exception:
    _bbox_overlap_risk = None
    _unstack_cfg_bool = None
    _round_fallback_sort_key = None
    _is_round_fallback_target_visible = None
    _stack_height_mm = None
    _target_depth = None
    _top_visible_sort_key = None
    is_pick_range_target = None
    is_safe_xy = None
    is_stacked_top_target = None
    _is_round_fallback_target = None


HERE = Path(__file__).parent
TARGETS_FILE = HERE / "robot_targets.json"
ACTION_STATUS_FILE = HERE / "robot_action_status.json"
OUT_DIR = HERE / "test_output"
CONFIG_FILE = HERE / "dual_camera_config.json"
COIN_PARAM_DB_FILE = HERE / "coin_param_db.json"
PYTHON = Path(r"C:\Users\user\miniconda3\envs\coin\python.exe")
SUBPROCESS_KW = {}
if hasattr(subprocess, "CREATE_NO_WINDOW"):
    SUBPROCESS_KW["creationflags"] = subprocess.CREATE_NO_WINDOW
OFFSET_SAFE_X_MIN = 100.0
OFFSET_SAFE_X_MAX = 390.0
OFFSET_SAFE_Y_MIN = -310.0
OFFSET_SAFE_Y_MAX = 220.0
OFFSET_SAFE_Z_MIN = -162.0
OFFSET_SAFE_Z_MAX = 180.0
OFFSET_START_Z = -155.0
DEFAULT_RETURN_X = 30.0
DEFAULT_RETURN_Y = 280.0
DEFAULT_RETURN_Z = 150.0

ROBOT_ERROR_HINTS = {
    23: {
        "title": "路徑被拒絕 / 動作中斷",
        "cause": "控制器停止了移動指令。常見原因是直線路徑不可達、接近關節/軟體極限、碰撞偵測觸發，或路徑不安全。",
        "action": "檢查障礙物，必要時提高轉移高度、降低速度，或先手動把手臂移到安全位置再重試。",
    },
    32: {
        "title": "控制器運動狀態異常",
        "cause": "控制器回報一組異常狀態值，常見於前一次運動未結束、姿態不可達、報警未完全清除，或起始避讓點離目前姿態太難規劃。",
        "action": "清除報警並啟用手臂，手動確認手臂在安全位置，再用較中央的相機避讓位置重試。",
    },
    96: {
        "title": "手臂未真正就緒 / 運動被拒",
        "cause": "EnableRobot 回覆異常或控制器尚未進入可運動狀態，導致前往避讓點時被中斷。",
        "action": "清除報警並啟用手臂，確認 DobotStudio 顯示已啟用且無 alarm，再重試。",
    },
    98: {
        "title": "手臂未就緒 / 報警狀態",
        "cause": "手臂可能仍在停用或報警狀態，尚未清除錯誤。",
        "action": "按「清除報警並啟用手臂」，確認 DobotStudio 內手臂已啟用，再從安全位置重試。",
    },
    2: {
        "title": "控制器報警 / 動作暫停",
        "cause": "控制器在動作中進入錯誤或暫停狀態。常見原因是路徑規劃失敗、接近工作範圍邊界、碰撞偵測，或前一次報警沒有完全清除。",
        "action": "停止流程，按「清除報警並啟用手臂」，移到安全位置，降低速度或跳過邊界目標後再試。",
    },
    17: {
        "title": "目標靠邊 / 高空轉移不可達",
        "cause": "選到的硬幣太靠近可靠工作範圍邊界，或手臂無法安全規劃到該 X/Y 的高空轉移路徑。",
        "action": "清除報警並啟用手臂，跳過這顆邊界硬幣，改選更靠中間且標示可取的硬幣。",
    },
    18: {
        "title": "路徑/下降報警",
        "cause": "控制器拒絕這段自動路徑，可能是高位轉移姿態不可達、下降高度太低、X/Y 還有局部偏差、碰撞偵測觸發，或目標靠近邊界。",
        "action": "清除報警並啟用手臂，先用單顆高位移動確認；若手動可到但自動失敗，改用較中央硬幣或讓程式用 MovL 備援路徑重試。",
    },
    66: {
        "title": "避讓點/起始姿態不可達",
        "cause": "控制器拒絕移到相機避讓位置。常見原因是避讓 Z 設得太低、路徑會低空橫移、目前姿態到目標姿態不可達，或手臂仍有未清除的 alarm。",
        "action": "清除報警並啟用手臂，確認避讓位置使用高 Z，再重試拆堆。若仍失敗，先用「回到相機避讓位置」單獨測試。",
    },
}

UI_TEXT = {
    "zh": {
        "title": "硬幣辨識手臂控制",
        "ready": "就緒",
        "busy": "執行中",
        "settings": "設定",
        "live_camera": "即時相機畫面",
        "vision_note": "按下手臂動作時才會自動辨識並鎖定座標",
        "recognizing": "正在辨識並鎖定座標...",
        "no_image": "尚無影像",
        "summary": "辨識結果",
        "total": "總金額",
        "valid_targets": "可取數量",
        "robot_controls": "手臂控制",
        "estop": "急停 / 停用手臂",
        "clear_enable": "清除報警並啟用手臂",
        "move_clear": "回到相機避讓位置",
        "hover_first": "辨識後移到第一顆上方",
        "lower_first": "辨識後下降第一顆到教點 Z",
        "unstack_coins": "拆堆到暫放區",
        "set_roi": "設定辨識 ROI",
        "refresh": "重新辨識 / 鎖定座標",
        "hover_selected": "移到選取硬幣上方",
        "lower_selected": "下降選取硬幣到教點 Z",
        "lower_all": "逐顆下降所有可取硬幣",
        "current_target": "目前目標",
        "no_target_selected": "尚未選取目標",
        "current_action_idle": "目前動作：待機",
        "dry_only": "目前不開真空/DO，只做乾跑測試。",
        "coin_targets": "硬幣目標",
        "index": "編號",
        "class": "幣別",
        "diameter": "直徑",
        "status": "狀態",
        "ok": "可取",
        "check": "檢查",
        "operator": "操作員模式",
        "engineer": "工程師模式",
        "offset_calib": "誤差校正模式",
        "offset_calib_title": "誤差校正",
        "offset_calib_hint": "放一顆硬幣，按開始到硬幣上方，再用搖桿對準中心後教點。",
        "start_offset_calib": "開始到選取硬幣上方",
        "teach_offset_add": "新增公差樣本",
        "teach_offset_replace": "重新教點",
        "reset_offset_samples": "清空重測公差",
        "language": "語言",
        "mode": "模式",
        "camera": "相機",
        "quality": "畫質相機",
        "gemini": "深度相機",
        "combined": "雙相機",
        "move_speed": "移動速度",
        "lower_speed": "下降速度",
        "save": "套用",
        "cancel": "取消",
    },
    "en": {
        "title": "CoinVision MG400 Control",
        "ready": "Ready",
        "busy": "Busy",
        "settings": "Settings",
        "live_camera": "Live Camera",
        "vision_note": "Recognition runs automatically only when a robot action starts",
        "recognizing": "Recognizing and locking target coordinates...",
        "no_image": "No image yet",
        "summary": "Summary",
        "total": "Total",
        "valid_targets": "Pickable",
        "robot_controls": "MG400 Controls",
        "estop": "Emergency Stop / Disable",
        "clear_enable": "Clear Alarm + Enable",
        "move_clear": "Move to Camera-Clear Pose",
        "hover_first": "Detect then Hover First OK",
        "lower_first": "Detect then Lower First OK to taught Z",
        "unstack_coins": "Unstack to Staging Slots",
        "set_roi": "Set Detection ROI",
        "refresh": "Refresh / Lock Vision",
        "hover_selected": "Hover Selected Coin",
        "lower_selected": "Lower Selected Coin to taught Z",
        "lower_all": "Lower All Pickable Coins",
        "current_target": "Current Target",
        "no_target_selected": "No target selected",
        "current_action_idle": "Current action: idle",
        "dry_only": "Vacuum/DO disabled. Dry-run only.",
        "coin_targets": "Coin Targets",
        "index": "Q",
        "class": "Class",
        "diameter": "Diameter",
        "status": "Status",
        "ok": "OK",
        "check": "Check",
        "operator": "Operator Mode",
        "engineer": "Engineer Mode",
        "offset_calib": "Offset Calibration",
        "offset_calib_title": "Offset Calibration",
        "offset_calib_hint": "Place one coin, start above it, jog to the real center, then teach.",
        "start_offset_calib": "Start Above Selected Coin",
        "teach_offset_add": "Add Offset Sample",
        "teach_offset_replace": "Retake Offset",
        "reset_offset_samples": "Clear And Retake",
        "language": "Language",
        "mode": "Mode",
        "camera": "Camera",
        "quality": "Quality",
        "gemini": "Gemini",
        "combined": "Combined",
        "move_speed": "Move Speed",
        "lower_speed": "Lower Speed",
        "save": "Apply",
        "cancel": "Cancel",
    },
}


class CoinRobotUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.ui_config = self._load_config()
        self.ui_language = str(self.ui_config.get("ui_language", "zh"))
        if self.ui_language not in UI_TEXT:
            self.ui_language = "zh"
        self.ui_mode = str(self.ui_config.get("ui_mode", "operator"))
        if self.ui_mode not in ("operator", "engineer", "offset_calib"):
            self.ui_mode = "operator"
        self.title(self._t("title"))
        self.geometry("1280x800")
        self.minsize(1120, 700)
        self.configure(bg="#202326")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.targets_data = {}
        self.selected_index = tk.IntVar(value=0)
        self.camera_view = tk.StringVar(value="Quality")
        self.busy = False
        self.preview_busy = False
        self.pending_robot_action = None
        self._photo = None
        self.last_failure_signature = None
        self._session_started_at = time.time()
        self._last_image_path = None
        self._display_image_size = (0, 0)
        self._display_image_offset = (0, 0)
        self._loaded_image_size = (0, 0)
        self._zoom_center = (0.5, 0.5)
        self._quality_cap = None
        self._quality_thread = None
        self._quality_stop = threading.Event()
        self._quality_lock = threading.Lock()
        self._quality_latest_frame = None
        self._quality_latest_id = 0
        self._quality_displayed_id = 0
        self._quality_preview_error_logged = False
        self._quality_preview_error_after = None
        self._yolo_live_enabled = False
        self._yolo_live_thread = None
        self._yolo_live_stop = threading.Event()
        self._yolo_live_model = None
        self._yolo_live_model_path = None
        self._yolo_live_last_frame_id = 0
        self._yolo_live_last_draw_at = 0.0
        self._yolo_live_tracks = []
        self._yolo_live_next_track_index = 1
        self._display_pick_plan = []
        self._display_pick_plan_until = 0.0
        self._display_pick_plan_source = None
        self._planned_action_text = None
        self._planned_action_until = 0.0
        self._active_robot_state = "idle"
        self._active_robot_target_index = None
        self._active_robot_target = None
        self._offset_hold_after = None
        self._offset_hold_vector = None
        self._offset_robot = None
        self._offset_robot_lock = threading.Lock()
        self._offset_jog_busy = False
        self._offset_jog_pose = None
        self._offset_jog_warned = False
        self._preview_request_times = {}
        self._gemini_preview_pil = None
        self._gemini_preview_mtime = 0.0
        self._last_gemini_preview_request = 0.0
        self.zoom = 1.0
        self.auto_preview = tk.BooleanVar(value=True)
        self.move_speed_var = tk.IntVar(value=int(self.ui_config.get("ui_move_speed", 40)))
        self.lower_speed_var = tk.IntVar(value=int(self.ui_config.get("ui_lower_speed", 25)))
        self.pick_offset_x_var = tk.DoubleVar(value=float(self.ui_config.get("robot_target_offset_x_mm", 0.0)))
        self.pick_offset_y_var = tk.DoubleVar(value=float(self.ui_config.get("robot_target_offset_y_mm", 0.0)))
        self.pick_offset_z_var = tk.DoubleVar(value=float(self.ui_config.get("robot_target_offset_z_mm", 0.0)))
        self.offset_jog_step_var = tk.DoubleVar(value=float(self.ui_config.get("offset_calib_jog_step_mm", 1.0)))
        self.pick_x_min_var = tk.DoubleVar(value=float(self.ui_config.get("robot_pick_x_min_mm", 120.0)))
        self.pick_x_max_var = tk.DoubleVar(value=float(self.ui_config.get("robot_pick_x_max_mm", 380.0)))
        self.pick_y_min_var = tk.DoubleVar(value=float(self.ui_config.get("robot_pick_y_min_mm", -250.0)))
        self.pick_y_max_var = tk.DoubleVar(value=float(self.ui_config.get("robot_pick_y_max_mm", 190.0)))
        self.safe_x_min_var = tk.DoubleVar(value=float(self.ui_config.get("robot_safe_x_min_mm", OFFSET_SAFE_X_MIN)))
        self.safe_x_max_var = tk.DoubleVar(value=float(self.ui_config.get("robot_safe_x_max_mm", OFFSET_SAFE_X_MAX)))
        self.safe_y_min_var = tk.DoubleVar(value=float(self.ui_config.get("robot_safe_y_min_mm", OFFSET_SAFE_Y_MIN)))
        self.safe_y_max_var = tk.DoubleVar(value=float(self.ui_config.get("robot_safe_y_max_mm", OFFSET_SAFE_Y_MAX)))
        self.safe_z_min_var = tk.DoubleVar(value=float(self.ui_config.get("robot_safe_z_min_mm", OFFSET_SAFE_Z_MIN)))
        self.safe_z_max_var = tk.DoubleVar(value=float(self.ui_config.get("robot_safe_z_max_mm", OFFSET_SAFE_Z_MAX)))
        self.return_x_var = tk.DoubleVar(value=float(self.ui_config.get("robot_return_x_mm", DEFAULT_RETURN_X)))
        self.return_y_var = tk.DoubleVar(value=float(self.ui_config.get("robot_return_y_mm", DEFAULT_RETURN_Y)))
        self.return_z_var = tk.DoubleVar(value=float(self.ui_config.get("robot_return_z_mm", DEFAULT_RETURN_Z)))
        self.offset_calib_travel_z_var = tk.DoubleVar(value=float(self.ui_config.get("offset_calib_travel_z_mm", 100.0)))
        self.offset_calib_target_z_var = tk.DoubleVar(value=float(self.ui_config.get("offset_calib_target_z_mm", OFFSET_START_Z)))

        self._build_styles()
        self._build_layout()
        self._reset_stale_action_status()
        if self.camera_view.get() == "Quality":
            self._start_quality_preview_thread()
        else:
            self._show_preview_message(self._t("no_image"))
        self.after(1000, self._tick)
        self.after(700, self._auto_preview_loop)

    def _t(self, key):
        return UI_TEXT.get(self.ui_language, UI_TEXT["zh"]).get(key, key)

    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background="#202326")
        style.configure("Panel.TFrame", background="#2b2f33")
        style.configure("TLabel", background="#202326", foreground="#d7dde2")
        style.configure("Panel.TLabel", background="#2b2f33", foreground="#d7dde2")
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), background="#202326", foreground="#f4f7f9")
        style.configure("Metric.TLabel", font=("Segoe UI", 18, "bold"), background="#2b2f33", foreground="#f4f7f9")
        style.configure("StatusOk.TLabel", background="#26342c", foreground="#56d364", padding=(8, 4))
        style.configure("StatusWarn.TLabel", background="#3d3520", foreground="#f2cc60", padding=(8, 4))
        style.configure("TButton", padding=(10, 7))
        style.configure("Danger.TButton", foreground="#ffffff", background="#7f2d2d")
        style.map("Danger.TButton", background=[("active", "#963838")])
        style.configure("Treeview", rowheight=26, fieldbackground="#25292d", background="#25292d", foreground="#e7edf2")
        style.configure("Treeview.Heading", background="#343a40", foreground="#f4f7f9", font=("Segoe UI", 9, "bold"))

    def _build_layout(self):
        for child in self.winfo_children():
            child.destroy()
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=14, pady=12)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0)
        top.rowconfigure(1, weight=1)

        header = ttk.Frame(top)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=self._t("title"), style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value=self._t("ready"))
        self.status_lbl = ttk.Label(header, textvariable=self.status_var, style="StatusOk.TLabel")
        self.status_lbl.grid(row=0, column=1, sticky="e")
        ttk.Button(header, text=self._t("settings"), command=self._open_settings).grid(row=0, column=2, sticky="e", padx=(8, 0))

        main = ttk.Frame(top)
        main.grid(row=1, column=0, columnspan=2, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)

        image_panel = ttk.Frame(main, style="Panel.TFrame", padding=10)
        image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        image_panel.columnconfigure(0, weight=1)
        image_panel.rowconfigure(1, weight=1)
        image_head = ttk.Frame(image_panel, style="Panel.TFrame")
        image_head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        image_head.columnconfigure(5, weight=1)
        ttk.Label(image_head, text=self._t("live_camera"), style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        for i, (value, key) in enumerate((("Quality", "quality"), ("Gemini", "gemini"), ("Combined", "combined")), start=1):
            ttk.Radiobutton(
                image_head,
                text=self._t(key),
                value=value,
                variable=self.camera_view,
                command=self._on_camera_view_change,
            ).grid(row=0, column=i, sticky="w", padx=(12, 0))
        self.vision_note_var = tk.StringVar(value=self._t("vision_note"))
        ttk.Label(image_head, textvariable=self.vision_note_var, style="Panel.TLabel").grid(row=0, column=5, sticky="e")
        self.image_label = ttk.Label(image_panel, text=self._t("no_image"), anchor="center", style="Panel.TLabel")
        self.image_label.grid(row=1, column=0, sticky="nsew")
        self.image_label.bind("<Configure>", lambda _e: self._refresh_current_view())
        self.image_label.bind("<MouseWheel>", self._on_image_wheel)
        self.image_label.bind("<Button-1>", self._on_image_click)

        side_wrap = ttk.Frame(main, style="Panel.TFrame", width=310)
        side_wrap.grid(row=0, column=1, sticky="ns")
        side_wrap.grid_propagate(False)
        side_canvas = tk.Canvas(side_wrap, width=300, highlightthickness=0, bg="#2b2f33")
        side_scroll = ttk.Scrollbar(side_wrap, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_canvas.pack(side="left", fill="both", expand=True)
        side_scroll.pack(side="right", fill="y")
        side = ttk.Frame(side_canvas, style="Panel.TFrame", padding=12)
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda _e: side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>", lambda e: side_canvas.itemconfigure(side_window, width=e.width))
        side_canvas.bind("<MouseWheel>", lambda e: side_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        side.columnconfigure(0, weight=1)

        ttk.Label(side, text=self._t("summary"), style="Panel.TLabel", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        metrics = ttk.Frame(side, style="Panel.TFrame")
        metrics.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        for c in range(2):
            metrics.columnconfigure(c, weight=1)
        self.total_var = tk.StringVar(value="0 NT")
        self.valid_var = tk.StringVar(value="0")
        ttk.Label(metrics, text=self._t("total"), style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(metrics, textvariable=self.total_var, style="Metric.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(metrics, text=self._t("valid_targets"), style="Panel.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(metrics, textvariable=self.valid_var, style="Metric.TLabel").grid(row=1, column=1, sticky="w")

        self.count_vars = {name: tk.StringVar(value=f"{name} x 0") for name in ("1NT", "5NT", "10NT", "50NT")}
        for i, name in enumerate(("1NT", "5NT", "10NT", "50NT"), start=2):
            ttk.Label(side, textvariable=self.count_vars[name], style="Panel.TLabel").grid(row=i, column=0, sticky="ew", pady=2)
        self.db_count_var = tk.StringVar(value="DB 1:0  5:0  10:0  50:0")
        ttk.Label(side, textvariable=self.db_count_var, style="Panel.TLabel").grid(row=6, column=0, sticky="ew", pady=(8, 0))

        ttk.Separator(side).grid(row=7, column=0, sticky="ew", pady=12)
        ttk.Label(side, text=self._t("robot_controls"), style="Panel.TLabel", font=("Segoe UI", 12, "bold")).grid(row=8, column=0, sticky="w")
        ttk.Button(side, text=self._t("estop"), style="Danger.TButton", command=self._emergency_stop).grid(row=9, column=0, sticky="ew", pady=(10, 8), ipady=5)
        self.move_speed_label = tk.StringVar(value=f"{self._t('move_speed')} {int(self.move_speed_var.get())}%")
        self.lower_speed_label = tk.StringVar(value=f"{self._t('lower_speed')} {int(self.lower_speed_var.get())}%")
        ttk.Button(side, text=self._t("clear_enable"), command=self._clear_enable_robot).grid(row=11, column=0, sticky="ew", pady=(10, 4))
        ttk.Button(side, text=self._t("move_clear"), command=self._move_start_pose).grid(row=12, column=0, sticky="ew", pady=(10, 4))
        ttk.Button(side, text=self._t("hover_first"), command=self._hover_first_cycle).grid(row=13, column=0, sticky="ew", pady=(12, 4))
        ttk.Button(side, text=self._t("lower_first"), style="Danger.TButton", command=self._safe_cycle).grid(row=14, column=0, sticky="ew", pady=4)
        ttk.Button(side, text=self._t("unstack_coins"), style="Danger.TButton", command=self._unstack_coins).grid(row=15, column=0, sticky="ew", pady=4)
        ttk.Label(side, text="夾取偏移校正", style="Panel.TLabel", font=("Segoe UI", 10, "bold")).grid(row=16, column=0, sticky="w", pady=(12, 2))
        self.pick_offset_label = tk.StringVar()
        ttk.Label(side, textvariable=self.pick_offset_label, style="Panel.TLabel").grid(row=17, column=0, sticky="ew")
        offset_buttons = ttk.Frame(side, style="Panel.TFrame")
        offset_buttons.grid(row=18, column=0, sticky="ew", pady=(4, 8))
        for c in range(4):
            offset_buttons.columnconfigure(c, weight=1)
        ttk.Button(offset_buttons, text="X -1", command=lambda: self._nudge_pick_offset(-1.0, 0.0)).grid(row=0, column=0, sticky="ew", padx=1)
        ttk.Button(offset_buttons, text="X +1", command=lambda: self._nudge_pick_offset(1.0, 0.0)).grid(row=0, column=1, sticky="ew", padx=1)
        ttk.Button(offset_buttons, text="Y -1", command=lambda: self._nudge_pick_offset(0.0, -1.0)).grid(row=0, column=2, sticky="ew", padx=1)
        ttk.Button(offset_buttons, text="Y +1", command=lambda: self._nudge_pick_offset(0.0, 1.0)).grid(row=0, column=3, sticky="ew", padx=1)
        self._update_pick_offset_label()
        info_start_row = 25
        if self.ui_mode == "offset_calib":
            ttk.Separator(side).grid(row=19, column=0, sticky="ew", pady=10)
            ttk.Label(side, text=self._t("offset_calib_title"), style="Panel.TLabel", font=("Segoe UI", 12, "bold")).grid(row=20, column=0, sticky="w")
            ttk.Label(side, text=self._t("offset_calib_hint"), style="Panel.TLabel", wraplength=280, justify="left").grid(row=21, column=0, sticky="ew", pady=(4, 8))
            ttk.Button(side, text=self._t("refresh"), command=self._refresh_vision).grid(row=22, column=0, sticky="ew", pady=4)
            ttk.Button(side, text=self._t("start_offset_calib"), command=self._offset_calib_start).grid(row=23, column=0, sticky="ew", pady=4)
            step_row = ttk.Frame(side, style="Panel.TFrame")
            step_row.grid(row=24, column=0, sticky="ew", pady=(6, 2))
            step_row.columnconfigure(0, weight=1)
            step_row.columnconfigure(1, weight=1)
            step_row.columnconfigure(2, weight=1)
            self.offset_jog_step_label = tk.StringVar(value=f"步距 {self.offset_jog_step_var.get():.1f}mm")
            ttk.Label(step_row, textvariable=self.offset_jog_step_label, style="Panel.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Button(step_row, text="1", command=lambda: self._set_offset_jog_step(1.0)).grid(row=0, column=1, sticky="ew", padx=2)
            step_buttons = ttk.Frame(step_row, style="Panel.TFrame")
            step_buttons.grid(row=0, column=2, sticky="ew")
            step_buttons.columnconfigure(0, weight=1)
            step_buttons.columnconfigure(1, weight=1)
            ttk.Button(step_buttons, text="5", command=lambda: self._set_offset_jog_step(5.0)).grid(row=0, column=0, sticky="ew", padx=1)
            ttk.Button(step_buttons, text="10", command=lambda: self._set_offset_jog_step(10.0)).grid(row=0, column=1, sticky="ew", padx=1)
            jog = ttk.Frame(side, style="Panel.TFrame")
            jog.grid(row=25, column=0, sticky="ew", pady=(8, 4))
            for c in range(3):
                jog.columnconfigure(c, weight=1)
            self._make_offset_jog_button(jog, "Y+", 0, 1, 0).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
            self._make_offset_jog_button(jog, "X-", -1, 0, 0).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
            ttk.Label(jog, textvariable=self.offset_jog_step_label, style="Panel.TLabel").grid(row=1, column=1)
            self._make_offset_jog_button(jog, "X+", 1, 0, 0).grid(row=1, column=2, sticky="ew", padx=2, pady=2)
            self._make_offset_jog_button(jog, "Y-", 0, -1, 0).grid(row=2, column=1, sticky="ew", padx=2, pady=2)
            zrow = ttk.Frame(side, style="Panel.TFrame")
            zrow.grid(row=26, column=0, sticky="ew")
            zrow.columnconfigure(0, weight=1)
            zrow.columnconfigure(1, weight=1)
            self._make_offset_jog_button(zrow, "Z+", 0, 0, 1).grid(row=0, column=0, sticky="ew", padx=2)
            self._make_offset_jog_button(zrow, "Z-", 0, 0, -1).grid(row=0, column=1, sticky="ew", padx=2)
            teach_row = ttk.Frame(side, style="Panel.TFrame")
            teach_row.grid(row=27, column=0, sticky="ew", pady=(8, 4))
            teach_row.columnconfigure(0, weight=1)
            teach_row.columnconfigure(1, weight=1)
            ttk.Button(teach_row, text=self._t("teach_offset_add"), command=lambda: self._offset_calib_teach(False)).grid(row=0, column=0, sticky="ew", padx=2)
            ttk.Button(teach_row, text=self._t("teach_offset_replace"), command=lambda: self._offset_calib_teach(True)).grid(row=0, column=1, sticky="ew", padx=2)
            ttk.Button(side, text=self._t("reset_offset_samples"), style="Danger.TButton", command=self._offset_calib_reset).grid(row=28, column=0, sticky="ew", pady=4)
            info_start_row = 29
        if self.ui_mode == "engineer":
            ttk.Separator(side).grid(row=19, column=0, sticky="ew", pady=10)
            ttk.Button(side, text=self._t("set_roi"), command=self._select_roi).grid(row=20, column=0, sticky="ew", pady=4)
            ttk.Button(side, text=self._t("refresh"), command=self._refresh_vision).grid(row=21, column=0, sticky="ew", pady=4)
            ttk.Button(side, text=self._t("hover_selected"), command=self._hover_selected).grid(row=22, column=0, sticky="ew", pady=4)
            ttk.Button(side, text=self._t("lower_selected"), style="Danger.TButton", command=self._dry_lower_selected).grid(row=23, column=0, sticky="ew", pady=4)
            ttk.Button(side, text=self._t("lower_all"), style="Danger.TButton", command=self._dry_lower_all).grid(row=24, column=0, sticky="ew", pady=4)

        ttk.Separator(side).grid(row=info_start_row, column=0, sticky="ew", pady=12)
        ttk.Label(side, text=self._t("current_target"), style="Panel.TLabel", font=("Segoe UI", 12, "bold")).grid(row=info_start_row + 1, column=0, sticky="w")
        self.selected_var = tk.StringVar(value=self._t("no_target_selected"))
        ttk.Label(side, textvariable=self.selected_var, style="Panel.TLabel", wraplength=280, justify="left").grid(row=info_start_row + 2, column=0, sticky="ew", pady=(8, 0))
        self.action_var = tk.StringVar(value=self._t("current_action_idle"))
        ttk.Label(side, textvariable=self.action_var, style="Panel.TLabel", wraplength=280, justify="left").grid(row=info_start_row + 3, column=0, sticky="ew", pady=(10, 0))
        self.note_var = tk.StringVar(value=self._t("dry_only"))
        ttk.Label(side, textvariable=self.note_var, style="Panel.TLabel", wraplength=280, justify="left").grid(row=info_start_row + 4, column=0, sticky="ew", pady=(12, 0))

        table_panel = ttk.Frame(main, style="Panel.TFrame", padding=10)
        table_panel.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        table_panel.columnconfigure(0, weight=1)
        ttk.Label(table_panel, text=self._t("coin_targets"), style="Panel.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.pick_plan_var = tk.StringVar(value="拆堆下一顆：尚無資料")
        ttk.Label(table_panel, textvariable=self.pick_plan_var, style="Panel.TLabel", wraplength=1050, justify="left").grid(row=0, column=0, sticky="e", pady=(0, 6))
        cols = ("rank", "index", "label", "diam", "depth", "height", "x", "y", "z", "valid", "why")
        self.tree = ttk.Treeview(table_panel, columns=cols, show="headings", height=7)
        headings = {
            "rank": "拆堆",
            "index": self._t("index"),
            "label": self._t("class"),
            "diam": self._t("diameter"),
            "depth": "深度",
            "height": "高出桌面",
            "x": "MG400 X",
            "y": "MG400 Y",
            "z": "MG400 Z",
            "valid": self._t("status"),
            "why": "判斷",
        }
        widths = {
            "rank": 58,
            "index": 52,
            "label": 70,
            "diam": 76,
            "depth": 72,
            "height": 82,
            "x": 86,
            "y": 86,
            "z": 78,
            "valid": 70,
            "why": 250,
        }
        for col in cols:
            self.tree.heading(col, text=headings[col])
            anchor = "w" if col == "why" else "center"
            self.tree.column(col, width=widths[col], minwidth=widths[col], anchor=anchor, stretch=True)
        self.tree.tag_configure("next_pick", background="#314533", foreground="#f4fff4")
        yscroll = ttk.Scrollbar(table_panel, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.grid(row=1, column=0, sticky="ew")
        yscroll.grid(row=1, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        self.log = tk.Text(top, height=4, wrap="word", bg="#181b1e", fg="#c9d1d9", insertbackground="#c9d1d9", relief="flat")
        self.log.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self._log(self._t("vision_note"))

    def _tick(self):
        if self._yolo_live_enabled:
            self._update_db_count_display()
        else:
            self._load_targets()
        self._load_action_status()
        if self.camera_view.get() not in ("Quality", "Combined"):
            self._load_latest_image()
        self.after(1000, self._tick)

    def _auto_preview_loop(self):
        if self.auto_preview.get() and not self.busy:
            if self.camera_view.get() == "Quality":
                self._start_quality_preview_thread()
                self._update_quality_live_frame()
                self.after(30, self._auto_preview_loop)
                return
            if self.camera_view.get() == "Combined":
                self._start_quality_preview_thread()
                now = time.time()
                if now - self._last_gemini_preview_request > 1.2:
                    self._last_gemini_preview_request = now
                    self._preview_only("Gemini")
                self._update_combined_live_frame()
                self.after(35, self._auto_preview_loop)
                return
            self._close_quality_preview()
            self._preview_only()
        elif self.camera_view.get() != "Quality":
            self._close_quality_preview()
        self.after(900, self._auto_preview_loop)

    def _set_busy(self, busy, text=None):
        self.busy = busy
        self.status_var.set(text or (self._t("busy") if busy else self._t("ready")))
        self.status_lbl.configure(style="StatusWarn.TLabel" if busy else "StatusOk.TLabel")
        if busy and text and ("Refresh" in text or "detect" in text.lower() or "vision" in text.lower()):
            self.vision_note_var.set(self._t("recognizing"))
        elif not busy:
            self.vision_note_var.set(self._t("vision_note"))

    def _on_speed_change(self, _value=None):
        self.move_speed_label.set(f"{self._t('move_speed')} {int(self.move_speed_var.get())}%")
        self.lower_speed_label.set(f"{self._t('lower_speed')} {int(self.lower_speed_var.get())}%")

    def _speed_args(self):
        return [
            "--move-speed", str(int(self.move_speed_var.get())),
            "--lower-speed", str(int(self.lower_speed_var.get())),
            "--move-speed-j", str(int(self.ui_config.get("robot_move_speed_j_pct", 80))),
            "--move-acc-j", str(int(self.ui_config.get("robot_move_acc_j_pct", 70))),
            "--move-speed-l", str(int(self.ui_config.get("robot_move_speed_l_pct", 70))),
            "--move-acc-l", str(int(self.ui_config.get("robot_move_acc_l_pct", 60))),
            "--lower-speed-l", str(int(self.ui_config.get("robot_lower_speed_l_pct", 35))),
            "--lower-acc-l", str(int(self.ui_config.get("robot_lower_acc_l_pct", 25))),
        ]

    def _start_pose_args(self):
        x, y, z = self._robot_return_pose()
        return ["--start-pose", f"{x:.3f},{y:.3f},{z:.3f}"]

    def _log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _schedule_quality_open_error(self):
        if self._quality_preview_error_after is not None or self._quality_preview_error_logged:
            return

        def report_if_still_blank():
            self._quality_preview_error_after = None
            with self._quality_lock:
                has_frame = self._quality_latest_frame is not None
            if has_frame or self.camera_view.get() not in ("Quality", "Combined"):
                return
            self._log("畫質相機暫時未回應，正在等待相機釋放。")
            self._quality_preview_error_logged = True

        self._quality_preview_error_after = self.after(1800, report_if_still_blank)

    def _clear_quality_open_error(self):
        self._quality_preview_error_logged = False
        if self._quality_preview_error_after is not None:
            try:
                self.after_cancel(self._quality_preview_error_after)
            except Exception:
                pass
            self._quality_preview_error_after = None

    def _normalize_roi(self, roi, shape):
        h, w = shape[:2]
        if not roi:
            return 0, 0, w, h
        try:
            x1, y1, x2, y2 = [int(v) for v in roi]
        except Exception:
            return 0, 0, w, h
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return 0, 0, w, h
        return x1, y1, x2, y2

    def _coin_label_from_yolo(self, raw_name):
        text = str(raw_name).strip().lower()
        mapping = {
            "1nt": ("1yuan", "1NT", 1),
            "1yuan": ("1yuan", "1NT", 1),
            "1": ("1yuan", "1NT", 1),
            "5nt": ("5yuan", "5NT", 5),
            "5yuan": ("5yuan", "5NT", 5),
            "5": ("5yuan", "5NT", 5),
            "10nt": ("10yuan", "10NT", 10),
            "10yuan": ("10yuan", "10NT", 10),
            "10": ("10yuan", "10NT", 10),
            "50nt": ("50yuan", "50NT", 50),
            "50yuan": ("50yuan", "50NT", 50),
            "50": ("50yuan", "50NT", 50),
        }
        return mapping.get(text, ("?", "?", 0))

    def _resolve_yolo_model_path(self, cfg):
        candidates = []
        if cfg.get("quality_yolo_model_path"):
            candidates.append(Path(str(cfg["quality_yolo_model_path"])))
        candidates.extend([
            HERE.parent / "runs_yolo_coin" / "yolov8m_960_finetune_120" / "weights" / "best.pt",
            HERE.parent / "runs_yolo_coin" / "yolov8m_960_final" / "weights" / "best.pt",
        ])
        for path in candidates:
            if path.exists():
                return path
        return candidates[0] if candidates else None

    def _get_yolo_live_model(self, cfg):
        path = self._resolve_yolo_model_path(cfg)
        if path is None or not path.exists():
            raise FileNotFoundError("找不到 YOLO 權重")
        if self._yolo_live_model is None or self._yolo_live_model_path != path:
            from ultralytics import YOLO

            self._log(f"載入 YOLO 即時模型：{path}")
            self._yolo_live_model = YOLO(str(path))
            self._yolo_live_model_path = path
        return self._yolo_live_model

    def _yolo_live_targets_from_frame(self, frame, cfg):
        x1, y1, x2, y2 = self._normalize_roi(cfg.get("quality_roi"), frame.shape)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        model = self._get_yolo_live_model(cfg)
        results = model.predict(
            source=crop,
            imgsz=int(cfg.get("quality_yolo_imgsz", 960)),
            conf=float(cfg.get("quality_yolo_confidence", 0.25)),
            iou=float(cfg.get("quality_yolo_iou", 0.7)),
            max_det=int(cfg.get("quality_max_detections", 60)),
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []
        names = getattr(results[0], "names", {}) or {}
        targets = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
            bx1, by1, bx2, by2 = xyxy.tolist()
            bw = max(1.0, bx2 - bx1)
            bh = max(1.0, by2 - by1)
            cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
            conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            label, label_name, value = self._coin_label_from_yolo(names.get(cls_id, str(cls_id)))
            targets.append({
                "index": len(targets) + 1,
                "label": label,
                "label_name": label_name,
                "value_nt": value,
                "bbox_xyxy": [
                    round(float(x1 + bx1), 3),
                    round(float(y1 + by1), 3),
                    round(float(x1 + bx2), 3),
                    round(float(y1 + by2), 3),
                ],
                "quality_x_px": round(float(x1 + (bx1 + bx2) / 2.0), 3),
                "quality_y_px": round(float(y1 + (by1 + by2) / 2.0), 3),
                "diameter_mm": None,
                "robot_x_mm": None,
                "robot_y_mm": None,
                "robot_z_mm": None,
                "axis_ratio": round(float(min(bw, bh) / max(bw, bh)), 4),
                "visible_score": round(float(min(bw, bh) / max(bw, bh)), 4),
                "top_coin_score": round(conf, 4),
                "valid_for_pick": False,
                "pick_check_reason": "live_preview_only",
                "live_preview": True,
                "class_confidence": round(conf, 4),
                "class_source": "yolo_live",
            })
        targets.sort(key=lambda t: (float(t["quality_y_px"]), float(t["quality_x_px"])))
        for idx, target in enumerate(targets, 1):
            target["index"] = idx
        return targets

    def _stabilize_yolo_live_targets(self, targets, cfg):
        if not bool(cfg.get("quality_yolo_live_stabilize_enabled", True)):
            self._yolo_live_tracks = []
            self._yolo_live_next_track_index = 1
            return targets
        now = time.time()
        max_age = max(0.2, float(cfg.get("quality_yolo_live_track_max_age_sec", 1.8)))
        ghost_age = max(0.0, float(cfg.get("quality_yolo_live_track_ghost_sec", 0.65)))
        max_dist = max(8.0, float(cfg.get("quality_yolo_live_track_match_px", 160.0)))
        alpha = min(1.0, max(0.05, float(cfg.get("quality_yolo_live_smooth_alpha", 0.45))))
        self._yolo_live_tracks = [
            tr for tr in self._yolo_live_tracks
            if now - float(tr.get("last_seen", 0.0)) <= max_age
        ]
        if not targets:
            ghosts = []
            for track in self._yolo_live_tracks:
                if now - float(track.get("last_seen", 0.0)) > ghost_age:
                    continue
                item = dict(track.get("last_target") or {})
                if not item:
                    continue
                item["index"] = int(track["index"])
                item["quality_x_px"] = round(float(track.get("x", item.get("quality_x_px", 0.0))), 3)
                item["quality_y_px"] = round(float(track.get("y", item.get("quality_y_px", 0.0))), 3)
                item["pick_check_reason"] = "live_preview_hold"
                ghosts.append(item)
            return sorted(ghosts, key=lambda t: int(t.get("index", 9999)))
        used_tracks = set()
        stabilized = []
        for target in sorted(targets, key=lambda t: float(t.get("class_confidence", 0.0) or 0.0), reverse=True):
            try:
                tx = float(target["quality_x_px"])
                ty = float(target["quality_y_px"])
            except Exception:
                stabilized.append(target)
                continue
            best_i = None
            best_score = None
            for i, track in enumerate(self._yolo_live_tracks):
                if i in used_tracks:
                    continue
                dx = tx - float(track.get("x", tx))
                dy = ty - float(track.get("y", ty))
                dist = (dx * dx + dy * dy) ** 0.5
                track_radius = float(track.get("radius", 0.0) or 0.0)
                dynamic_max_dist = max(max_dist, track_radius * 1.4)
                if dist > dynamic_max_dist:
                    continue
                label_penalty = 0.0 if target.get("label_name") == track.get("label_name") else dynamic_max_dist * 0.45
                score = dist + label_penalty
                if best_score is None or score < best_score:
                    best_score = score
                    best_i = i
            item = dict(target)
            radius = 0.0
            try:
                bx1, by1, bx2, by2 = [float(v) for v in target.get("bbox_xyxy") or []]
                radius = max(0.0, 0.25 * ((bx2 - bx1) + (by2 - by1)))
            except Exception:
                radius = 0.0
            if best_i is None:
                track = {
                    "index": self._yolo_live_next_track_index,
                    "x": tx,
                    "y": ty,
                    "radius": radius,
                    "label_name": target.get("label_name"),
                    "last_seen": now,
                    "last_target": item,
                }
                self._yolo_live_next_track_index += 1
                self._yolo_live_tracks.append(track)
                best_i = len(self._yolo_live_tracks) - 1
            else:
                track = self._yolo_live_tracks[best_i]
                track["x"] = (1.0 - alpha) * float(track.get("x", tx)) + alpha * tx
                track["y"] = (1.0 - alpha) * float(track.get("y", ty)) + alpha * ty
                if radius > 0.0:
                    track["radius"] = (1.0 - alpha) * float(track.get("radius", radius)) + alpha * radius
                track["label_name"] = target.get("label_name")
                track["last_seen"] = now
                track["last_target"] = item
            used_tracks.add(best_i)
            item["index"] = int(track["index"])
            item["quality_x_px"] = round(float(track["x"]), 3)
            item["quality_y_px"] = round(float(track["y"]), 3)
            item["pick_check_reason"] = "live_preview_stabilized"
            stabilized.append(item)
        stabilized.sort(key=lambda t: int(t.get("index", 9999)))
        return stabilized

    def _apply_live_targets(self, targets):
        if not self._yolo_live_enabled:
            return
        try:
            cfg = self._load_config()
        except Exception:
            cfg = {}
        targets = self._stabilize_yolo_live_targets(targets, cfg)
        counts = {name: 0 for name in ("1NT", "5NT", "10NT", "50NT")}
        total = 0
        for target in targets:
            name = target.get("label_name", "?")
            if name in counts:
                counts[name] += 1
            total += int(target.get("value_nt", 0) or 0)
        self.targets_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "yolo_live_preview",
            "counts": counts,
            "total_value_nt": total,
            "targets": targets,
        }
        self._render_targets_display()
        self._quality_displayed_id = -1
        now = time.time()
        if now - self._yolo_live_last_draw_at > 0.12:
            self._yolo_live_last_draw_at = now
            if self.camera_view.get() == "Quality":
                self.after(1, self._update_quality_live_frame)
            elif self.camera_view.get() == "Combined":
                self.after(1, self._update_combined_live_frame)

    def _is_yolo_mode(self):
        return str(self._load_config().get("quality_detection_method", "")).lower() in ("yolo", "yolov8", "yolov8m")

    def _start_yolo_live_detection(self):
        if self._yolo_live_enabled:
            return
        self._yolo_live_enabled = True
        self._yolo_live_stop.clear()
        self._yolo_live_last_frame_id = 0
        self._yolo_live_last_draw_at = 0.0
        self._yolo_live_tracks = []
        self._yolo_live_next_track_index = 1
        if self.camera_view.get() != "Quality":
            self.camera_view.set("Quality")
        self.auto_preview.set(True)
        self._start_quality_preview_thread()
        self._update_quality_live_frame()
        self.status_var.set("YOLO 即時辨識中")
        self._log("YOLO 即時辨識已啟動；手臂動作時仍會正式擷取並鎖定座標。")

        def worker():
            last_infer_at = 0.0
            while not self._yolo_live_stop.is_set():
                now = time.time()
                try:
                    cfg = self._load_config()
                    interval_sec = max(0.08, float(cfg.get("quality_yolo_live_interval_sec", 0.25)))
                except Exception:
                    cfg = {}
                    interval_sec = 0.25
                if now - last_infer_at < interval_sec:
                    time.sleep(0.03)
                    continue
                frame, frame_id = self._latest_quality_frame_copy()
                if frame is None or frame_id == self._yolo_live_last_frame_id:
                    time.sleep(0.05)
                    continue
                self._yolo_live_last_frame_id = frame_id
                last_infer_at = now
                try:
                    targets = self._yolo_live_targets_from_frame(frame, cfg)
                except Exception as exc:
                    self.after(0, lambda err=exc: self._log(f"YOLO 即時辨識失敗：{err}"))
                    time.sleep(1.0)
                    continue
                self.after(0, lambda items=targets: self._apply_live_targets(items))
                time.sleep(0.02)

        self._yolo_live_thread = threading.Thread(target=worker, daemon=True)
        self._yolo_live_thread.start()

    def _stop_yolo_live_detection(self, clear_status=False):
        if not self._yolo_live_enabled:
            return
        self._yolo_live_enabled = False
        self._yolo_live_stop.set()
        if clear_status:
            self.status_var.set(self._t("ready"))
        self._log("YOLO 即時辨識已停止。")

    def _toggle_yolo_live_detection(self):
        if self._yolo_live_enabled:
            self._stop_yolo_live_detection(clear_status=True)
        else:
            self._start_yolo_live_detection()

    def _load_targets(self):
        self._update_db_count_display()
        if not TARGETS_FILE.exists():
            self._clear_targets_display()
            return
        try:
            if TARGETS_FILE.stat().st_mtime < self._session_started_at:
                self._clear_targets_display()
                return
        except Exception:
            return
        try:
            self.targets_data = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        self._render_targets_display()

    def _render_targets_display(self):
        counts = self.targets_data.get("counts", {})
        for name in ("1NT", "5NT", "10NT", "50NT"):
            self.count_vars[name].set(f"{name} x {counts.get(name, 0)}")
        self.total_var.set(f"{self.targets_data.get('total_value_nt', 0)} NT")
        targets = self.targets_data.get("targets", [])
        valid_count = sum(1 for t in targets if t.get("valid_for_pick"))
        self.valid_var.set(str(valid_count))
        pick_plan = self._get_display_pick_plan(targets)
        pick_by_index = {int(item["target"].get("index", -1)): item for item in pick_plan}
        self._update_pick_plan_text(pick_plan, targets)
        selected = self._current_selected_index()
        self.tree.delete(*self.tree.get_children())
        def table_sort_key(target):
            try:
                idx = int(target.get("index", 9999))
            except Exception:
                idx = 9999
            pick_item = pick_by_index.get(idx)
            if pick_item:
                return (0, int(pick_item.get("rank", 9999)), idx)
            return (1, idx, idx)

        for t in sorted(targets, key=table_sort_key):
            idx = int(t.get("index", 0))
            iid = str(idx)
            pick_item = pick_by_index.get(idx)
            pick_rank = ""
            pick_why = ""
            tags = ()
            if pick_item:
                pick_rank = "NEXT" if pick_item["rank"] == 1 else str(pick_item["rank"])
                pick_why = pick_item["short_reason"]
                if pick_item["rank"] == 1:
                    tags = ("next_pick",)
            values = (
                pick_rank,
                idx,
                t.get("label_name", "?"),
                self._fmt(t.get("diameter_mm"), "mm"),
                self._fmt(t.get("depth_z_mm"), "mm"),
                self._fmt(t.get("height_above_table_mm"), "mm"),
                self._fmt(t.get("robot_x_mm")),
                self._fmt(t.get("robot_y_mm")),
                self._fmt(t.get("robot_z_mm")),
                "LIVE" if t.get("live_preview") else (self._t("ok") if t.get("valid_for_pick") else self._t("check")),
                pick_why,
            )
            self.tree.insert("", "end", iid=iid, values=values, tags=tags)
        if selected and self.tree.exists(str(selected)):
            self.tree.selection_set(str(selected))
            self.tree.focus(str(selected))
        self._update_selected_text()

    def _remap_pick_plan_targets(self, plan, targets):
        by_index = {}
        for target in targets or []:
            try:
                by_index[int(target.get("index", -1))] = target
            except Exception:
                continue
        remapped = []
        for item in plan or []:
            try:
                idx = int(item["target"].get("index", -1))
            except Exception:
                continue
            current = by_index.get(idx)
            if current is None:
                continue
            new_item = dict(item)
            new_item["target"] = current
            remapped.append(new_item)
        return remapped

    def _get_display_pick_plan(self, targets):
        cfg = self.ui_config or self._load_config()
        current = self._compute_unstack_pick_plan(targets)
        is_live = any(t.get("live_preview") for t in (targets or []))
        if not is_live:
            self._display_pick_plan = current
            self._display_pick_plan_until = 0.0
            self._display_pick_plan_source = "locked"
            return current
        now = time.time()
        hold_sec = max(0.2, float(cfg.get("quality_yolo_live_rank_hold_sec", 2.0)))
        cached = self._remap_pick_plan_targets(self._display_pick_plan, targets)
        current_next = None if not current else int(current[0]["target"].get("index", -1))
        cached_next = None if not cached else int(cached[0]["target"].get("index", -1))
        if cached and cached_next is not None and now < self._display_pick_plan_until:
            return cached
        if cached and cached_next is not None and current_next is not None and current_next != cached_next:
            self._display_pick_plan_until = now + hold_sec
            return cached
        self._display_pick_plan = current
        self._display_pick_plan_until = now + hold_sec
        self._display_pick_plan_source = "live"
        return current

    def _compute_unstack_pick_plan(self, targets):
        if not targets or any(fn is None for fn in (
            is_pick_range_target,
            is_safe_xy,
            is_stacked_top_target,
            _top_visible_sort_key,
            _stack_height_mm,
            _target_depth,
            _bbox_overlap_risk,
        )):
            return []
        cfg = self.ui_config or self._load_config()
        all_targets = list(targets)
        safe_targets = []
        for target in all_targets:
            try:
                if target.get("robot_x_mm") is None or target.get("robot_y_mm") is None:
                    continue
                if target.get("diameter_mm") is None:
                    continue
                if not is_pick_range_target(target, cfg):
                    continue
                if not is_safe_xy(target["robot_x_mm"], target["robot_y_mm"], cfg):
                    continue
                if cfg.get("unstack_require_classified_pick", True):
                    if target.get("label") in (None, "?") or target.get("label_name") in (None, "?"):
                        continue
                safe_targets.append(target)
            except Exception:
                continue
        checked = []
        for target in safe_targets:
            try:
                ok, reason = is_stacked_top_target(target, all_targets, cfg)
            except Exception as exc:
                ok, reason = False, f"check_error:{exc}"
            checked.append((target, ok, reason))
        candidates = [(target, reason, "top") for target, ok, reason in checked if ok and _target_depth(target) is not None]
        if candidates:
            candidates.sort(key=lambda item: _top_visible_sort_key(item[0], all_targets, cfg))
        elif _unstack_cfg_bool(cfg, "unstack_fallback_pick_any_round", True) if _unstack_cfg_bool else bool(cfg.get("unstack_fallback_pick_any_round", True)):
            fallback = []
            for target, _ok, _reason in checked:
                if _is_round_fallback_target_visible is None or _round_fallback_sort_key is None:
                    continue
                try:
                    fb_ok, fb_reason = _is_round_fallback_target_visible(target, all_targets, cfg)
                except Exception as exc:
                    fb_ok, fb_reason = False, f"fallback_error:{exc}"
                if fb_ok:
                    fallback.append((target, fb_reason, "fallback"))
            fallback.sort(key=lambda item: _round_fallback_sort_key(item[0]))
            candidates = fallback
        if not candidates and any(t.get("live_preview") for t in all_targets):
            live_candidates = []
            live_points = []
            for item in all_targets:
                try:
                    live_points.append((float(item.get("quality_x_px")), float(item.get("quality_y_px"))))
                except Exception:
                    continue
            if live_points:
                cluster_cx = sum(p[0] for p in live_points) / len(live_points)
                cluster_cy = sum(p[1] for p in live_points) / len(live_points)
            else:
                cluster_cx = cluster_cy = 0.0
            neighbor_radius = float(cfg.get("quality_yolo_live_cluster_neighbor_px", 115.0))
            cluster_penalty = float(cfg.get("quality_yolo_live_cluster_distance_penalty", 0.35))
            neighbor_weight = float(cfg.get("quality_yolo_live_neighbor_weight", 0.22))
            center_cover_penalty = float(cfg.get("quality_yolo_live_center_cover_penalty", 2.0))
            center_cover_pad = float(cfg.get("quality_yolo_live_center_cover_pad_px", 6.0))
            for target in all_targets:
                try:
                    tx = float(target.get("quality_x_px"))
                    ty = float(target.get("quality_y_px"))
                    visible = float(target.get("visible_score", target.get("top_visible_score")) or 0.0)
                    axis = float(target.get("axis_ratio") or 0.0)
                    score = float(target.get("class_confidence") or target.get("top_coin_score") or 0.0)
                    overlap = _bbox_overlap_risk(target, all_targets)
                    center_covered = 0
                    try:
                        target_index = int(target.get("index", -1))
                    except Exception:
                        target_index = -1
                    for other in all_targets:
                        try:
                            if int(other.get("index", -2)) == target_index:
                                continue
                            ox1, oy1, ox2, oy2 = [float(v) for v in other.get("bbox_xyxy") or []]
                        except Exception:
                            continue
                        if (ox1 - center_cover_pad) <= tx <= (ox2 + center_cover_pad) and (oy1 - center_cover_pad) <= ty <= (oy2 + center_cover_pad):
                            center_covered += 1
                    neighbor_count = 0
                    for px, py in live_points:
                        dist = ((tx - px) ** 2 + (ty - py) ** 2) ** 0.5
                        if 1.0 < dist <= neighbor_radius:
                            neighbor_count += 1
                    neighbor_score = min(1.0, neighbor_count / 3.0)
                    cluster_dist = ((tx - cluster_cx) ** 2 + (ty - cluster_cy) ** 2) ** 0.5
                    cluster_score = min(1.0, cluster_dist / max(neighbor_radius * 2.0, 1.0))
                    overlap_limit = float(cfg.get("quality_yolo_live_prefer_clean_overlap_max", 0.16))
                    overlap_penalty = float(cfg.get("quality_yolo_live_overlap_penalty", 1.25))
                    visible_weight = float(cfg.get("quality_yolo_live_visible_weight", 0.45))
                    axis_weight = float(cfg.get("quality_yolo_live_axis_weight", 0.35))
                    conf_weight = float(cfg.get("quality_yolo_live_conf_weight", 0.20))
                    rank_score = (
                        visible_weight * visible
                        + axis_weight * axis
                        + conf_weight * score
                        + neighbor_weight * neighbor_score
                        - overlap_penalty * overlap
                        - cluster_penalty * cluster_score
                        - center_cover_penalty * min(1.0, center_covered)
                    )
                    overlap_bucket = 0 if overlap <= overlap_limit else 1
                    cover_bucket = 0 if center_covered == 0 else 1
                    live_candidates.append((target, "live_preview_rank", "live", cover_bucket, overlap_bucket, overlap, rank_score, visible, axis, score, neighbor_count, cluster_dist))
                except Exception:
                    continue
            live_candidates.sort(key=lambda item: (item[3], item[4], -item[6], item[5], -item[7], -item[8], -item[9], -item[10], item[11], int(item[0].get("index", 9999))))
            candidates = [(target, reason, mode) for target, reason, mode, _cover_bucket, _overlap_bucket, _overlap, _rank_score, _visible, _axis, _score, _neighbors, _cluster_dist in live_candidates]
        plan = []
        radius = float(cfg.get("unstack_stack_neighbor_distance_mm", 60.0))
        for rank, (target, reason, mode) in enumerate(candidates, 1):
            visible = float(target.get("visible_score", target.get("top_visible_score")) or 0.0)
            axis = float(target.get("axis_ratio") or 0.0)
            score = float(target.get("top_coin_score") or 0.0)
            height = _stack_height_mm(target, all_targets, radius)
            overlap = _bbox_overlap_risk(target, all_targets)
            depth = _target_depth(target)
            short_reason = (
                f"{mode} 可見{visible:.2f} 重疊{overlap:.2f} "
                f"圓度{axis:.2f} 上表高{height:.1f} 分{score:.2f}"
            )
            plan.append({
                "rank": rank,
                "target": target,
                "reason": reason,
                "short_reason": short_reason,
                "visible": visible,
                "overlap": overlap,
                "axis": axis,
                "height": height,
                "score": score,
                "depth": depth,
                "mode": mode,
            })
        return plan

    def _update_pick_plan_text(self, plan, targets):
        if not hasattr(self, "pick_plan_var"):
            return
        if self._active_robot_target and self._active_robot_state not in ("idle", "done"):
            t = self._active_robot_target
            prefix = "實際拆堆目標" if self._active_robot_state != "failed" else "上次失敗目標"
            self.pick_plan_var.set(
                f"{prefix}：Q{t.get('index')} {t.get('label_name', '?')} | "
                f"X={self._fmt(t.get('robot_x_mm'))} Y={self._fmt(t.get('robot_y_mm'))} | "
                "此列為手臂實際執行目標，不是預覽排序"
            )
            return
        cfg = self.ui_config or {}
        priority = str(cfg.get("unstack_pick_priority", "top_visible_highest"))
        if plan:
            item = plan[0]
            target = item["target"]
            self.pick_plan_var.set(
                f"拆堆下一顆：Q{target.get('index')} {target.get('label_name', '?')} | "
                f"原因：可見度 {item['visible']:.2f}、重疊 {item['overlap']:.2f}、"
                f"圓度 {item['axis']:.2f}、上表高 {item['height']:.1f}mm、分數 {item['score']:.2f} | "
                f"模式：{priority}"
            )
            return
        if not targets:
            self.pick_plan_var.set("拆堆下一顆：尚無辨識資料")
            return
        self.pick_plan_var.set(
            "拆堆下一顆：目前沒有符合條件的候選。條件包含：在夾取/安全範圍內、有分類、有完整直徑、有深度、圓度達標。"
        )

    def _is_active_robot_target(self, target):
        active = self._active_robot_target
        if not active or self._active_robot_state in ("idle", "done"):
            return False
        try:
            ax = float(active.get("robot_x_mm"))
            ay = float(active.get("robot_y_mm"))
            tx = float(target.get("robot_x_mm"))
            ty = float(target.get("robot_y_mm"))
            tol = max(1.0, float((self.ui_config or {}).get("active_target_match_tolerance_mm", 18.0)))
            return ((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5 <= tol
        except Exception:
            pass
        try:
            return int(active.get("index")) == int(target.get("index"))
        except Exception:
            return False

    def _clear_targets_display(self):
        self.targets_data = {}
        self._update_db_count_display()
        for name in ("1NT", "5NT", "10NT", "50NT"):
            if hasattr(self, "count_vars"):
                self.count_vars[name].set(f"{name} x 0")
        if hasattr(self, "total_var"):
            self.total_var.set("0 NT")
        if hasattr(self, "valid_var"):
            self.valid_var.set("0")
        if hasattr(self, "tree"):
            self.tree.delete(*self.tree.get_children())
        if hasattr(self, "pick_plan_var"):
            self.pick_plan_var.set("拆堆下一顆：尚無資料")
        if hasattr(self, "selected_var"):
            self.selected_var.set(self._t("no_target_selected"))

    def _update_db_count_display(self):
        if not hasattr(self, "db_count_var"):
            return
        labels = [("1yuan", "1"), ("5yuan", "5"), ("10yuan", "10"), ("50yuan", "50")]
        counts = {short: 0 for _key, short in labels}
        try:
            data = json.loads(COIN_PARAM_DB_FILE.read_text(encoding="utf-8"))
            coins = data.get("coins", {})
            for key, short in labels:
                spec = coins.get(key, {})
                kept = len(spec.get("learned_samples") or [])
                total = int(spec.get("learned_total_count", kept) or kept)
                counts[short] = f"{kept}/{total}" if total != kept else str(kept)
        except Exception:
            pass
        self.db_count_var.set(f"DB 1:{counts['1']}  5:{counts['5']}  10:{counts['10']}  50:{counts['50']}")

    def _fmt(self, value, suffix=""):
        if value is None:
            return "-"
        try:
            return f"{float(value):.2f}{suffix}"
        except Exception:
            return str(value)

    def _state_text(self, state):
        zh = {
            "idle": "待機", "running": "執行中", "done": "完成",
            "failed": "失敗", "return_start": "返回安全位置", "moving": "移動中",
        }
        en = {
            "idle": "idle", "running": "running", "done": "done",
            "failed": "failed", "return_start": "returning", "moving": "moving",
        }
        return (zh if self.ui_language == "zh" else en).get(str(state), str(state))

    def _latest_snapshot(self):
        min_time = self._session_started_at
        if self.camera_view.get() == "Quality":
            preview = OUT_DIR / "live_preview_quality.jpg"
            if preview.exists() and preview.stat().st_mtime >= min_time:
                return preview
            return None
        if self.camera_view.get() == "Gemini":
            preview = OUT_DIR / "live_preview_gemini.jpg"
            if preview.exists() and preview.stat().st_mtime >= min_time:
                return preview
            return None
        if self.camera_view.get() == "Combined":
            preview = OUT_DIR / "live_preview_combined.jpg"
            if preview.exists() and preview.stat().st_mtime >= min_time:
                return preview
            return None
        prefix = {
            "Gemini": "gemini_view_*.jpg",
            "Quality": "quality_view_*.jpg",
            "Combined": "dual_camera_snapshot_*.jpg",
        }.get(self.camera_view.get(), "gemini_view_*.jpg")
        files = sorted(OUT_DIR.glob(prefix), key=lambda p: p.stat().st_mtime, reverse=True)
        files = [p for p in files if p.stat().st_mtime >= min_time]
        if not files and prefix != "dual_camera_snapshot_*.jpg":
            files = sorted(OUT_DIR.glob("dual_camera_snapshot_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
            files = [p for p in files if p.stat().st_mtime >= min_time]
        return files[0] if files else None

    def _show_preview_message(self, text):
        self._photo = None
        self._last_image_path = None
        self.image_label.configure(image="", text=text)

    def _display_pil_image(self, img, source_key=None):
        original_size = img.size
        box_w = max(400, self.image_label.winfo_width() - 4)
        box_h = max(280, self.image_label.winfo_height() - 4)
        fit_scale = min(box_w / img.width, box_h / img.height)
        scale = fit_scale * self.zoom
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if new_w > box_w or new_h > box_h:
            center_x = max(0.0, min(1.0, float(getattr(self, "_zoom_center", (0.5, 0.5))[0])))
            center_y = max(0.0, min(1.0, float(getattr(self, "_zoom_center", (0.5, 0.5))[1])))
            left = int(round(center_x * new_w - box_w / 2.0))
            top = int(round(center_y * new_h - box_h / 2.0))
            left = max(0, min(left, max(0, new_w - box_w)))
            top = max(0, min(top, max(0, new_h - box_h)))
            img = img.crop((left, top, left + min(box_w, new_w), top + min(box_h, new_h)))
            self._crop_offset_scaled = (left, top)
        else:
            self._crop_offset_scaled = (0, 0)
        self._loaded_image_size = original_size
        self._display_image_size = img.size
        label_w = max(1, self.image_label.winfo_width())
        label_h = max(1, self.image_label.winfo_height())
        self._display_image_offset = ((label_w - img.size[0]) // 2, (label_h - img.size[1]) // 2)
        self._image_scale = scale
        self._photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self._photo, text="")
        self._last_image_path = source_key

    def _draw_target_overlay_cv(self, frame, view):
        if frame is None:
            return frame
        out = frame.copy()
        cfg = self._load_config()
        pick_plan = self._get_display_pick_plan(self.targets_data.get("targets", []))
        rank_limit = max(1, int(cfg.get("overlay_pick_rank_limit", 5)))
        hide_non_pick_labels = bool(cfg.get("overlay_hide_non_pick_labels_when_plan", True))
        pick_by_index = {}
        for item in pick_plan:
            try:
                if int(item.get("rank", 9999)) <= rank_limit:
                    pick_by_index[int(item["target"].get("index", -1))] = item
            except Exception:
                continue
        coin_colors = {
            "1NT": (255, 90, 90),
            "5NT": (70, 220, 70),
            "10NT": (0, 170, 255),
            "50NT": (220, 80, 220),
            "1yuan": (255, 90, 90),
            "5yuan": (70, 220, 70),
            "10yuan": (0, 170, 255),
            "50yuan": (220, 80, 220),
            "?": (0, 255, 255),
        }
        for t in self.targets_data.get("targets", []):
            pos = self._target_view_xy(t, view, cfg)
            if pos is None:
                continue
            x, y = int(round(pos[0])), int(round(pos[1]))
            if x < 0 or y < 0 or x >= out.shape[1] or y >= out.shape[0]:
                continue
            valid = bool(t.get("valid_for_pick"))
            label_name = t.get("label_name", "?")
            color = coin_colors.get(label_name, coin_colors.get(t.get("label", "?"), coin_colors["?"]))
            try:
                target_index = int(t.get("index", -1))
            except Exception:
                target_index = -1
            is_active_target = self._is_active_robot_target(t)
            pick_item = pick_by_index.get(target_index)
            status = "LIVE" if t.get("live_preview") else ("OK" if valid else "CHECK")
            display_scale = max(1.0, min(out.shape[1] / 640.0, out.shape[0] / 720.0))
            overlay_scale = max(0.2, float(cfg.get("quality_overlay_axis_scale", 1.8)))
            marker_size = max(18, int(round(18 * display_scale * overlay_scale)))
            circle_r = max(14, int(round(14 * display_scale * overlay_scale)))
            pick_r = max(20, int(round(20 * display_scale * overlay_scale)))
            line_w = max(2, int(round(2 * display_scale)))
            valid_line_w = max(line_w + 1, int(round(3 * display_scale)))
            text_scale = max(0.48, 0.48 * display_scale)
            text_w = max(1, int(round(display_scale)))
            pad = max(10, int(round(10 * display_scale)))
            cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, marker_size, line_w, cv2.LINE_AA)
            cv2.circle(out, (x, y), circle_r, color, valid_line_w if valid else line_w, cv2.LINE_AA)
            if valid and not t.get("live_preview"):
                cv2.circle(out, (x, y), pick_r, (0, 255, 80), line_w, cv2.LINE_AA)
            if is_active_target:
                active_r = max(pick_r + 14, int(round((pick_r + 14) * 1.2)))
                cv2.circle(out, (x, y), active_r, (0, 0, 255), max(4, int(round(5 * display_scale))), cv2.LINE_AA)
            if pick_item is not None:
                rank = int(pick_item.get("rank", 9999))
                rank_color = (0, 255, 0) if rank == 1 else (0, 220, 255)
                rank_r = max(pick_r + 8, int(round((pick_r + 8) * (1.15 if rank == 1 else 1.0))))
                rank_w = max(3, int(round((4 if rank == 1 else 2) * display_scale)))
                cv2.circle(out, (x, y), rank_r, rank_color, rank_w, cv2.LINE_AA)
            label_mode = str(cfg.get("overlay_label_mode", "q_only")).lower()
            if is_active_target:
                label = f"ACTIVE Q{t.get('index')} {label_name}"
            elif pick_item is not None:
                rank = int(pick_item.get("rank", 9999))
                rank_text = "PREVIEW NEXT" if rank == 1 else f"PREVIEW #{rank}"
                label = f"{rank_text} Q{t.get('index')}"
                if label_mode not in ("q_only", "q", "compact"):
                    label = (
                        f"{rank_text} Q{t.get('index')} {label_name} "
                        f"v={float(pick_item.get('visible', 0.0)):.2f} "
                        f"r={float(pick_item.get('axis', 0.0)):.2f}"
                    )
            else:
                if hide_non_pick_labels and pick_plan:
                    continue
                label = f"Q{t.get('index')}" if label_mode in ("q_only", "q", "compact") else f"Q{t.get('index')} {label_name} {status}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_w)
            text_x = x + pad
            text_y = max(th + pad, y - pad)
            label_color = (0, 0, 255) if is_active_target else ((0, 255, 0) if pick_item is not None and int(pick_item.get("rank", 9999)) == 1 else color)
            if not is_active_target and pick_item is not None and int(pick_item.get("rank", 9999)) != 1:
                label_color = (0, 220, 255)
            cv2.rectangle(
                out,
                (text_x - int(4 * display_scale), max(0, text_y - th - int(5 * display_scale))),
                (min(out.shape[1] - 1, text_x + tw + int(4 * display_scale)), min(out.shape[0] - 1, text_y + baseline + int(4 * display_scale))),
                (25, 25, 25),
                -1,
            )
            cv2.putText(out, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, label_color, text_w, cv2.LINE_AA)
        return out

    def _draw_quality_overlay_on_cropped_frame(self, frame):
        self._quality_overlay_cropped = True
        try:
            return self._draw_target_overlay_cv(frame, "Quality")
        finally:
            self._quality_overlay_cropped = False

    def _resize_pil_to_height(self, img, height):
        if img.height <= 0:
            return img
        width = max(1, int(img.width * (height / img.height)))
        return img.resize((width, height), Image.Resampling.LANCZOS)

    def _load_latest_image(self, force=False):
        path = self._latest_snapshot()
        if path is None:
            self._show_preview_message(self._t("no_image"))
            return
        if not force and path == self._last_image_path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self._display_pil_image(img, path)
        except Exception as e:
            self._log(f"影像載入失敗：{e}")

    def _open_quality_preview(self):
        if self._quality_cap is not None and self._quality_cap.isOpened():
            return self._quality_cap
        cfg = self._load_config()
        backend_name = str(cfg.get("quality_camera_backend", "dshow")).lower()
        backends = {
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "any": cv2.CAP_ANY,
        }
        backend_order = [backend_name, "dshow"] if backend_name != "msmf" else ["msmf", "dshow"]
        backend_order = list(dict.fromkeys(name for name in backend_order if name in backends))
        cap = None
        for name in backend_order:
            candidate = cv2.VideoCapture(int(cfg.get("quality_camera_index", 0)), backends[name])
            if candidate.isOpened():
                cap = candidate
                break
            candidate.release()
        if cap is None or not cap.isOpened():
            return None

        def apply_props(width, height, fps):
            fourcc = str(cfg.get("quality_fourcc", "MJPG")).strip().upper()
            if fourcc and fourcc not in ("NONE", "DEFAULT", "0"):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4].ljust(4)))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            cap.set(cv2.CAP_PROP_FPS, int(fps))
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, int(cfg.get("quality_buffer_size", 1)))
            except Exception:
                pass

        def read_after_warmup():
            frame = None
            ok = False
            for _ in range(max(1, int(cfg.get("quality_warmup_frames", 8)))):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
            return ok, frame

        def frame_matches_request(frame, width, height):
            if frame is None:
                return False
            actual_h, actual_w = frame.shape[:2]
            min_ratio = float(cfg.get("quality_min_actual_resolution_ratio", 0.80))
            return actual_w >= int(width) * min_ratio and actual_h >= int(height) * min_ratio

        apply_props(cfg.get("quality_width", 1920), cfg.get("quality_height", 1080), cfg.get("quality_fps", 15))
        ok, _frame = read_after_warmup()
        if ok and not frame_matches_request(_frame, cfg.get("quality_width", 1920), cfg.get("quality_height", 1080)):
            ok = False
            self._log(
                f"畫質相機解析度不符：要求 {cfg.get('quality_width', 1920)}x{cfg.get('quality_height', 1080)}，"
                f"實際 {_frame.shape[1]}x{_frame.shape[0]}，改試 fallback。"
            )
        if not ok:
            fallback_modes = cfg.get("quality_fallback_modes") or [
                [cfg.get("quality_fallback_width", 1280), cfg.get("quality_fallback_height", 720), cfg.get("quality_fallback_fps", 15)]
            ]
            for mode in fallback_modes:
                try:
                    width, height, fps = mode
                except Exception:
                    continue
                apply_props(width, height, fps)
                ok, _frame = read_after_warmup()
                if ok and frame_matches_request(_frame, width, height):
                    break
                if ok:
                    self._log(f"畫質相機 fallback 解析度不符：要求 {width}x{height}，實際 {_frame.shape[1]}x{_frame.shape[0]}，跳過。")
                    ok = False
        if not ok:
            cap.release()
            return None
        self._quality_cap = cap
        self._quality_preview_error_logged = False
        return cap

    def _close_quality_preview(self):
        self._stop_yolo_live_detection()
        self._quality_stop.set()
        thread = self._quality_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        self._quality_thread = None
        if self._quality_cap is not None:
            try:
                self._quality_cap.release()
            except Exception:
                pass
            self._quality_cap = None
        time.sleep(0.35)
        with self._quality_lock:
            self._quality_latest_frame = None
            self._quality_latest_id = 0
            self._quality_displayed_id = 0

    def _crop_quality_roi(self, frame):
        roi = self._load_config().get("quality_roi")
        if not roi:
            return frame
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in roi]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        return frame[y1:y2, x1:x2].copy()

    def _start_quality_preview_thread(self):
        if self._quality_thread is not None and self._quality_thread.is_alive():
            return
        self._quality_stop.clear()

        def worker():
            cap = None
            for _ in range(5):
                if self._quality_stop.is_set():
                    return
                cap = self._open_quality_preview()
                if cap is not None:
                    break
                time.sleep(0.25)
            if cap is None:
                self.after(0, self._schedule_quality_open_error)
                return
            while not self._quality_stop.is_set():
                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._quality_lock:
                        self._quality_latest_frame = frame
                        self._quality_latest_id += 1
                    if self._quality_preview_error_logged or self._quality_preview_error_after is not None:
                        self.after(0, self._clear_quality_open_error)
                time.sleep(0.001)

        self._quality_thread = threading.Thread(target=worker, daemon=True)
        self._quality_thread.start()

    def _update_quality_live_frame(self):
        with self._quality_lock:
            if self._quality_latest_frame is None or self._quality_latest_id == self._quality_displayed_id:
                return
            frame = self._quality_latest_frame.copy()
            self._quality_displayed_id = self._quality_latest_id
        raw = self._crop_quality_roi(frame)
        overlay = self._draw_quality_overlay_on_cropped_frame(raw)
        cfg = self._load_config()
        if bool(cfg.get("quality_live_compare_raw_overlay", True)):
            gap = max(4, int(raw.shape[1] * 0.01))
            h = max(raw.shape[0], overlay.shape[0])
            combined = np.zeros((h, raw.shape[1] + gap + overlay.shape[1], 3), dtype=np.uint8)
            combined[:raw.shape[0], :raw.shape[1]] = raw
            combined[:overlay.shape[0], raw.shape[1] + gap:] = overlay
            cv2.putText(combined, "RAW", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, "PICK", (raw.shape[1] + gap + 16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            shown = combined
        else:
            shown = overlay
        img = Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
        self._display_pil_image(img, "quality-live")

    def _latest_quality_frame_copy(self):
        with self._quality_lock:
            if self._quality_latest_frame is None:
                return None, self._quality_latest_id
            return self._quality_latest_frame.copy(), self._quality_latest_id

    def _latest_gemini_preview_pil(self):
        path = OUT_DIR / "live_preview_gemini.jpg"
        min_time = self._session_started_at
        if path.exists():
            mtime = path.stat().st_mtime
            if mtime >= min_time and mtime != self._gemini_preview_mtime:
                try:
                    self._gemini_preview_pil = Image.open(path).convert("RGB")
                    self._gemini_preview_mtime = mtime
                except Exception as exc:
                    self._log(f"深度相機預覽載入失敗：{exc}")
        return self._gemini_preview_pil

    def _update_combined_live_frame(self):
        frame, frame_id = self._latest_quality_frame_copy()
        if frame is None:
            return
        frame = self._crop_quality_roi(frame)
        frame = self._draw_quality_overlay_on_cropped_frame(frame)
        quality_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gemini_img = self._latest_gemini_preview_pil()
        target_h = 540
        quality_img = self._resize_pil_to_height(quality_img, target_h)
        if gemini_img is None:
            gemini_img = Image.new("RGB", (max(1, int(target_h * 16 / 9)), target_h), "#050505")
        else:
            gemini_img = self._resize_pil_to_height(gemini_img, target_h)
        gap = 8
        combined = Image.new("RGB", (quality_img.width + gap + gemini_img.width, target_h), "#000000")
        combined.paste(quality_img, (0, 0))
        combined.paste(gemini_img, (quality_img.width + gap, 0))
        self._quality_displayed_id = frame_id
        self._display_pil_image(combined, f"combined-live-{frame_id}-{self._gemini_preview_mtime}")

    def _on_camera_view_change(self):
        self._zoom_center = (0.5, 0.5)
        if self.camera_view.get() != "Quality":
            self._last_image_path = None
            if self.camera_view.get() == "Combined":
                self._start_quality_preview_thread()
                self._show_preview_message("正在開啟深度相機...")
                self._last_gemini_preview_request = time.time()
                self._preview_only("Gemini")
                self._update_combined_live_frame()
            else:
                self._close_quality_preview()
                self._show_preview_message("正在開啟相機...")
                self._preview_only()
            return
        self._start_quality_preview_thread()
        self._last_image_path = None
        self._update_quality_live_frame()

    def _on_close(self):
        self._stop_yolo_live_detection()
        self._close_offset_robot()
        self._close_quality_preview()
        self.destroy()

    def _refresh_current_view(self):
        if self.camera_view.get() == "Quality":
            self._start_quality_preview_thread()
            self._update_quality_live_frame()
        elif self.camera_view.get() == "Combined":
            self._start_quality_preview_thread()
            self._update_combined_live_frame()
        else:
            self._load_latest_image(force=True)

    def _resume_live_preview(self):
        self.auto_preview.set(True)
        self._refresh_current_view()
        for delay in (120, 300, 700, 1200):
            self.after(delay, self._refresh_current_view)

    def _on_image_wheel(self, event):
        view_xy = self._event_to_view_xy(event)
        if event.delta > 0:
            self.zoom = min(8.0, self.zoom * 1.18)
        else:
            self.zoom = max(1.0, self.zoom / 1.18)
        if view_xy is not None:
            view_x, view_y = view_xy
            img_w, img_h = self._loaded_image_size
            label_w = max(1, self.image_label.winfo_width())
            label_h = max(1, self.image_label.winfo_height())
            box_w = max(400, label_w - 4)
            box_h = max(280, label_h - 4)
            fit_scale = min(box_w / max(img_w, 1), box_h / max(img_h, 1))
            scale = max(fit_scale * self.zoom, 1e-6)
            # Keep the image point under the cursor after zooming.
            display_x = event.x - max(0, (label_w - min(box_w, int(img_w * scale))) // 2)
            display_y = event.y - max(0, (label_h - min(box_h, int(img_h * scale))) // 2)
            crop_left = view_x * scale - display_x
            crop_top = view_y * scale - display_y
            center_x = (crop_left + box_w / 2.0) / max(img_w * scale, 1.0)
            center_y = (crop_top + box_h / 2.0) / max(img_h * scale, 1.0)
            self._zoom_center = (
                max(0.0, min(1.0, center_x)),
                max(0.0, min(1.0, center_y)),
            )
        else:
            self._zoom_center = (0.5, 0.5)
        self._refresh_current_view()

    def _event_to_view_xy(self, event):
        off_x, off_y = self._display_image_offset
        disp_w, disp_h = self._display_image_size
        img_w, img_h = self._loaded_image_size
        if disp_w <= 0 or disp_h <= 0 or img_w <= 0 or img_h <= 0:
            return None
        x = event.x - off_x
        y = event.y - off_y
        if x < 0 or y < 0 or x >= disp_w or y >= disp_h:
            return None
        crop_x, crop_y = getattr(self, "_crop_offset_scaled", (0, 0))
        scale = max(getattr(self, "_image_scale", 1.0), 1e-6)
        return (x + crop_x) / scale, (y + crop_y) / scale

    def _load_action_status(self):
        if not ACTION_STATUS_FILE.exists():
            return
        try:
            data = json.loads(ACTION_STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        state = str(data.get("state", "idle"))
        self._active_robot_state = state
        t = data.get("target") or {}
        if t:
            self._planned_action_text = None
            try:
                self._active_robot_target_index = int(t.get("index"))
            except Exception:
                self._active_robot_target_index = None
            self._active_robot_target = dict(t)
            if hasattr(self, "pick_plan_var") and state not in ("idle", "done"):
                prefix = "實際拆堆目標" if state != "failed" else "上次失敗目標"
                self.pick_plan_var.set(
                    f"{prefix}：Q{t.get('index')} {t.get('label_name', '?')} | "
                    f"X={self._fmt(t.get('robot_x_mm'))} Y={self._fmt(t.get('robot_y_mm'))} | "
                    "此列為手臂實際執行目標，不是預覽排序"
                )
            self.action_var.set(
                f"目前動作：{self._state_text(data.get('state', '-'))}\n"
                f"Q{t.get('index')} {t.get('label_name', '?')}  "
                f"X={self._fmt(t.get('robot_x_mm'))} Y={self._fmt(t.get('robot_y_mm'))}"
            )
        else:
            if state in ("idle", "done", "failed"):
                self._active_robot_target_index = None
                self._active_robot_target = None
            if self._planned_action_text and time.time() < self._planned_action_until and data.get("state") in (None, "idle"):
                self.action_var.set(self._planned_action_text)
                return
            self.action_var.set(f"目前動作：{self._state_text(data.get('state', 'idle'))}  {data.get('message', '')}")
        if data.get("state") == "failed" and data.get("requires_human_intervention"):
            sig = json.dumps(data, sort_keys=True, ensure_ascii=False)
            if sig != self.last_failure_signature:
                self.last_failure_signature = sig
                self._show_robot_failure(data)

    def _reset_stale_action_status(self):
        if not ACTION_STATUS_FILE.exists():
            return
        try:
            data = json.loads(ACTION_STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        state = str(data.get("state", "idle"))
        if state not in ("failed", "done", "return_start"):
            return
        idle_status = {
            "state": "idle",
            "message": "UI started; previous robot action status cleared",
            "requires_human_intervention": False,
        }
        try:
            ACTION_STATUS_FILE.write_text(json.dumps(idle_status, indent=2, ensure_ascii=False), encoding="utf-8")
            self.last_failure_signature = json.dumps(idle_status, sort_keys=True, ensure_ascii=False)
        except Exception as exc:
            self._log(f"動作狀態重設失敗：{exc}")

    def _show_robot_failure(self, data):
        t = data.get("target") or {}
        attempted = data.get("attempted_robot_xyz_mm") or ["?", "?", "?"]
        code = data.get("error_code")
        hint = ROBOT_ERROR_HINTS.get(code, {
            "title": "未知 MG400 錯誤",
            "cause": "控制器回傳了本機提示表中尚未登錄的錯誤碼。",
            "action": "請到 DobotStudio 查看詳細 alarm，清除錯誤後從安全位置重試。",
        })
        text = (
            "MG400 動作失敗。\n\n"
            f"目標：Q{t.get('index', '?')} {t.get('label_name', '?')}\n"
            f"目標 XY：X={self._fmt(t.get('robot_x_mm'))}  Y={self._fmt(t.get('robot_y_mm'))}\n"
            f"嘗試位置：X={self._fmt(attempted[0])}  Y={self._fmt(attempted[1])}  Z={self._fmt(attempted[2])}\n"
            f"錯誤碼：{code if code is not None else '?'}\n"
            f"控制器回覆：{data.get('controller_response', '-')}\n"
            f"意思：{hint['title']}\n\n"
            f"可能原因：{hint['cause']}\n\n"
            f"建議處理：{hint['action']}"
        )
        self._log(text)
        messagebox.showerror("手臂需要處理", text)

    def _load_config(self):
        if CONFIG_FILE.exists():
            try:
                return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_config_values(self, **values):
        cfg = self._load_config()
        cfg.update(values)
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    def _robot_return_pose(self):
        cfg = self._load_config()
        return (
            float(cfg.get("robot_return_x_mm", self.return_x_var.get())),
            float(cfg.get("robot_return_y_mm", self.return_y_var.get())),
            float(cfg.get("robot_return_z_mm", self.return_z_var.get())),
        )

    def _robot_safe_bounds(self):
        cfg = self._load_config()
        return {
            "x_min": float(cfg.get("robot_safe_x_min_mm", self.safe_x_min_var.get())),
            "x_max": float(cfg.get("robot_safe_x_max_mm", self.safe_x_max_var.get())),
            "y_min": float(cfg.get("robot_safe_y_min_mm", self.safe_y_min_var.get())),
            "y_max": float(cfg.get("robot_safe_y_max_mm", self.safe_y_max_var.get())),
            "z_min": float(cfg.get("robot_safe_z_min_mm", self.safe_z_min_var.get())),
            "z_max": float(cfg.get("robot_safe_z_max_mm", self.safe_z_max_var.get())),
        }

    def _sync_pick_offset_from_config(self):
        cfg = self._load_config()
        self.pick_offset_x_var.set(float(cfg.get("robot_target_offset_x_mm", 0.0)))
        self.pick_offset_y_var.set(float(cfg.get("robot_target_offset_y_mm", 0.0)))
        self.pick_offset_z_var.set(float(cfg.get("robot_target_offset_z_mm", 0.0)))
        self.offset_jog_step_var.set(float(cfg.get("offset_calib_jog_step_mm", self.offset_jog_step_var.get())))
        self.pick_x_min_var.set(float(cfg.get("robot_pick_x_min_mm", self.pick_x_min_var.get())))
        self.pick_x_max_var.set(float(cfg.get("robot_pick_x_max_mm", self.pick_x_max_var.get())))
        self.pick_y_min_var.set(float(cfg.get("robot_pick_y_min_mm", self.pick_y_min_var.get())))
        self.pick_y_max_var.set(float(cfg.get("robot_pick_y_max_mm", self.pick_y_max_var.get())))
        self.safe_x_min_var.set(float(cfg.get("robot_safe_x_min_mm", self.safe_x_min_var.get())))
        self.safe_x_max_var.set(float(cfg.get("robot_safe_x_max_mm", self.safe_x_max_var.get())))
        self.safe_y_min_var.set(float(cfg.get("robot_safe_y_min_mm", self.safe_y_min_var.get())))
        self.safe_y_max_var.set(float(cfg.get("robot_safe_y_max_mm", self.safe_y_max_var.get())))
        self.safe_z_min_var.set(float(cfg.get("robot_safe_z_min_mm", self.safe_z_min_var.get())))
        self.safe_z_max_var.set(float(cfg.get("robot_safe_z_max_mm", self.safe_z_max_var.get())))
        self.return_x_var.set(float(cfg.get("robot_return_x_mm", self.return_x_var.get())))
        self.return_y_var.set(float(cfg.get("robot_return_y_mm", self.return_y_var.get())))
        self.return_z_var.set(float(cfg.get("robot_return_z_mm", self.return_z_var.get())))
        self.offset_calib_travel_z_var.set(float(cfg.get("offset_calib_travel_z_mm", self.offset_calib_travel_z_var.get())))
        self.offset_calib_target_z_var.set(float(cfg.get("offset_calib_target_z_mm", self.offset_calib_target_z_var.get())))
        self._update_pick_offset_label()

    def _update_pick_offset_label(self):
        if hasattr(self, "pick_offset_label"):
            self.pick_offset_label.set(
                f"目前 X={self.pick_offset_x_var.get():+.1f}mm  Y={self.pick_offset_y_var.get():+.1f}mm  Z={self.pick_offset_z_var.get():+.1f}mm"
            )

    def _apply_pick_offset_to_current_targets(self, new_x, new_y, old_x=None, old_y=None, new_z=None, old_z=None):
        if not TARGETS_FILE.exists():
            return
        old_x = float(self.pick_offset_x_var.get()) if old_x is None else float(old_x)
        old_y = float(self.pick_offset_y_var.get()) if old_y is None else float(old_y)
        old_z = float(self.pick_offset_z_var.get()) if old_z is None else float(old_z)
        new_z = float(self.pick_offset_z_var.get()) if new_z is None else float(new_z)
        try:
            data = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
            for t in data.get("targets", []):
                raw_x = t.get("raw_robot_x_mm")
                raw_y = t.get("raw_robot_y_mm")
                raw_z = t.get("raw_robot_z_mm")
                if raw_x is None and t.get("robot_x_mm") is not None:
                    raw_x = float(t["robot_x_mm"]) - old_x
                    t["raw_robot_x_mm"] = round(raw_x, 3)
                if raw_y is None and t.get("robot_y_mm") is not None:
                    raw_y = float(t["robot_y_mm"]) - old_y
                    t["raw_robot_y_mm"] = round(raw_y, 3)
                if raw_z is None and t.get("robot_z_mm") is not None:
                    raw_z = float(t["robot_z_mm"]) - old_z
                    t["raw_robot_z_mm"] = round(raw_z, 3)
                if raw_x is not None:
                    t["robot_x_mm"] = round(float(raw_x) + float(new_x), 3)
                if raw_y is not None:
                    t["robot_y_mm"] = round(float(raw_y) + float(new_y), 3)
                if raw_z is not None:
                    t["robot_z_mm"] = round(float(raw_z) + float(new_z), 3)
                    t["applied_offset_z_mm"] = round(float(new_z), 3)
            TARGETS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            self._log(f"更新目前目標偏移失敗：{exc}")

    def _set_pick_offset(self, x, y, update_current=True):
        old_x = float(self.pick_offset_x_var.get())
        old_y = float(self.pick_offset_y_var.get())
        new_x = float(x)
        new_y = float(y)
        self.pick_offset_x_var.set(new_x)
        self.pick_offset_y_var.set(new_y)
        self._save_config_values(robot_target_offset_x_mm=new_x, robot_target_offset_y_mm=new_y)
        if update_current:
            self._apply_pick_offset_to_current_targets(new_x, new_y, old_x, old_y)
        self._update_pick_offset_label()
        self._load_targets()
        self._refresh_current_view()
        self._log(f"夾取偏移已更新：X={new_x:+.1f}mm  Y={new_y:+.1f}mm")

    def _nudge_pick_offset(self, dx, dy):
        self._set_pick_offset(self.pick_offset_x_var.get() + dx, self.pick_offset_y_var.get() + dy)

    def _open_settings(self):
        win = tk.Toplevel(self)
        win.title(self._t("settings"))
        win.configure(bg="#2b2f33")
        win.resizable(True, True)
        win.minsize(520, 560)
        win.transient(self)
        win.grab_set()

        mode_var = tk.StringVar(value=self.ui_mode)
        lang_var = tk.StringVar(value=self.ui_language)
        overlay_var = tk.StringVar(value=str(self.ui_config.get("overlay_label_mode", "q_only")))
        detection_var = tk.StringVar(value=str(self.ui_config.get("quality_detection_method", "sam3_ellipse")))
        move_var = tk.IntVar(value=int(self.move_speed_var.get()))
        lower_var = tk.IntVar(value=int(self.lower_speed_var.get()))
        move_speed_j_var = tk.IntVar(value=int(self.ui_config.get("robot_move_speed_j_pct", 80)))
        move_acc_j_var = tk.IntVar(value=int(self.ui_config.get("robot_move_acc_j_pct", 70)))
        move_speed_l_var = tk.IntVar(value=int(self.ui_config.get("robot_move_speed_l_pct", 70)))
        move_acc_l_var = tk.IntVar(value=int(self.ui_config.get("robot_move_acc_l_pct", 60)))
        lower_speed_l_var = tk.IntVar(value=int(self.ui_config.get("robot_lower_speed_l_pct", 35)))
        lower_acc_l_var = tk.IntVar(value=int(self.ui_config.get("robot_lower_acc_l_pct", 25)))
        offset_x_var = tk.DoubleVar(value=float(self.pick_offset_x_var.get()))
        offset_y_var = tk.DoubleVar(value=float(self.pick_offset_y_var.get()))
        offset_z_var = tk.DoubleVar(value=float(self.pick_offset_z_var.get()))
        pick_touch_offset_var = tk.DoubleVar(value=float(self.ui_config.get("unstack_pick_touch_offset_mm", -2.0)))
        pick_x_min_var = tk.DoubleVar(value=float(self.pick_x_min_var.get()))
        pick_x_max_var = tk.DoubleVar(value=float(self.pick_x_max_var.get()))
        pick_y_min_var = tk.DoubleVar(value=float(self.pick_y_min_var.get()))
        pick_y_max_var = tk.DoubleVar(value=float(self.pick_y_max_var.get()))
        safe_x_min_var = tk.DoubleVar(value=float(self.safe_x_min_var.get()))
        safe_x_max_var = tk.DoubleVar(value=float(self.safe_x_max_var.get()))
        safe_y_min_var = tk.DoubleVar(value=float(self.safe_y_min_var.get()))
        safe_y_max_var = tk.DoubleVar(value=float(self.safe_y_max_var.get()))
        safe_z_min_var = tk.DoubleVar(value=float(self.safe_z_min_var.get()))
        safe_z_max_var = tk.DoubleVar(value=float(self.safe_z_max_var.get()))
        auto_x_min_var = tk.DoubleVar(value=float(self.ui_config.get("robot_auto_x_min_mm", self.safe_x_min_var.get())))
        auto_x_max_var = tk.DoubleVar(value=float(self.ui_config.get("robot_auto_x_max_mm", self.safe_x_max_var.get())))
        auto_y_min_var = tk.DoubleVar(value=float(self.ui_config.get("robot_auto_y_min_mm", self.safe_y_min_var.get())))
        auto_y_max_var = tk.DoubleVar(value=float(self.ui_config.get("robot_auto_y_max_mm", self.safe_y_max_var.get())))
        lift_x_min_var = tk.DoubleVar(value=float(self.ui_config.get("robot_lift_x_min_mm", auto_x_min_var.get())))
        lift_x_max_var = tk.DoubleVar(value=float(self.ui_config.get("robot_lift_x_max_mm", auto_x_max_var.get())))
        lift_y_min_var = tk.DoubleVar(value=float(self.ui_config.get("robot_lift_y_min_mm", auto_y_min_var.get())))
        lift_y_max_var = tk.DoubleVar(value=float(self.ui_config.get("robot_lift_y_max_mm", auto_y_max_var.get())))
        return_x_var = tk.DoubleVar(value=float(self.return_x_var.get()))
        return_y_var = tk.DoubleVar(value=float(self.return_y_var.get()))
        return_z_var = tk.DoubleVar(value=float(self.return_z_var.get()))
        offset_calib_travel_z_var = tk.DoubleVar(value=float(self.offset_calib_travel_z_var.get()))
        offset_calib_target_z_var = tk.DoubleVar(value=float(self.offset_calib_target_z_var.get()))

        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)
        container = ttk.Frame(win, style="Panel.TFrame")
        container.grid(row=0, column=0, sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        canvas = tk.Canvas(container, bg="#2b2f33", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        frame = ttk.Frame(canvas, style="Panel.TFrame", padding=16)
        frame_window = canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(frame_window, width=e.width))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=self._t("mode"), style="Panel.TLabel").grid(row=0, column=0, sticky="w", pady=6)
        mode_box = ttk.Frame(frame, style="Panel.TFrame")
        mode_box.grid(row=0, column=1, sticky="w", pady=6)
        ttk.Radiobutton(mode_box, text=self._t("operator"), value="operator", variable=mode_var).grid(row=0, column=0, padx=(0, 12))
        ttk.Radiobutton(mode_box, text=self._t("engineer"), value="engineer", variable=mode_var).grid(row=0, column=1, padx=(0, 12))
        ttk.Radiobutton(mode_box, text=self._t("offset_calib"), value="offset_calib", variable=mode_var).grid(row=0, column=2)

        ttk.Label(frame, text=self._t("language"), style="Panel.TLabel").grid(row=1, column=0, sticky="w", pady=6)
        lang_box = ttk.Frame(frame, style="Panel.TFrame")
        lang_box.grid(row=1, column=1, sticky="w", pady=6)
        ttk.Radiobutton(lang_box, text="中文", value="zh", variable=lang_var).grid(row=0, column=0, padx=(0, 12))
        ttk.Radiobutton(lang_box, text="English", value="en", variable=lang_var).grid(row=0, column=1)

        ttk.Label(frame, text=self._t("move_speed"), style="Panel.TLabel").grid(row=2, column=0, sticky="w", pady=6)
        move_scale = ttk.Scale(frame, from_=5, to=80, variable=move_var)
        move_scale.grid(row=2, column=1, sticky="ew", pady=6)
        move_label = ttk.Label(frame, text=f"{move_var.get()}%", style="Panel.TLabel")
        move_label.grid(row=2, column=2, padx=(8, 0))

        ttk.Label(frame, text=self._t("lower_speed"), style="Panel.TLabel").grid(row=3, column=0, sticky="w", pady=6)
        lower_scale = ttk.Scale(frame, from_=5, to=60, variable=lower_var)
        lower_scale.grid(row=3, column=1, sticky="ew", pady=6)
        lower_label = ttk.Label(frame, text=f"{lower_var.get()}%", style="Panel.TLabel")
        lower_label.grid(row=3, column=2, padx=(8, 0))

        ttk.Label(frame, text="夾取 X 偏移 mm", style="Panel.TLabel").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Spinbox(frame, from_=-60.0, to=60.0, increment=0.5, textvariable=offset_x_var, width=8).grid(row=4, column=1, sticky="w", pady=6)
        ttk.Label(frame, text="夾取 Y 偏移 mm", style="Panel.TLabel").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Spinbox(frame, from_=-60.0, to=60.0, increment=0.5, textvariable=offset_y_var, width=8).grid(row=5, column=1, sticky="w", pady=6)

        ttk.Label(frame, text="夾取 Z 偏移 mm", style="Panel.TLabel").grid(row=6, column=0, sticky="w", pady=6)
        ttk.Spinbox(frame, from_=-60.0, to=60.0, increment=0.5, textvariable=offset_z_var, width=8).grid(row=6, column=1, sticky="w", pady=6)

        ttk.Label(frame, text="吸盤接觸補償 mm", style="Panel.TLabel").grid(row=7, column=0, sticky="w", pady=6)
        ttk.Spinbox(frame, from_=-8.0, to=4.0, increment=0.2, textvariable=pick_touch_offset_var, width=8).grid(row=7, column=1, sticky="w", pady=6)

        ttk.Label(frame, text="可取 X 範圍 mm", style="Panel.TLabel").grid(row=8, column=0, sticky="w", pady=6)
        pick_x_box = ttk.Frame(frame, style="Panel.TFrame")
        pick_x_box.grid(row=8, column=1, sticky="w", pady=6)
        ttk.Spinbox(pick_x_box, from_=-200.0, to=500.0, increment=1.0, textvariable=pick_x_min_var, width=8).grid(row=0, column=0)
        ttk.Label(pick_x_box, text=" ~ ", style="Panel.TLabel").grid(row=0, column=1)
        ttk.Spinbox(pick_x_box, from_=-200.0, to=500.0, increment=1.0, textvariable=pick_x_max_var, width=8).grid(row=0, column=2)

        ttk.Label(frame, text="可取 Y 範圍 mm", style="Panel.TLabel").grid(row=9, column=0, sticky="w", pady=6)
        pick_y_box = ttk.Frame(frame, style="Panel.TFrame")
        pick_y_box.grid(row=9, column=1, sticky="w", pady=6)
        ttk.Spinbox(pick_y_box, from_=-330.0, to=250.0, increment=1.0, textvariable=pick_y_min_var, width=8).grid(row=0, column=0)
        ttk.Label(pick_y_box, text=" ~ ", style="Panel.TLabel").grid(row=0, column=1)
        ttk.Spinbox(pick_y_box, from_=-330.0, to=250.0, increment=1.0, textvariable=pick_y_max_var, width=8).grid(row=0, column=2)

        ttk.Label(frame, text="手臂安全 X 範圍 mm", style="Panel.TLabel").grid(row=10, column=0, sticky="w", pady=6)
        safe_x_box = ttk.Frame(frame, style="Panel.TFrame")
        safe_x_box.grid(row=10, column=1, sticky="w", pady=6)
        ttk.Spinbox(safe_x_box, from_=-200.0, to=500.0, increment=1.0, textvariable=safe_x_min_var, width=8).grid(row=0, column=0)
        ttk.Label(safe_x_box, text=" ~ ", style="Panel.TLabel").grid(row=0, column=1)
        ttk.Spinbox(safe_x_box, from_=-200.0, to=500.0, increment=1.0, textvariable=safe_x_max_var, width=8).grid(row=0, column=2)

        ttk.Label(frame, text="手臂安全 Y 範圍 mm", style="Panel.TLabel").grid(row=11, column=0, sticky="w", pady=6)
        safe_y_box = ttk.Frame(frame, style="Panel.TFrame")
        safe_y_box.grid(row=11, column=1, sticky="w", pady=6)
        ttk.Spinbox(safe_y_box, from_=-400.0, to=400.0, increment=1.0, textvariable=safe_y_min_var, width=8).grid(row=0, column=0)
        ttk.Label(safe_y_box, text=" ~ ", style="Panel.TLabel").grid(row=0, column=1)
        ttk.Spinbox(safe_y_box, from_=-400.0, to=400.0, increment=1.0, textvariable=safe_y_max_var, width=8).grid(row=0, column=2)

        ttk.Label(frame, text="自動動作 X 範圍 mm", style="Panel.TLabel").grid(row=12, column=0, sticky="w", pady=6)
        auto_x_box = ttk.Frame(frame, style="Panel.TFrame")
        auto_x_box.grid(row=12, column=1, sticky="w", pady=6)
        ttk.Spinbox(auto_x_box, from_=-200.0, to=500.0, increment=1.0, textvariable=auto_x_min_var, width=8).grid(row=0, column=0)
        ttk.Label(auto_x_box, text=" ~ ", style="Panel.TLabel").grid(row=0, column=1)
        ttk.Spinbox(auto_x_box, from_=-200.0, to=500.0, increment=1.0, textvariable=auto_x_max_var, width=8).grid(row=0, column=2)

        ttk.Label(frame, text="自動動作 Y 範圍 mm", style="Panel.TLabel").grid(row=13, column=0, sticky="w", pady=6)
        auto_y_box = ttk.Frame(frame, style="Panel.TFrame")
        auto_y_box.grid(row=13, column=1, sticky="w", pady=6)
        ttk.Spinbox(auto_y_box, from_=-400.0, to=400.0, increment=1.0, textvariable=auto_y_min_var, width=8).grid(row=0, column=0)
        ttk.Label(auto_y_box, text=" ~ ", style="Panel.TLabel").grid(row=0, column=1)
        ttk.Spinbox(auto_y_box, from_=-400.0, to=400.0, increment=1.0, textvariable=auto_y_max_var, width=8).grid(row=0, column=2)

        ttk.Label(frame, text="抬高安全 X 範圍 mm", style="Panel.TLabel").grid(row=14, column=0, sticky="w", pady=6)
        lift_x_box = ttk.Frame(frame, style="Panel.TFrame")
        lift_x_box.grid(row=14, column=1, sticky="w", pady=6)
        ttk.Spinbox(lift_x_box, from_=-200.0, to=500.0, increment=1.0, textvariable=lift_x_min_var, width=8).grid(row=0, column=0)
        ttk.Label(lift_x_box, text=" ~ ", style="Panel.TLabel").grid(row=0, column=1)
        ttk.Spinbox(lift_x_box, from_=-200.0, to=500.0, increment=1.0, textvariable=lift_x_max_var, width=8).grid(row=0, column=2)

        ttk.Label(frame, text="抬高安全 Y 範圍 mm", style="Panel.TLabel").grid(row=15, column=0, sticky="w", pady=6)
        lift_y_box = ttk.Frame(frame, style="Panel.TFrame")
        lift_y_box.grid(row=15, column=1, sticky="w", pady=6)
        ttk.Spinbox(lift_y_box, from_=-400.0, to=400.0, increment=1.0, textvariable=lift_y_min_var, width=8).grid(row=0, column=0)
        ttk.Label(lift_y_box, text=" ~ ", style="Panel.TLabel").grid(row=0, column=1)
        ttk.Spinbox(lift_y_box, from_=-400.0, to=400.0, increment=1.0, textvariable=lift_y_max_var, width=8).grid(row=0, column=2)

        ttk.Label(frame, text="手臂安全 Z 範圍 mm", style="Panel.TLabel").grid(row=16, column=0, sticky="w", pady=6)
        safe_z_box = ttk.Frame(frame, style="Panel.TFrame")
        safe_z_box.grid(row=16, column=1, sticky="w", pady=6)
        ttk.Spinbox(safe_z_box, from_=-250.0, to=250.0, increment=1.0, textvariable=safe_z_min_var, width=8).grid(row=0, column=0)
        ttk.Label(safe_z_box, text=" ~ ", style="Panel.TLabel").grid(row=0, column=1)
        ttk.Spinbox(safe_z_box, from_=-250.0, to=250.0, increment=1.0, textvariable=safe_z_max_var, width=8).grid(row=0, column=2)

        ttk.Label(frame, text="回去位置 X/Y/Z mm", style="Panel.TLabel").grid(row=17, column=0, sticky="w", pady=6)
        return_box = ttk.Frame(frame, style="Panel.TFrame")
        return_box.grid(row=17, column=1, sticky="w", pady=6)
        ttk.Spinbox(return_box, from_=-200.0, to=500.0, increment=1.0, textvariable=return_x_var, width=7).grid(row=0, column=0)
        ttk.Spinbox(return_box, from_=-400.0, to=400.0, increment=1.0, textvariable=return_y_var, width=7).grid(row=0, column=1, padx=(6, 0))
        ttk.Spinbox(return_box, from_=-250.0, to=250.0, increment=1.0, textvariable=return_z_var, width=7).grid(row=0, column=2, padx=(6, 0))

        ttk.Label(frame, text="校正平移 Z mm", style="Panel.TLabel").grid(row=18, column=0, sticky="w", pady=6)
        ttk.Spinbox(frame, from_=-250.0, to=250.0, increment=1.0, textvariable=offset_calib_travel_z_var, width=8).grid(row=18, column=1, sticky="w", pady=6)

        ttk.Label(frame, text="校正下降 Z mm", style="Panel.TLabel").grid(row=19, column=0, sticky="w", pady=6)
        ttk.Spinbox(frame, from_=-250.0, to=250.0, increment=1.0, textvariable=offset_calib_target_z_var, width=8).grid(row=19, column=1, sticky="w", pady=6)

        ttk.Label(frame, text="畫面標籤", style="Panel.TLabel").grid(row=20, column=0, sticky="w", pady=6)
        overlay_box = ttk.Frame(frame, style="Panel.TFrame")
        overlay_box.grid(row=20, column=1, sticky="w", pady=6)
        ttk.Radiobutton(overlay_box, text="只顯示 Q 編號", value="q_only", variable=overlay_var).grid(row=0, column=0, padx=(0, 12))
        ttk.Radiobutton(overlay_box, text="全部顯示", value="full", variable=overlay_var).grid(row=0, column=1)

        ttk.Label(frame, text="辨識模式", style="Panel.TLabel").grid(row=21, column=0, sticky="w", pady=6)
        detection_box = ttk.Frame(frame, style="Panel.TFrame")
        detection_box.grid(row=21, column=1, sticky="w", pady=6)
        ttk.Radiobutton(detection_box, text="YOLO", value="yolo", variable=detection_var).grid(row=0, column=0, padx=(0, 12))
        ttk.Radiobutton(detection_box, text="SAM3", value="sam3_ellipse", variable=detection_var).grid(row=0, column=1)

        ttk.Label(frame, text="MovJ 速度/加速度 %", style="Panel.TLabel").grid(row=22, column=0, sticky="w", pady=6)
        movj_box = ttk.Frame(frame, style="Panel.TFrame")
        movj_box.grid(row=22, column=1, sticky="w", pady=6)
        ttk.Spinbox(movj_box, from_=5, to=100, increment=1, textvariable=move_speed_j_var, width=7).grid(row=0, column=0)
        ttk.Spinbox(movj_box, from_=5, to=100, increment=1, textvariable=move_acc_j_var, width=7).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(frame, text="MovL 移動速度/加速度 %", style="Panel.TLabel").grid(row=23, column=0, sticky="w", pady=6)
        movl_box = ttk.Frame(frame, style="Panel.TFrame")
        movl_box.grid(row=23, column=1, sticky="w", pady=6)
        ttk.Spinbox(movl_box, from_=5, to=100, increment=1, textvariable=move_speed_l_var, width=7).grid(row=0, column=0)
        ttk.Spinbox(movl_box, from_=5, to=100, increment=1, textvariable=move_acc_l_var, width=7).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(frame, text="下降 MovL 速度/加速度 %", style="Panel.TLabel").grid(row=24, column=0, sticky="w", pady=6)
        lower_profile_box = ttk.Frame(frame, style="Panel.TFrame")
        lower_profile_box.grid(row=24, column=1, sticky="w", pady=6)
        ttk.Spinbox(lower_profile_box, from_=5, to=100, increment=1, textvariable=lower_speed_l_var, width=7).grid(row=0, column=0)
        ttk.Spinbox(lower_profile_box, from_=5, to=100, increment=1, textvariable=lower_acc_l_var, width=7).grid(row=0, column=1, padx=(6, 0))

        def update_speed_labels(_=None):
            move_label.configure(text=f"{int(move_var.get())}%")
            lower_label.configure(text=f"{int(lower_var.get())}%")

        move_scale.configure(command=update_speed_labels)
        lower_scale.configure(command=update_speed_labels)

        buttons = ttk.Frame(frame, style="Panel.TFrame")
        buttons.grid(row=25, column=0, columnspan=3, sticky="e", pady=(14, 0))

        def apply_settings():
            self.ui_mode = mode_var.get()
            self.ui_language = lang_var.get()
            self.move_speed_var.set(int(move_var.get()))
            self.lower_speed_var.set(int(lower_var.get()))
            old_x = float(self.pick_offset_x_var.get())
            old_y = float(self.pick_offset_y_var.get())
            old_z = float(self.pick_offset_z_var.get())
            new_x = float(offset_x_var.get())
            new_y = float(offset_y_var.get())
            new_z = float(offset_z_var.get())
            pick_touch_offset = float(pick_touch_offset_var.get())
            pick_x_min = float(pick_x_min_var.get())
            pick_x_max = float(pick_x_max_var.get())
            pick_y_min = float(pick_y_min_var.get())
            pick_y_max = float(pick_y_max_var.get())
            safe_x_min = float(safe_x_min_var.get())
            safe_x_max = float(safe_x_max_var.get())
            safe_y_min = float(safe_y_min_var.get())
            safe_y_max = float(safe_y_max_var.get())
            safe_z_min = float(safe_z_min_var.get())
            safe_z_max = float(safe_z_max_var.get())
            auto_x_min = float(auto_x_min_var.get())
            auto_x_max = float(auto_x_max_var.get())
            auto_y_min = float(auto_y_min_var.get())
            auto_y_max = float(auto_y_max_var.get())
            lift_x_min = float(lift_x_min_var.get())
            lift_x_max = float(lift_x_max_var.get())
            lift_y_min = float(lift_y_min_var.get())
            lift_y_max = float(lift_y_max_var.get())
            return_x = float(return_x_var.get())
            return_y = float(return_y_var.get())
            return_z = float(return_z_var.get())
            offset_calib_travel_z = float(offset_calib_travel_z_var.get())
            offset_calib_target_z = float(offset_calib_target_z_var.get())
            motion_values = {
                "robot_move_speed_j_pct": int(move_speed_j_var.get()),
                "robot_move_acc_j_pct": int(move_acc_j_var.get()),
                "robot_move_speed_l_pct": int(move_speed_l_var.get()),
                "robot_move_acc_l_pct": int(move_acc_l_var.get()),
                "robot_lower_speed_l_pct": int(lower_speed_l_var.get()),
                "robot_lower_acc_l_pct": int(lower_acc_l_var.get()),
            }
            if pick_x_min >= pick_x_max or pick_y_min >= pick_y_max:
                messagebox.showerror("設定錯誤", "可取範圍的最小值必須小於最大值。")
                return
            if safe_x_min >= safe_x_max or safe_y_min >= safe_y_max or safe_z_min >= safe_z_max:
                messagebox.showerror("設定錯誤", "手臂安全範圍的最小值必須小於最大值。")
                return
            if auto_x_min >= auto_x_max or auto_y_min >= auto_y_max:
                messagebox.showerror("設定錯誤", "自動動作範圍的最小值必須小於最大值。")
                return
            if lift_x_min >= lift_x_max or lift_y_min >= lift_y_max:
                messagebox.showerror("設定錯誤", "抬高安全範圍的最小值必須小於最大值。")
                return
            if any(v < 1 or v > 100 for v in motion_values.values()):
                messagebox.showerror("設定錯誤", "MovJ/MovL 速度與加速度必須在 1..100%。")
                return
            self.pick_offset_x_var.set(new_x)
            self.pick_offset_y_var.set(new_y)
            self.pick_offset_z_var.set(new_z)
            self.pick_x_min_var.set(pick_x_min)
            self.pick_x_max_var.set(pick_x_max)
            self.pick_y_min_var.set(pick_y_min)
            self.pick_y_max_var.set(pick_y_max)
            self.safe_x_min_var.set(safe_x_min)
            self.safe_x_max_var.set(safe_x_max)
            self.safe_y_min_var.set(safe_y_min)
            self.safe_y_max_var.set(safe_y_max)
            self.safe_z_min_var.set(safe_z_min)
            self.safe_z_max_var.set(safe_z_max)
            self.return_x_var.set(return_x)
            self.return_y_var.set(return_y)
            self.return_z_var.set(return_z)
            self.offset_calib_travel_z_var.set(offset_calib_travel_z)
            self.offset_calib_target_z_var.set(offset_calib_target_z)
            self._save_config_values(
                ui_mode=self.ui_mode,
                ui_language=self.ui_language,
                overlay_label_mode=overlay_var.get(),
                quality_detection_method=detection_var.get(),
                ui_move_speed=int(self.move_speed_var.get()),
                ui_lower_speed=int(self.lower_speed_var.get()),
                robot_target_offset_x_mm=new_x,
                robot_target_offset_y_mm=new_y,
                robot_target_offset_z_mm=new_z,
                unstack_pick_touch_offset_mm=pick_touch_offset,
                robot_pick_x_min_mm=pick_x_min,
                robot_pick_x_max_mm=pick_x_max,
                robot_pick_y_min_mm=pick_y_min,
                robot_pick_y_max_mm=pick_y_max,
                robot_safe_x_min_mm=safe_x_min,
                robot_safe_x_max_mm=safe_x_max,
                robot_safe_y_min_mm=safe_y_min,
                robot_safe_y_max_mm=safe_y_max,
                robot_safe_z_min_mm=safe_z_min,
                robot_safe_z_max_mm=safe_z_max,
                robot_auto_x_min_mm=auto_x_min,
                robot_auto_x_max_mm=auto_x_max,
                robot_auto_y_min_mm=auto_y_min,
                robot_auto_y_max_mm=auto_y_max,
                robot_lift_x_min_mm=lift_x_min,
                robot_lift_x_max_mm=lift_x_max,
                robot_lift_y_min_mm=lift_y_min,
                robot_lift_y_max_mm=lift_y_max,
                robot_return_x_mm=return_x,
                robot_return_y_mm=return_y,
                robot_return_z_mm=return_z,
                offset_calib_travel_z_mm=offset_calib_travel_z,
                offset_calib_target_z_mm=offset_calib_target_z,
                **motion_values,
            )
            self.ui_config = self._load_config()
            self._apply_pick_offset_to_current_targets(new_x, new_y, old_x, old_y, new_z, old_z)
            win.destroy()
            self.title(self._t("title"))
            self._build_layout()
            self._load_targets()
            self._refresh_current_view()

        ttk.Button(buttons, text=self._t("cancel"), command=win.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text=self._t("save"), command=apply_settings).grid(row=0, column=1)

    def _on_image_click(self, event):
        if not self.targets_data.get("targets"):
            return
        view_xy = self._event_to_view_xy(event)
        if view_xy is None:
            return
        view_x, view_y = view_xy
        best_idx = self._nearest_target_index(view_x, view_y)
        if best_idx is not None:
            self.tree.selection_set(str(best_idx))
            self.tree.focus(str(best_idx))
            self._update_selected_text()

    def _nearest_target_index(self, view_x, view_y):
        cfg = self._load_config()
        view = self.camera_view.get()
        best = None
        best_d = 80.0 if view == "Quality" else 55.0
        for t in self.targets_data.get("targets", []):
            pos = self._target_view_xy(t, view, cfg)
            if pos is None:
                continue
            d = ((view_x - pos[0]) ** 2 + (view_y - pos[1]) ** 2) ** 0.5
            if d < best_d:
                best = int(t.get("index"))
                best_d = d
        return best

    def _fit_point(self, x, y, src_w, src_h, dst_w, dst_h):
        scale = min(dst_w / src_w, dst_h / src_h)
        off_x = (dst_w - src_w * scale) / 2.0
        off_y = (dst_h - src_h * scale) / 2.0
        return x * scale + off_x, y * scale + off_y

    def _target_view_xy(self, target, view, cfg):
        if view == "Quality":
            qx = target.get("quality_x_px")
            qy = target.get("quality_y_px")
            if qx is None or qy is None:
                return None
            if getattr(self, "_quality_overlay_cropped", False) or self._last_image_path == "quality-live":
                roi = cfg.get("quality_roi")
                if roi:
                    x1, y1, _x2, _y2 = [float(v) for v in roi]
                    return float(qx) - x1, float(qy) - y1
            return float(qx), float(qy)
        if view == "Gemini":
            gx = target.get("gemini_x_px")
            gy = target.get("gemini_y_px")
            if gx is None or gy is None:
                return None
            roi = cfg.get("gemini_display_roi")
            if roi:
                x1, y1, x2, y2 = [float(v) for v in roi]
                return self._fit_point(float(gx) - x1, float(gy) - y1, x2 - x1, y2 - y1, 960.0, 720.0)
            return self._fit_point(float(gx), float(gy), 1280.0, 720.0, 960.0, 720.0)
        return None

    def _on_select(self, _event=None):
        self._update_selected_text()

    def _current_selected_index(self):
        sel = self.tree.selection()
        if not sel:
            return None
        try:
            return int(sel[0])
        except Exception:
            return None

    def _selected_or_first_valid(self):
        selected = self._current_selected_index()
        targets = self.targets_data.get("targets", [])
        if selected is not None:
            for t in targets:
                if int(t.get("index", -1)) == selected:
                    return t
        for t in targets:
            if t.get("valid_for_pick"):
                return t
        return None

    def _update_selected_text(self):
        t = self._selected_or_first_valid()
        if not t:
            self.selected_var.set(self._t("no_target_selected"))
            return
        self.selected_var.set(
            f"Q{t.get('index')} {t.get('label_name')}  d={self._fmt(t.get('diameter_mm'), 'mm')}\n"
            f"depth={self._fmt(t.get('depth_z_mm'), 'mm')}  table+={self._fmt(t.get('height_above_table_mm'), 'mm')}\n"
            f"X={self._fmt(t.get('robot_x_mm'))}  Y={self._fmt(t.get('robot_y_mm'))}  Z={self._fmt(t.get('robot_z_mm'))}\n"
            f"{'OK' if t.get('valid_for_pick') else 'CHECK'} {t.get('pick_check_reason', '')}"
        )

    def _target_text(self, t):
        return (
            f"Q{t.get('index')} {t.get('label_name')}  d={self._fmt(t.get('diameter_mm'), 'mm')}\n"
            f"X={self._fmt(t.get('robot_x_mm'))}  Y={self._fmt(t.get('robot_y_mm'))}  Z={self._fmt(t.get('robot_z_mm'))}"
        )

    def _set_planned_target(self, title, target=None):
        if target is None:
            target = self._selected_or_first_valid()
        if target:
            text = f"即將作動：{title}\n本次目標：{self._target_text(target)}"
        else:
            text = f"即將作動：{title}\n本次目標：重新辨識後第一顆可取硬幣"
        self._planned_action_text = text
        self._planned_action_until = time.time() + 90.0
        self.action_var.set(text)

    def _confirm(self, title, message):
        return messagebox.askyesno(title, message, icon="warning")

    def _run_async(self, title, cmd, done_refresh=True, silent=False, pause_preview=True):
        if self.busy:
            if not silent:
                messagebox.showinfo("忙碌中", "上一個動作還在執行。")
            return
        if pause_preview:
            self._close_quality_preview()
        self._set_busy(True, title)
        if not silent:
            self._log(f"開始：{title}")

        def worker():
            error_message = None
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUTF8"] = "1"
                result = subprocess.run(
                    cmd,
                    cwd=str(HERE),
                    text=True,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    **SUBPROCESS_KW,
                )
                output = (result.stdout or "") + (result.stderr or "")
                if result.returncode == 0:
                    if not silent:
                        self.after(0, lambda: self._log(f"DONE: {title}"))
                        if output.strip():
                            self.after(0, lambda out=output[-1200:]: self._log(out))
                else:
                    error_message = output[-1200:] or str(result.returncode)
                    self.after(0, lambda msg=error_message: self._log(f"失敗：{title}\n{msg}"))
            finally:
                def finish(msg=error_message):
                    if msg is None:
                        self._planned_action_text = None
                        self._planned_action_until = 0.0
                    if done_refresh:
                        self._sync_pick_offset_from_config()
                        self._load_targets()
                    self._set_busy(False, self._t("ready"))
                    if done_refresh:
                        self._resume_live_preview()
                    elif pause_preview and self.camera_view.get() in ("Quality", "Combined"):
                        self._resume_live_preview()
                    self._run_pending_robot_action()
                    if msg:
                        self.after(150, lambda: messagebox.showerror("動作失敗", msg))

                self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _run_preview_async(self, view=None):
        if self.preview_busy:
            return
        self.preview_busy = True
        view = view or self.camera_view.get()
        active_view = self.camera_view.get()
        self._preview_request_times[view] = time.time()
        def worker():
            try:
                result = subprocess.run(
                    [str(PYTHON), "camera_preview_once.py", "--view", view],
                    cwd=str(HERE),
                    text=True,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    **SUBPROCESS_KW,
                )
                if result.returncode == 0:
                    self.after(0, lambda expected=active_view: self._refresh_current_view() if self.camera_view.get() == expected else None)
                else:
                    self.after(0, lambda: self._log("相機預覽失敗"))
            finally:
                self.preview_busy = False
        threading.Thread(target=worker, daemon=True).start()

    def _run_robot_action_when_ready(self, action, pause_preview=True, queue_when_busy=True):
        if pause_preview:
            self.auto_preview.set(False)
        if self.busy:
            if not queue_when_busy:
                return
            self.pending_robot_action = action
            self.status_var.set("等待相機預覽結束...")
            self._log("手臂動作已排隊，等待目前相機預覽結束。")
            return
        action()

    def _run_pending_robot_action(self):
        if self.pending_robot_action is None or self.busy:
            return
        action = self.pending_robot_action
        self.pending_robot_action = None
        self.after(450, action)

    def _move_start_pose(self):
        def action():
            x, y, z = self._robot_return_pose()
            if not self._confirm("移動手臂", f"要把手臂移到回去位置 X={x:.1f} Y={y:.1f} Z={z:.1f} 嗎？"):
                return
            cmd = [str(PYTHON), "hover_robot_target.py", "--start-only", "--yes"] + self._start_pose_args() + self._speed_args()
            self._run_async("回到相機避讓位置", cmd, done_refresh=False)
        self._run_robot_action_when_ready(action)

    def _emergency_stop(self):
        self.auto_preview.set(False)
        self.pending_robot_action = None
        cmd = [str(PYTHON), "robot_emergency_stop.py"]
        self.status_var.set("急停")
        self.status_lbl.configure(style="StatusWarn.TLabel")
        self._log("已送出急停：關閉 DO13 並停用手臂")

        def worker():
            result = subprocess.run(
                cmd,
                cwd=str(HERE),
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                **SUBPROCESS_KW,
            )
            output = (result.stdout or "") + (result.stderr or "")
            self.after(0, lambda: self._log(output[-1200:] if output.strip() else f"ESTOP returncode={result.returncode}"))
            self.after(0, lambda: self._set_busy(False, "手臂已停用" if result.returncode == 0 else "請檢查急停紀錄"))

        threading.Thread(target=worker, daemon=True).start()

    def _clear_enable_robot(self):
        self.auto_preview.set(False)
        cmd = [str(PYTHON), "robot_clear_enable.py"]
        self._run_async("清除報警並啟用手臂", cmd, done_refresh=False)

    def _select_roi(self):
        if not self._confirm("設定 ROI", "要開啟 ROI 選取工具嗎？\n只框選桌面硬幣區，排除手臂。"):
            return
        self.auto_preview.set(False)
        cmd = [str(PYTHON), "select_quality_roi.py"]
        self._run_async("設定辨識 ROI", cmd)

    def _refresh_vision(self, silent=False):
        if self._is_yolo_mode():
            self._toggle_yolo_live_detection()
            return
        cmd = [str(PYTHON), "dual_camera_live.py", "--save-once", "--fast", "--quality-only"]
        self._run_async("重新辨識", cmd, silent=silent)

    def _preview_only(self, view=None):
        self._run_preview_async(view=view)

    def _set_offset_jog_step(self, step):
        step = max(0.1, min(10.0, float(step)))
        self.offset_jog_step_var.set(step)
        self._save_config_values(offset_calib_jog_step_mm=step)
        if hasattr(self, "offset_jog_step_label"):
            self.offset_jog_step_label.set(f"步距 {step:.1f}mm")

    def _make_offset_jog_button(self, parent, text, dx, dy, dz):
        btn = ttk.Button(parent, text=text)
        btn.bind("<ButtonPress-1>", lambda _e: self._start_offset_hold(dx, dy, dz))
        btn.bind("<ButtonRelease-1>", lambda _e: self._stop_offset_hold())
        btn.bind("<Leave>", lambda _e: self._stop_offset_hold())
        return btn

    def _start_offset_hold(self, dx, dy, dz):
        self._stop_offset_hold()
        self._offset_hold_vector = (dx, dy, dz)
        self._offset_fast_jog(dx, dy, dz)
        self._offset_hold_after = self.after(260, self._offset_hold_tick)

    def _offset_hold_tick(self):
        vector = self._offset_hold_vector
        if vector is None:
            self._offset_hold_after = None
            return
        self._offset_fast_jog(*vector)
        self._offset_hold_after = self.after(180, self._offset_hold_tick)

    def _stop_offset_hold(self):
        self._offset_hold_vector = None
        if self._offset_hold_after is not None:
            try:
                self.after_cancel(self._offset_hold_after)
            except Exception:
                pass
            self._offset_hold_after = None

    def _close_offset_robot(self):
        with self._offset_robot_lock:
            robot = self._offset_robot
            self._offset_robot = None
            self._offset_jog_pose = None
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass

    def _offset_check_xyz(self, x, y, z):
        bounds = self._robot_safe_bounds()
        if not (bounds["x_min"] <= x <= bounds["x_max"]):
            raise RuntimeError(f"X={x:.2f} 超出安全範圍 {bounds['x_min']}..{bounds['x_max']}")
        if not (bounds["y_min"] <= y <= bounds["y_max"]):
            raise RuntimeError(f"Y={y:.2f} 超出安全範圍 {bounds['y_min']}..{bounds['y_max']}")
        if not (bounds["z_min"] <= z <= bounds["z_max"]):
            raise RuntimeError(f"Z={z:.2f} 超出安全範圍 {bounds['z_min']}..{bounds['z_max']}")

    def _offset_fast_jog(self, dx, dy, dz):
        if self._offset_jog_busy or self.busy:
            return
        self.auto_preview.set(False)
        step = max(0.1, min(10.0, float(self.offset_jog_step_var.get())))
        self._offset_jog_busy = True

        def worker():
            error = None
            try:
                with self._offset_robot_lock:
                    if self._offset_robot is None:
                        self._offset_robot = MG400(timeout=1.2)
                        self._offset_robot.connect()
                        self._offset_robot.enable()
                    robot = self._offset_robot
                    robot.set_speed(max(1, min(45, int(self.lower_speed_var.get()))))
                    pose = self._offset_jog_pose
                    if pose is None:
                        pose = robot.get_pose()
                    if pose is None:
                        raise RuntimeError("讀取目前手臂座標失敗。")
                    x, y, z, r = pose
                    nx = float(x) + float(dx) * step
                    ny = float(y) + float(dy) * step
                    nz = float(z) + float(dz) * step
                    self._offset_check_xyz(nx, ny, nz)
                    resp = robot._send(robot._move, f"MovL({nx:.3f},{ny:.3f},{nz:.3f},{float(r):.3f})")
                    robot.last_response = resp
                    if not resp.startswith("0"):
                        raise RuntimeError(f"MovL 被拒：{resp}")
                    self._offset_jog_pose = (nx, ny, nz, float(r))
            except Exception as exc:
                error = str(exc)
                self._close_offset_robot()
            finally:
                def finish(msg=error):
                    self._offset_jog_busy = False
                    if msg:
                        self._stop_offset_hold()
                        if not self._offset_jog_warned:
                            self._offset_jog_warned = True
                            self._log(f"微調停止：{msg}")
                            messagebox.showerror("微調停止", msg)
                    else:
                        self._offset_jog_warned = False
                        self.status_var.set("微調中" if self._offset_hold_vector else self._t("ready"))

                self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _offset_calib_index_arg(self):
        t = self._selected_or_first_valid()
        if not t:
            messagebox.showinfo("沒有目標", "目前沒有可校正目標，請先重新辨識。")
            return None, None
        return int(t["index"]), t

    def _offset_calib_start(self):
        def action():
            self._close_offset_robot()
            idx, target = self._offset_calib_index_arg()
            if idx is None:
                return
            target_z = float(self.offset_calib_target_z_var.get())
            if not self._confirm("開始誤差校正", f"手臂會直接使用目前鎖定座標移到 Q{idx} 的校正高度 Z={target_z:.0f}，不會重新辨識。\n請確認路徑安全。"):
                return
            self._set_planned_target("誤差校正：移到硬幣上方", target)
            self._offset_jog_pose = (
                float(target["robot_x_mm"]),
                float(target["robot_y_mm"]),
                target_z,
                0.0,
            )
            cmd = [
                str(PYTHON),
                "robot_offset_calibration.py",
                "start",
                "--index",
                str(idx),
                "--safe-z",
                str(target_z),
                "--pre-xy-lift",
                "20",
                "--speed",
                str(int(self.move_speed_var.get())),
            ]
            self._run_async(f"誤差校正移到 Q{idx} 上方", cmd, done_refresh=False, pause_preview=False)
        self._run_robot_action_when_ready(action, pause_preview=False)

    def _offset_calib_jog(self, dx, dy, dz, queue_when_busy=True):
        def action():
            step = max(0.1, min(10.0, float(self.offset_jog_step_var.get())))
            cmd = [
                str(PYTHON),
                "robot_offset_calibration.py",
                "jog",
                "--dx",
                str(float(dx)),
                "--dy",
                str(float(dy)),
                "--dz",
                str(float(dz)),
                "--step",
                str(step),
                "--speed",
                str(max(1, min(30, int(self.lower_speed_var.get())))),
            ]
            self._run_async(f"誤差校正微調 {step:.1f}mm", cmd, done_refresh=False, silent=True, pause_preview=False)
        self._run_robot_action_when_ready(action, pause_preview=False, queue_when_busy=queue_when_busy)

    def _offset_calib_teach(self, replace):
        def action():
            pose_for_teach = self._offset_jog_pose
            self._close_offset_robot()
            idx, target = self._offset_calib_index_arg()
            if idx is None:
                return
            prompt = "會用目前手臂 X/Y 當作硬幣真實中心，計算新的平均夾取偏移；不會重新辨識。"
            if replace:
                prompt += "\n這會先清空舊樣本，只保留本次教點。"
            if not self._confirm("儲存誤差校正", prompt):
                return
            self._set_planned_target("誤差校正：教點", target)
            cmd = [str(PYTHON), "robot_offset_calibration.py", "teach", "--index", str(idx)]
            if replace:
                cmd.append("--replace")
            if pose_for_teach is not None:
                cmd.extend([
                    "--taught-x", str(float(pose_for_teach[0])),
                    "--taught-y", str(float(pose_for_teach[1])),
                    "--taught-z", str(float(pose_for_teach[2])),
                ])
            self._run_async("儲存誤差校正樣本", cmd, done_refresh=True, pause_preview=False)
        self._run_robot_action_when_ready(action, pause_preview=False)

    def _offset_calib_reset(self):
        def action():
            self._close_offset_robot()
            if not self._confirm("清空公差樣本", "要清空所有誤差樣本，並把夾取偏移歸零嗎？"):
                return
            cmd = [str(PYTHON), "robot_offset_calibration.py", "reset"]
            self._run_async("清空誤差校正樣本", cmd, done_refresh=True, pause_preview=False)
        self._run_robot_action_when_ready(action, pause_preview=False)

    def _hover_selected(self):
        def action():
            t = self._selected_or_first_valid()
            if not t:
                messagebox.showinfo("沒有目標", "目前沒有可取目標。")
                return
            idx = int(t["index"])
            if not self._confirm("移到硬幣上方", f"手臂會先回避相機、重新辨識，然後移到 Q{idx} 上方 Z=100。\n不開真空/DO。"):
                return
            self._set_planned_target("移到硬幣上方", t)
            cmd = [str(PYTHON), "hover_robot_target.py", "--index", str(idx), "--fallback-first-valid", "--safe-z", "100", "--refresh-after-start", "--refresh-max-age-sec", "60", "--skip-start-if-close", "--yes"] + self._start_pose_args() + self._speed_args()
            self._run_async(f"辨識後移到 Q{idx} 上方", cmd)
        self._run_robot_action_when_ready(action)

    def _hover_first_cycle(self):
        def action():
            if not self._confirm("辨識後移到上方", "手臂會先回避相機、自動辨識，然後移到第一顆可取硬幣上方 Z=100。\n不開真空/DO。"):
                return
            self._set_planned_target("移到第一顆可取硬幣上方")
            cmd = [str(PYTHON), "hover_robot_target.py", "--safe-z", "100", "--refresh-after-start", "--refresh-max-age-sec", "60", "--skip-start-if-close", "--yes"] + self._start_pose_args() + self._speed_args()
            self._run_async("辨識後移到第一顆上方", cmd)
        self._run_robot_action_when_ready(action)

    def _dry_lower_selected(self):
        def action():
            t = self._selected_or_first_valid()
            if not t:
                messagebox.showinfo("沒有目標", "目前沒有可取目標。")
                return
            idx = int(t["index"])
            if not self._confirm("下降到硬幣", f"手臂會先回避相機、重新辨識，然後下降 Q{idx} 到教點/深度計算 Z。\n沒有 Z 教點時會退回 Z=-156。\n不開真空/DO。"):
                return
            self._set_planned_target("下降到硬幣", t)
            cmd = [str(PYTHON), "hover_robot_target.py", "--index", str(idx), "--fallback-first-valid", "--safe-z", "100", "--lower-z", "-156", "--use-target-lower-z", "--refresh-after-start", "--refresh-max-age-sec", "60", "--skip-start-if-close", "--yes"] + self._start_pose_args() + self._speed_args()
            self._run_async(f"辨識後下降 Q{idx}", cmd)
        self._run_robot_action_when_ready(action)

    def _safe_cycle(self):
        def action():
            if not self._confirm("辨識後下降", "要執行：回避相機 -> 自動辨識 -> 第一顆可取硬幣 -> 下降到教點/深度計算 Z 嗎？\n沒有 Z 教點時會退回 Z=-156。\n不開真空/DO。"):
                return
            self._set_planned_target("下降第一顆可取硬幣")
            cmd = [str(PYTHON), "hover_robot_target.py", "--safe-z", "100", "--lower-z", "-156", "--use-target-lower-z", "--refresh-after-start", "--refresh-max-age-sec", "60", "--skip-start-if-close", "--yes"] + self._start_pose_args() + self._speed_args()
            self._run_async("辨識後下降第一顆", cmd)
        self._run_robot_action_when_ready(action)

    def _unstack_coins(self):
        def action():
            if not self._confirm(
                "拆堆到暫放區",
                "會一直循環真實吸取，直到辨識不到需要拆開的堆疊硬幣：重新辨識上層硬幣 -> 下降到硬幣位置 -> 開啟吸盤 DO13 -> 抬起並保持吸盤開啟 -> 移到自動找到的空位 -> 下降放置並關閉 DO13。"
            ):
                return
            self._set_planned_target("拆堆到暫放區")
            cmd = [
                str(PYTHON),
                "unstack_coins.py",
                "--yes",
                "--real-pick",
                "--max-picks", "60",
                "--move-speed", str(int(self.move_speed_var.get())),
                "--lower-speed", str(int(self.lower_speed_var.get())),
            ] + self._start_pose_args() + self._speed_args()[4:]
            self._run_async("拆堆到暫放區", cmd)
        self._run_robot_action_when_ready(action)

    def _dry_lower_all(self):
        def action():
            count = sum(1 for t in self.targets_data.get("targets", []) if t.get("valid_for_pick"))
            if count <= 0:
                messagebox.showinfo("沒有目標", "目前沒有可取目標。")
                return
            if not self._confirm("全部下降測試", f"要依序測試 {count} 顆可取硬幣嗎？\n會先回避相機、重新辨識，再逐顆下降到教點/深度計算 Z。\n沒有 Z 教點時會退回 Z=-156。\n不開真空/DO。"):
                return
            self._set_planned_target(f"逐顆下降 {count} 顆")
            cmd = [str(PYTHON), "hover_robot_target.py", "--all", "--include-non-top-pickable", "--safe-z", "100", "--travel-z", "100", "--lower-z", "-156", "--use-target-lower-z", "--refresh-after-start", "--refresh-max-age-sec", "60", "--skip-start-if-close", "--yes"] + self._start_pose_args() + self._speed_args()
            self._run_async("辨識後逐顆下降", cmd)
        self._run_robot_action_when_ready(action)


if __name__ == "__main__":
    app = CoinRobotUI()
    app.mainloop()
