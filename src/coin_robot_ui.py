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
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk


HERE = Path(__file__).parent
TARGETS_FILE = HERE / "robot_targets.json"
ACTION_STATUS_FILE = HERE / "robot_action_status.json"
OUT_DIR = HERE / "test_output"
CONFIG_FILE = HERE / "dual_camera_config.json"
PYTHON = Path(r"C:\Users\user\miniconda3\envs\coin\python.exe")

ROBOT_ERROR_HINTS = {
    23: {
        "title": "路徑被拒絕 / 動作中斷",
        "cause": "控制器停止了移動指令。常見原因是直線路徑不可達、接近關節/軟體極限、碰撞偵測觸發，或路徑不安全。",
        "action": "檢查障礙物，必要時提高轉移高度、降低速度，或先手動把手臂移到安全位置再重試。",
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
        "title": "低位下降報警",
        "cause": "下降高度太低、X/Y 仍有偏差、碰撞偵測觸發，或目標太靠近可靠範圍邊界。",
        "action": "先停止低位測試，清除報警，確認高空對準後，再用中間硬幣重試。",
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
        "lower_first": "辨識後下降第一顆到 Z=-156",
        "set_roi": "設定辨識 ROI",
        "refresh": "重新辨識 / 鎖定座標",
        "hover_selected": "移到選取硬幣上方",
        "lower_selected": "下降選取硬幣到 Z=-156",
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
        "lower_first": "Detect then Lower First OK to Z=-156",
        "set_roi": "Set Detection ROI",
        "refresh": "Refresh / Lock Vision",
        "hover_selected": "Hover Selected Coin",
        "lower_selected": "Lower Selected Coin to Z=-156",
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
        if self.ui_mode not in ("operator", "engineer"):
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
        self._last_image_path = None
        self._display_image_size = (0, 0)
        self._display_image_offset = (0, 0)
        self._loaded_image_size = (0, 0)
        self._quality_cap = None
        self._quality_thread = None
        self._quality_stop = threading.Event()
        self._quality_lock = threading.Lock()
        self._quality_latest_frame = None
        self._quality_latest_id = 0
        self._quality_displayed_id = 0
        self._quality_preview_error_logged = False
        self._preview_request_times = {}
        self._gemini_preview_pil = None
        self._gemini_preview_mtime = 0.0
        self._last_gemini_preview_request = 0.0
        self.zoom = 1.0
        self.auto_preview = tk.BooleanVar(value=True)
        self.move_speed_var = tk.IntVar(value=int(self.ui_config.get("ui_move_speed", 40)))
        self.lower_speed_var = tk.IntVar(value=int(self.ui_config.get("ui_lower_speed", 25)))

        self._build_styles()
        self._build_layout()
        self._reset_stale_action_status()
        self._load_targets()
        if self.camera_view.get() == "Quality":
            self._start_quality_preview_thread()
        else:
            self._load_latest_image()
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

        ttk.Separator(side).grid(row=6, column=0, sticky="ew", pady=12)
        ttk.Label(side, text=self._t("robot_controls"), style="Panel.TLabel", font=("Segoe UI", 12, "bold")).grid(row=7, column=0, sticky="w")
        ttk.Button(side, text=self._t("estop"), style="Danger.TButton", command=self._emergency_stop).grid(row=8, column=0, sticky="ew", pady=(10, 8), ipady=5)
        self.move_speed_label = tk.StringVar(value=f"{self._t('move_speed')} {int(self.move_speed_var.get())}%")
        self.lower_speed_label = tk.StringVar(value=f"{self._t('lower_speed')} {int(self.lower_speed_var.get())}%")
        ttk.Button(side, text=self._t("clear_enable"), command=self._clear_enable_robot).grid(row=10, column=0, sticky="ew", pady=(10, 4))
        ttk.Button(side, text=self._t("move_clear"), command=self._move_start_pose).grid(row=11, column=0, sticky="ew", pady=(10, 4))
        ttk.Button(side, text=self._t("hover_first"), command=self._hover_first_cycle).grid(row=12, column=0, sticky="ew", pady=(12, 4))
        ttk.Button(side, text=self._t("lower_first"), style="Danger.TButton", command=self._safe_cycle).grid(row=13, column=0, sticky="ew", pady=4)
        if self.ui_mode == "engineer":
            ttk.Separator(side).grid(row=14, column=0, sticky="ew", pady=10)
            ttk.Button(side, text=self._t("set_roi"), command=self._select_roi).grid(row=15, column=0, sticky="ew", pady=4)
            ttk.Button(side, text=self._t("refresh"), command=self._refresh_vision).grid(row=16, column=0, sticky="ew", pady=4)
            ttk.Button(side, text=self._t("hover_selected"), command=self._hover_selected).grid(row=17, column=0, sticky="ew", pady=4)
            ttk.Button(side, text=self._t("lower_selected"), style="Danger.TButton", command=self._dry_lower_selected).grid(row=18, column=0, sticky="ew", pady=4)
            ttk.Button(side, text=self._t("lower_all"), style="Danger.TButton", command=self._dry_lower_all).grid(row=19, column=0, sticky="ew", pady=4)

        ttk.Separator(side).grid(row=18, column=0, sticky="ew", pady=12)
        ttk.Label(side, text=self._t("current_target"), style="Panel.TLabel", font=("Segoe UI", 12, "bold")).grid(row=20, column=0, sticky="w")
        self.selected_var = tk.StringVar(value=self._t("no_target_selected"))
        ttk.Label(side, textvariable=self.selected_var, style="Panel.TLabel", wraplength=280, justify="left").grid(row=20, column=0, sticky="ew", pady=(8, 0))
        self.action_var = tk.StringVar(value=self._t("current_action_idle"))
        ttk.Label(side, textvariable=self.action_var, style="Panel.TLabel", wraplength=280, justify="left").grid(row=21, column=0, sticky="ew", pady=(10, 0))
        self.note_var = tk.StringVar(value=self._t("dry_only"))
        ttk.Label(side, textvariable=self.note_var, style="Panel.TLabel", wraplength=280, justify="left").grid(row=22, column=0, sticky="ew", pady=(12, 0))

        table_panel = ttk.Frame(main, style="Panel.TFrame", padding=10)
        table_panel.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        table_panel.columnconfigure(0, weight=1)
        ttk.Label(table_panel, text=self._t("coin_targets"), style="Panel.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        cols = ("index", "label", "diam", "x", "y", "z", "valid")
        self.tree = ttk.Treeview(table_panel, columns=cols, show="headings", height=7)
        headings = {
            "index": self._t("index"),
            "label": self._t("class"),
            "diam": self._t("diameter"),
            "x": "MG400 X",
            "y": "MG400 Y",
            "z": "MG400 Z",
            "valid": self._t("status"),
        }
        widths = {"index": 52, "label": 90, "diam": 100, "x": 110, "y": 110, "z": 100, "valid": 80}
        for col in cols:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], minwidth=widths[col], anchor="center", stretch=True)
        yscroll = ttk.Scrollbar(table_panel, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.grid(row=1, column=0, sticky="ew")
        yscroll.grid(row=1, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        self.log = tk.Text(top, height=4, wrap="word", bg="#181b1e", fg="#c9d1d9", insertbackground="#c9d1d9", relief="flat")
        self.log.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self._log(self._t("vision_note"))

    def _tick(self):
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
        ]

    def _log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _load_targets(self):
        if not TARGETS_FILE.exists():
            return
        try:
            self.targets_data = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        counts = self.targets_data.get("counts", {})
        for name in ("1NT", "5NT", "10NT", "50NT"):
            self.count_vars[name].set(f"{name} x {counts.get(name, 0)}")
        self.total_var.set(f"{self.targets_data.get('total_value_nt', 0)} NT")
        targets = self.targets_data.get("targets", [])
        valid_count = sum(1 for t in targets if t.get("valid_for_pick"))
        self.valid_var.set(str(valid_count))
        selected = self._current_selected_index()
        self.tree.delete(*self.tree.get_children())
        for t in targets:
            idx = int(t.get("index", 0))
            iid = str(idx)
            values = (
                idx,
                t.get("label_name", "?"),
                self._fmt(t.get("diameter_mm"), "mm"),
                self._fmt(t.get("robot_x_mm")),
                self._fmt(t.get("robot_y_mm")),
                self._fmt(t.get("robot_z_mm")),
                self._t("ok") if t.get("valid_for_pick") else self._t("check"),
            )
            self.tree.insert("", "end", iid=iid, values=values)
        if selected and self.tree.exists(str(selected)):
            self.tree.selection_set(str(selected))
            self.tree.focus(str(selected))
        self._update_selected_text()

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
        if self.camera_view.get() == "Quality":
            preview = OUT_DIR / "live_preview_quality.jpg"
            if preview.exists():
                return preview
        if self.camera_view.get() == "Gemini":
            preview = OUT_DIR / "live_preview_gemini.jpg"
            min_time = self._preview_request_times.get("Gemini", 0.0)
            if preview.exists() and preview.stat().st_mtime >= min_time:
                return preview
            return None
        if self.camera_view.get() == "Combined":
            preview = OUT_DIR / "live_preview_combined.jpg"
            min_time = self._preview_request_times.get("Combined", 0.0)
            if preview.exists() and preview.stat().st_mtime >= min_time:
                return preview
            return None
        prefix = {
            "Gemini": "gemini_view_*.jpg",
            "Quality": "quality_view_*.jpg",
            "Combined": "dual_camera_snapshot_*.jpg",
        }.get(self.camera_view.get(), "gemini_view_*.jpg")
        files = sorted(OUT_DIR.glob(prefix), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files and prefix != "dual_camera_snapshot_*.jpg":
            files = sorted(OUT_DIR.glob("dual_camera_snapshot_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
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
            left = max(0, (new_w - box_w) // 2)
            top = max(0, (new_h - box_h) // 2)
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

    def _resize_pil_to_height(self, img, height):
        if img.height <= 0:
            return img
        width = max(1, int(img.width * (height / img.height)))
        return img.resize((width, height), Image.Resampling.LANCZOS)

    def _load_latest_image(self, force=False):
        path = self._latest_snapshot()
        if path is None:
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
        cap = cv2.VideoCapture(int(cfg.get("quality_camera_index", 0)), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.get("quality_width", 1280)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.get("quality_height", 720)))
        cap.set(cv2.CAP_PROP_FPS, int(cfg.get("quality_fps", 30)))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return None
        self._quality_cap = cap
        self._quality_preview_error_logged = False
        return cap

    def _close_quality_preview(self):
        self._quality_stop.set()
        thread = self._quality_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.8)
        self._quality_thread = None
        if self._quality_cap is not None:
            try:
                self._quality_cap.release()
            except Exception:
                pass
            self._quality_cap = None
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
            cap = self._open_quality_preview()
            if cap is None:
                if not self._quality_preview_error_logged:
                    self.after(0, lambda: self._log("畫質相機開啟失敗"))
                    self._quality_preview_error_logged = True
                return
            while not self._quality_stop.is_set():
                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._quality_lock:
                        self._quality_latest_frame = frame
                        self._quality_latest_id += 1
                time.sleep(0.001)

        self._quality_thread = threading.Thread(target=worker, daemon=True)
        self._quality_thread.start()

    def _update_quality_live_frame(self):
        with self._quality_lock:
            if self._quality_latest_frame is None or self._quality_latest_id == self._quality_displayed_id:
                return
            frame = self._quality_latest_frame.copy()
            self._quality_displayed_id = self._quality_latest_id
        frame = self._crop_quality_roi(frame)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._display_pil_image(img, "quality-live")

    def _latest_quality_frame_copy(self):
        with self._quality_lock:
            if self._quality_latest_frame is None:
                return None, self._quality_latest_id
            return self._quality_latest_frame.copy(), self._quality_latest_id

    def _latest_gemini_preview_pil(self):
        path = OUT_DIR / "live_preview_gemini.jpg"
        min_time = self._preview_request_times.get("Gemini", 0.0)
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
        self._close_quality_preview()
        self.destroy()

    def _refresh_current_view(self):
        if self.camera_view.get() == "Quality":
            self._update_quality_live_frame()
        elif self.camera_view.get() == "Combined":
            self._update_combined_live_frame()
        else:
            self._load_latest_image(force=True)

    def _on_image_wheel(self, event):
        if event.delta > 0:
            self.zoom = min(3.0, self.zoom * 1.15)
        else:
            self.zoom = max(1.0, self.zoom / 1.15)
        self._refresh_current_view()

    def _load_action_status(self):
        if not ACTION_STATUS_FILE.exists():
            return
        try:
            data = json.loads(ACTION_STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        t = data.get("target") or {}
        if t:
            self.action_var.set(
                f"目前動作：{self._state_text(data.get('state', '-'))}\n"
                f"Q{t.get('index')} {t.get('label_name', '?')}  "
                f"X={self._fmt(t.get('robot_x_mm'))} Y={self._fmt(t.get('robot_y_mm'))}"
            )
        else:
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

    def _open_settings(self):
        win = tk.Toplevel(self)
        win.title(self._t("settings"))
        win.configure(bg="#2b2f33")
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        mode_var = tk.StringVar(value=self.ui_mode)
        lang_var = tk.StringVar(value=self.ui_language)
        move_var = tk.IntVar(value=int(self.move_speed_var.get()))
        lower_var = tk.IntVar(value=int(self.lower_speed_var.get()))

        frame = ttk.Frame(win, style="Panel.TFrame", padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=self._t("mode"), style="Panel.TLabel").grid(row=0, column=0, sticky="w", pady=6)
        mode_box = ttk.Frame(frame, style="Panel.TFrame")
        mode_box.grid(row=0, column=1, sticky="w", pady=6)
        ttk.Radiobutton(mode_box, text=self._t("operator"), value="operator", variable=mode_var).grid(row=0, column=0, padx=(0, 12))
        ttk.Radiobutton(mode_box, text=self._t("engineer"), value="engineer", variable=mode_var).grid(row=0, column=1)

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

        def update_speed_labels(_=None):
            move_label.configure(text=f"{int(move_var.get())}%")
            lower_label.configure(text=f"{int(lower_var.get())}%")

        move_scale.configure(command=update_speed_labels)
        lower_scale.configure(command=update_speed_labels)

        buttons = ttk.Frame(frame, style="Panel.TFrame")
        buttons.grid(row=4, column=0, columnspan=3, sticky="e", pady=(14, 0))

        def apply_settings():
            self.ui_mode = mode_var.get()
            self.ui_language = lang_var.get()
            self.move_speed_var.set(int(move_var.get()))
            self.lower_speed_var.set(int(lower_var.get()))
            self._save_config_values(
                ui_mode=self.ui_mode,
                ui_language=self.ui_language,
                ui_move_speed=int(self.move_speed_var.get()),
                ui_lower_speed=int(self.lower_speed_var.get()),
            )
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
        off_x, off_y = self._display_image_offset
        disp_w, disp_h = self._display_image_size
        img_w, img_h = self._loaded_image_size
        if disp_w <= 0 or disp_h <= 0 or img_w <= 0 or img_h <= 0:
            return
        x = event.x - off_x
        y = event.y - off_y
        if x < 0 or y < 0 or x >= disp_w or y >= disp_h:
            return
        crop_x, crop_y = getattr(self, "_crop_offset_scaled", (0, 0))
        scale = max(getattr(self, "_image_scale", 1.0), 1e-6)
        view_x = (x + crop_x) / scale
        view_y = (y + crop_y) / scale
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
            f"X={self._fmt(t.get('robot_x_mm'))}  Y={self._fmt(t.get('robot_y_mm'))}  Z={self._fmt(t.get('robot_z_mm'))}"
        )

    def _confirm(self, title, message):
        return messagebox.askyesno(title, message, icon="warning")

    def _run_async(self, title, cmd, done_refresh=True, silent=False):
        if self.busy:
            if not silent:
                messagebox.showinfo("忙碌中", "上一個動作還在執行。")
            return
        self._close_quality_preview()
        self._set_busy(True, title)
        if not silent:
            self._log(f"開始：{title}")

        def worker():
            try:
                result = subprocess.run(cmd, cwd=str(HERE), text=True, capture_output=True, encoding="utf-8", errors="replace")
                output = (result.stdout or "") + (result.stderr or "")
                if result.returncode == 0:
                    if not silent:
                        self.after(0, lambda: self._log(f"DONE: {title}"))
                        if output.strip():
                            self.after(0, lambda out=output[-1200:]: self._log(out))
                else:
                    self.after(0, lambda: self._log(f"失敗：{title}\n{output[-1200:]}"))
                    self.after(0, lambda: messagebox.showerror("動作失敗", output[-1200:] or str(result.returncode)))
            finally:
                if done_refresh:
                    self.after(0, self._load_targets)
                    self.after(0, self._refresh_current_view)
                self.after(0, lambda: self._set_busy(False, self._t("ready")))
                self.after(0, self._run_pending_robot_action)

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
                result = subprocess.run([str(PYTHON), "camera_preview_once.py", "--view", view], cwd=str(HERE), text=True, capture_output=True, encoding="utf-8", errors="replace")
                if result.returncode == 0:
                    self.after(0, lambda expected=active_view: self._refresh_current_view() if self.camera_view.get() == expected else None)
                else:
                    self.after(0, lambda: self._log("相機預覽失敗"))
            finally:
                self.preview_busy = False
        threading.Thread(target=worker, daemon=True).start()

    def _run_robot_action_when_ready(self, action):
        self.auto_preview.set(False)
        if self.busy:
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
            if not self._confirm("移動手臂", "要把手臂移到相機避讓位置 X=30 Y=280 Z=150 嗎？"):
                return
            cmd = [str(PYTHON), "hover_robot_target.py", "--start-only", "--yes"] + self._speed_args()
            self._run_async("回到相機避讓位置", cmd, done_refresh=False)
        self._run_robot_action_when_ready(action)

    def _emergency_stop(self):
        self.auto_preview.set(False)
        self.pending_robot_action = None
        cmd = [str(PYTHON), "robot_emergency_stop.py"]
        self.status_var.set("急停")
        self.status_lbl.configure(style="StatusWarn.TLabel")
        self._log("已送出急停/停用手臂")

        def worker():
            result = subprocess.run(cmd, cwd=str(HERE), text=True, capture_output=True, encoding="utf-8", errors="replace")
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
        cmd = [str(PYTHON), "dual_camera_live.py", "--save-once", "--fast", "--quality-only"]
        self._run_async("重新辨識", cmd, silent=silent)

    def _preview_only(self, view=None):
        self._run_preview_async(view=view)

    def _hover_selected(self):
        def action():
            t = self._selected_or_first_valid()
            if not t:
                messagebox.showinfo("沒有目標", "目前沒有可取目標。")
                return
            idx = int(t["index"])
            if not self._confirm("移到硬幣上方", f"手臂會先回避相機、重新辨識，然後移到 Q{idx} 上方 Z=100。\n不開真空/DO。"):
                return
            cmd = [str(PYTHON), "hover_robot_target.py", "--index", str(idx), "--safe-z", "100", "--refresh-after-start", "--skip-start-if-close", "--yes"] + self._speed_args()
            self._run_async(f"辨識後移到 Q{idx} 上方", cmd)
        self._run_robot_action_when_ready(action)

    def _hover_first_cycle(self):
        def action():
            if not self._confirm("辨識後移到上方", "手臂會先回避相機、自動辨識，然後移到第一顆可取硬幣上方 Z=100。\n不開真空/DO。"):
                return
            cmd = [str(PYTHON), "hover_robot_target.py", "--safe-z", "100", "--refresh-after-start", "--skip-start-if-close", "--yes"] + self._speed_args()
            self._run_async("辨識後移到第一顆上方", cmd)
        self._run_robot_action_when_ready(action)

    def _dry_lower_selected(self):
        def action():
            t = self._selected_or_first_valid()
            if not t:
                messagebox.showinfo("沒有目標", "目前沒有可取目標。")
                return
            idx = int(t["index"])
            if not self._confirm("下降到硬幣", f"手臂會先回避相機、重新辨識，然後下降 Q{idx} 到 Z=-156。\n不開真空/DO。"):
                return
            cmd = [str(PYTHON), "hover_robot_target.py", "--index", str(idx), "--safe-z", "100", "--lower-z", "-156", "--refresh-after-start", "--skip-start-if-close", "--yes"] + self._speed_args()
            self._run_async(f"辨識後下降 Q{idx}", cmd)
        self._run_robot_action_when_ready(action)

    def _safe_cycle(self):
        def action():
            if not self._confirm("辨識後下降", "要執行：回避相機 -> 自動辨識 -> 第一顆可取硬幣 -> 下降到 Z=-156 嗎？\n不開真空/DO。"):
                return
            cmd = [str(PYTHON), "hover_robot_target.py", "--safe-z", "100", "--lower-z", "-156", "--refresh-after-start", "--skip-start-if-close", "--yes"] + self._speed_args()
            self._run_async("辨識後下降第一顆", cmd)
        self._run_robot_action_when_ready(action)

    def _dry_lower_all(self):
        def action():
            count = sum(1 for t in self.targets_data.get("targets", []) if t.get("valid_for_pick"))
            if count <= 0:
                messagebox.showinfo("沒有目標", "目前沒有可取目標。")
                return
            if not self._confirm("全部下降測試", f"要依序測試 {count} 顆可取硬幣嗎？\n會先回避相機、重新辨識，再逐顆下降到 Z=-156。\n不開真空/DO。"):
                return
            cmd = [str(PYTHON), "hover_robot_target.py", "--all", "--safe-z", "100", "--lower-z", "-156", "--refresh-after-start", "--skip-start-if-close", "--yes"] + self._speed_args()
            self._run_async("辨識後逐顆下降", cmd)
        self._run_robot_action_when_ready(action)


if __name__ == "__main__":
    app = CoinRobotUI()
    app.mainloop()
