# -*- coding: utf-8 -*-
"""Simple GUI editor for CoinVision auto-generated YOLO labels."""

import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

import yolo_label_tool as ytool


HERE = Path(__file__).parent
DATASET_DIR = HERE / "yolo_dataset"


class YoloLabelEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO 標註清理")
        self.geometry("1180x760")
        self.configure(bg="#22262a")

        self.split_var = tk.StringVar(value="train")
        self.sample_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="選一張圖片後，可點框或輸入編號刪除。")
        self.selected_index = None
        self.samples = []
        self.lines = []
        self.image_bgr = None
        self.preview_photo = None
        self.display_size = (1, 1)
        self.display_offset = (0, 0)
        self.display_scale = 1.0

        self._build_ui()
        self._load_samples()

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#22262a")
        style.configure("Dark.TLabel", background="#22262a", foreground="#f0f0f0")
        style.configure("Dark.TButton", padding=6)
        style.configure("Danger.TButton", padding=6, background="#8f2f2f", foreground="#ffffff")

        root = ttk.Frame(self, style="Dark.TFrame", padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        top = ttk.Frame(root, style="Dark.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="YOLO 標註清理", style="Dark.TLabel", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Button(top, text="重新整理", command=self._load_samples).pack(side="right", padx=4)
        ttk.Button(top, text="開啟資料夾", command=self._open_folder).pack(side="right", padx=4)

        left = ttk.Frame(root, style="Dark.TFrame", width=270)
        left.grid(row=1, column=0, sticky="ns", padx=(0, 10))
        left.grid_propagate(False)
        ttk.Label(left, text="資料集圖片", style="Dark.TLabel").pack(anchor="w")
        self.sample_list = tk.Listbox(left, bg="#171a1d", fg="#f0f0f0", selectbackground="#3b6ea8", height=28)
        self.sample_list.pack(fill="both", expand=True, pady=(6, 8))
        self.sample_list.bind("<<ListboxSelect>>", lambda _e: self._select_from_list())

        control = ttk.Frame(left, style="Dark.TFrame")
        control.pack(fill="x")
        ttk.Label(control, text="刪除編號", style="Dark.TLabel").grid(row=0, column=0, sticky="w")
        self.delete_entry = ttk.Entry(control)
        self.delete_entry.grid(row=1, column=0, sticky="ew", pady=4)
        control.columnconfigure(0, weight=1)
        ttk.Button(control, text="刪除選取/輸入", style="Danger.TButton", command=self._delete_selected).grid(row=2, column=0, sticky="ew", pady=3)
        ttk.Button(control, text="重產預覽", command=self._redraw_current).grid(row=3, column=0, sticky="ew", pady=3)

        self.image_label = tk.Label(root, bg="#111315", text="尚無圖片", fg="#e0e0e0")
        self.image_label.grid(row=1, column=1, sticky="nsew")
        self.image_label.bind("<Button-1>", self._on_image_click)
        self.image_label.bind("<Configure>", lambda _e: self._render())

        ttk.Label(root, textvariable=self.status_var, style="Dark.TLabel").grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _open_folder(self):
        import subprocess
        folder = DATASET_DIR / "previews" / self.split_var.get()
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(folder)])

    def _load_samples(self):
        labels_dir = DATASET_DIR / "labels" / self.split_var.get()
        self.samples = sorted([p.stem for p in labels_dir.glob("*.txt")]) if labels_dir.exists() else []
        self.sample_list.delete(0, "end")
        for stem in self.samples:
            self.sample_list.insert("end", stem)
        if self.samples:
            self.sample_list.selection_set(len(self.samples) - 1)
            self.sample_list.see(len(self.samples) - 1)
            self._load_sample(self.samples[-1])
        else:
            self.status_var.set("目前沒有 YOLO label。先按一次重新辨識產生資料。")

    def _select_from_list(self):
        sel = self.sample_list.curselection()
        if not sel:
            return
        self._load_sample(self.samples[sel[0]])

    def _paths(self, stem=None):
        return ytool.paths_for(stem or self.sample_var.get(), self.split_var.get())

    def _load_sample(self, stem):
        self.sample_var.set(stem)
        self.selected_index = None
        self.delete_entry.delete(0, "end")
        paths = self._paths(stem)
        self.image_bgr = cv2.imread(str(paths["image"]))
        self.lines = ytool.read_labels(paths["label"])
        if self.image_bgr is None:
            self.status_var.set(f"找不到圖片：{paths['image']}")
            return
        self.status_var.set(f"{stem}：{len(self.lines)} 個框。點錯框或輸入編號後刪除。")
        self._render()

    def _label_boxes(self):
        if self.image_bgr is None:
            return []
        h, w = self.image_bgr.shape[:2]
        boxes = []
        for i, line in enumerate(self.lines, 1):
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = int(round((xc - bw / 2.0) * w))
            y1 = int(round((yc - bh / 2.0) * h))
            x2 = int(round((xc + bw / 2.0) * w))
            y2 = int(round((yc + bh / 2.0) * h))
            boxes.append((i, cls, x1, y1, x2, y2))
        return boxes

    def _render(self):
        if self.image_bgr is None:
            return
        img = self.image_bgr.copy()
        for i, cls, x1, y1, x2, y2 in self._label_boxes():
            color = ytool.COLORS[cls] if 0 <= cls < len(ytool.COLORS) else (0, 255, 255)
            if i == self.selected_index:
                color = (255, 255, 255)
            name = ytool.NAMES[cls] if 0 <= cls < len(ytool.NAMES) else str(cls)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if i == self.selected_index else 2, cv2.LINE_AA)
            cv2.putText(img, f"{i}:{name}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        box_w = max(1, self.image_label.winfo_width())
        box_h = max(1, self.image_label.winfo_height())
        scale = min(box_w / pil.width, box_h / pil.height)
        new_size = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
        resized = pil.resize(new_size, Image.Resampling.LANCZOS)
        self.display_scale = scale
        self.display_size = new_size
        self.display_offset = ((box_w - new_size[0]) // 2, (box_h - new_size[1]) // 2)
        self.preview_photo = ImageTk.PhotoImage(resized)
        self.image_label.configure(image=self.preview_photo, text="")

    def _on_image_click(self, event):
        if self.image_bgr is None:
            return
        ox, oy = self.display_offset
        x = (event.x - ox) / max(self.display_scale, 1e-6)
        y = (event.y - oy) / max(self.display_scale, 1e-6)
        best = None
        best_area = None
        for i, _cls, x1, y1, x2, y2 in self._label_boxes():
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best is None or area < best_area:
                    best = i
                    best_area = area
        if best is not None:
            self.selected_index = best
            self.delete_entry.delete(0, "end")
            self.delete_entry.insert(0, str(best))
            self.status_var.set(f"已選取第 {best} 個框。")
            self._render()

    def _delete_selected(self):
        raw = self.delete_entry.get().replace(",", " ").split()
        if not raw and self.selected_index is not None:
            raw = [str(self.selected_index)]
        try:
            indices = [int(x) for x in raw]
        except ValueError:
            messagebox.showerror("格式錯誤", "請輸入框的編號，例如：39 或 12 19 39")
            return
        if not indices:
            return
        stem = self.sample_var.get()
        if not messagebox.askyesno("刪除框", f"要刪除 {stem} 的框：{indices} 嗎？"):
            return
        deleted, _numbered = ytool.delete_indices(stem, indices, self.split_var.get())
        self._load_sample(stem)
        self.status_var.set(f"已刪除 {deleted} 個框，並更新 label / preview。")

    def _redraw_current(self):
        stem = self.sample_var.get()
        if not stem:
            return
        ytool.draw_preview(stem, self.split_var.get(), numbered=False)
        ytool.draw_preview(stem, self.split_var.get(), numbered=True)
        self._load_sample(stem)
        self.status_var.set("已重產 preview 和 numbered preview。")


if __name__ == "__main__":
    YoloLabelEditor().mainloop()
