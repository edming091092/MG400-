# -*- coding: utf-8 -*-
"""Helpers for calibration capture sessions."""

import shutil
from datetime import datetime
from pathlib import Path


def choose_session_mode(title, existing_count=0, default="reset"):
    """Ask whether to archive old data or append to it.

    Returns "reset" or "append".  Batch files use the interactive prompt; tests
    and advanced callers can pass CLI flags and skip this function.
    """
    print("")
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"目前舊資料數量：{existing_count}")
    print("1 = 清空舊資料重新開始（會先自動備份，不直接刪除）")
    print("2 = 保留舊資料，繼續新增")
    suffix = "1" if default == "reset" else "2"
    choice = input(f"請選擇 [1/2]，直接 Enter = {suffix}: ").strip()
    if choice == "2":
        return "append"
    if choice == "1":
        return "reset"
    return default


def archive_path(path):
    path = Path(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.name}_backup_{stamp}")


def archive_dir(path):
    path = Path(path)
    if not path.exists():
        return None
    backup = archive_path(path)
    shutil.move(str(path), str(backup))
    return backup


def archive_file(path):
    path = Path(path)
    if not path.exists():
        return None
    backup = archive_path(path)
    shutil.move(str(path), str(backup))
    return backup
