# -*- coding: utf-8 -*-
"""Shared defaults for legacy external_core modules."""

from pathlib import Path

import cv2


HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

ROBOT_IP = "192.168.1.6"
DASH_PORT = 29999
MOVE_PORT = 30003

MOVE_SPEED_PCT = 40
MOVE_TIMEOUT_S = 30.0
MOVE_TOL_MM = 2.0

TABLE_Z_MM = -160.0

try:
    ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
except AttributeError:
    ARUCO_DICT_ID = 0

MARKER_IDS = [0, 1, 2, 3]
MARKER_DATA_FILE = str(HERE / "marker_world_coords.json")
EXTRINSICS_FILE = str(DATA_DIR / "camera_extrinsics.npz")
