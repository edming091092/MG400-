# -*- coding: utf-8 -*-
"""Clear MG400 errors and enable robot."""

import sys
from pathlib import Path


GEMINI_LIBS = Path(r"C:\Users\user\Desktop\sam3+座標轉換與夾取")
if str(GEMINI_LIBS) not in sys.path:
    sys.path.append(str(GEMINI_LIBS))

from core.robot import MG400


def main():
    robot = MG400()
    try:
        robot.connect()
        robot.clear_error()
        resp = robot.enable()
        errs = robot.get_errors()
        print(f"[MG400] Enable response: {resp}")
        print(f"[MG400] Current errors: {errs}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
