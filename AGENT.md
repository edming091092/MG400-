# Agent Handoff: CoinVision MG400 Control

This package is a handoff snapshot for the dual-camera coin detection + MG400 dry-pick system.

Use Chinese when talking to the user.

## Main Goal

Detect coins on a table, classify them by diameter, map selected/all coin centers to MG400 coordinates, and dry-run the MG400 to each target. Vacuum/DO is not implemented yet.

## Main Entry

Preferred UI:

```bat
RUN_COIN_ROBOT_UI.bat
```

Original development entry:

```bat
C:\Users\user\Desktop\coin_classifier\run_coin_robot_ui.bat
```

Main UI source:

```text
src/coin_robot_ui.py
```

## Hardware

- Gemini2 depth camera: depth + robot coordinate mapping.
- Logitech C270 / Quality camera: object detection and coin edge quality.
- Quality camera index: `3`.
- Quality resolution: `1280x720 @ 30fps`.
- Gemini color stream: `1280x720 RGB @ 30fps`.
- MG400 start / camera-clear pose: `X=30, Y=280, Z=150`.
- Table Z: `-160.0`.
- Current dry lower Z: `-156`.
- Current robot target offset: `X=+13 mm`, `Y=-18 mm` in `src/dual_camera_config.json`.
- Conservative auto-pick workspace: `X=120..380 mm`, `Y=-250..190 mm`; outside targets should be `Check`, not automatic dry lower.

## Important Files

Core UI / runtime:

- `src/coin_robot_ui.py`
- `src/dual_camera_live.py`
- `src/camera_preview_once.py`
- `src/hover_robot_target.py`
- `src/robot_emergency_stop.py`
- `src/robot_clear_enable.py`
- `src/select_quality_roi.py`

Calibration and tuning:

- `src/capture_quality_calib.py`
- `src/calibrate_camera.py`
- `src/capture_stereo_calib_pairs.py`
- `src/calibrate_quality_to_gemini_homography.py`
- `src/calibrate_stereo_extrinsics.py`
- `src/calibrate_robot_tabletop_homography.py`
- `src/tune_gemini_display_roi.py`
- `src/tune_gemini_exposure.py`

Current config:

- `config/dual_camera_config.json`
- `config/quality_camera_calib.json`
- `config/quality_to_gemini_homography.json`
- `config/robot_tabletop_homography.json`
- `config/robot_targets.json`
- `config/robot_action_status.json`
- `config/roi_config.json`
- `config/stereo_extrinsics.json`

The same JSON files are also copied into `src/` because most scripts read config from their own directory.

`src/core/` is a runnable copy of the external core. Package scripts append the old external path as fallback, but should prefer local `src/core` first.

External core backup:

- `external_core/camera.py`
- `external_core/robot.py`
- `external_core/transform.py`

The live original core path used during development was:

```text
C:\Users\user\Desktop\sam3+座標轉換與夾取\core
```

## Runtime Behavior

The UI intentionally keeps normal preview lightweight:

- Preview mode: ROI-cropped Quality camera only, no overlay, no SAM3, no Gemini depth.
- Manual `Refresh / Lock Vision`, `Set Detection ROI`, camera switching, and speed sliders are hidden from the normal operator UI.
- Robot start buttons stop preview, move to camera-clear pose, run real detection, lock coordinates, then execute MG400 motion.

Robot actions are dry-run only. No vacuum output is toggled.

## Current Detection Pipeline

1. Quality camera image is processed with SAM3 + ellipse fitting.
2. Coin class is estimated from measured diameter:
   - 1NT: 20 mm
   - 5NT: 22 mm
   - 10NT: 26 mm
   - 50NT: 28 mm
3. Quality coin center is mapped to Gemini pixel via `quality_to_gemini_homography.json`.
4. Gemini samples depth near that mapped pixel.
5. Diameter is refined by projecting ellipse endpoints into Gemini/depth space.
6. Gemini pixel is mapped to MG400 tabletop XY via `robot_tabletop_homography.json`.
7. Robot target Z is currently fixed by table model / dry lower height.

## Current Robot Coordinate Conversion

This is the most important implementation detail.

The system currently does NOT use ArUco during normal operation and does NOT do full 3D world-coordinate reconstruction for the robot.

The live MG400 mapping is:

```text
Quality pixel -> Gemini pixel -> MG400 tabletop X/Y
```

Details:

- `quality_to_gemini_homography.json` maps Quality image pixels to Gemini image pixels.
- `robot_tabletop_homography.json` maps Gemini image pixels to MG400 `X/Y` on the tabletop plane.
- `robot_table_z_mm` is fixed at `-160.0`.
- Dry lower target is `-156`.
- Runtime target compensation is currently `X +7 mm`, `Y -6 mm`.

`dual_camera_config.json` still contains fallback paths:

- `robot_h_path`: old ArUco / homography fallback `H.npy`.
- `robot_extrinsics_path`: old camera extrinsics fallback.

But the preferred/current path is `robot_tabletop_homography.json`.

Current tabletop homography quality:

- Source: manual tabletop points.
- Points: 12.
- Mean error: about `0.907 mm`.
- Max error: about `2.707 mm`.

## How to Improve Robot Coordinate Stability

Prioritize these:

1. Keep `robot_tabletop_homography.json` calibration points around 12-16 and cover the actual coin area.
2. Spread points across the whole usable table ROI, especially corners and edges.
3. Use a sharp TCP pointer or rigid calibration tool instead of eyeballing with a large suction cup.
4. Keep Gemini, table, and MG400 base mechanically fixed.
5. Do not change Gemini original resolution after calibration.
6. If only display ROI changes, calibration can stay. If actual camera resolution changes, recalibrate.
7. Add TCP offset calibration before real vacuum pickup.
8. Add a table-plane model if Z needs to be accurate over a non-flat table.
9. Add a quick 3-point verification routine before production use.
10. For faster runtime, keep SAM3 and camera streams resident instead of launching a subprocess per recognition.

## Recalibration Rules

If Quality camera moves:

- Redo Quality to Gemini homography.
- Redo Quality intrinsics only if focus/resolution/lens setting changed significantly.
- Gemini to MG400 can stay if Gemini/table/robot did not move.

If Gemini moves:

- Redo Quality to Gemini homography.
- Redo Gemini to MG400 tabletop homography.

If table or MG400 base moves:

- Redo Gemini to MG400 tabletop homography.
- Strongly consider redoing Quality to Gemini homography.

If actual camera resolution changes:

- Redo all pixel-based calibrations.

## Known Good Calibration Values

Quality intrinsics:

- File: `quality_camera_calib.json`
- Chessboard: 9x6 inner corners.
- Square: 26 mm.
- Images: 22.
- Reprojection error: `0.2887 px`.

Robot tabletop homography:

- File: `robot_tabletop_homography.json`
- Mean error: `0.907 mm`.
- Max error: `2.707 mm`.
- Points: 12.

Quality to Gemini homography:

- File: `quality_to_gemini_homography.json`
- Purpose: camera-to-camera image matching, not world coordinates.
- Current synced-chessboard calibration uses 18 pairs.
- Mean error: `0.22 px`.
- Max error: `9.01 px`.
- If this drifts, robot XY can be wrong even when tabletop homography is good.

Stereo extrinsics:

- File: `stereo_extrinsics.json`.
- Not the primary robot mapping method.

## UI Notes

User-requested UI behaviors already implemented:

- One camera view at a time by default.
- Opens as a plain live ROI camera monitor for speed.
- Top camera selector is available for Quality / Gemini / Combined views.
- `Settings` / `設定` dialog controls operator-vs-engineer mode, Chinese/English UI language, move speed, and lower speed.
- Operator mode keeps only daily controls; engineer mode reveals ROI selection, manual refresh/lock, selected-target actions, and all-target dry run.
- Mouse wheel zoom remains available.
- ROI and engineering calibration controls are fixed outside the operator flow.
- Start actions:
  - `辨識後移到第一顆上方`.
  - `辨識後下降第一顆到 Z=-156`.
- Shows selected target, current action, coordinates, summary counts, total NT.
- Shows loading/model state during recognition.
- Emergency stop and Clear+Enable are in UI.
- Robot returns to camera-clear/start pose after motion so next recognition is not blocked.

## MG400 Errors

`hover_robot_target.py` writes `robot_action_status.json`.

UI reads this file and shows a modal with:

- Target name/class.
- Target XY.
- Attempted XYZ.
- Error code.
- Controller response.
- Meaning.
- Likely cause.
- Suggested action.

Local hints currently cover common codes:

- 2: controller alarm / paused / not ready.
- 17: edge or unreachable travel target; often a selected coin outside the conservative workspace.
- 18: alarm during low-Z motion; often dry-lower too low, XY offset, collision detection, or edge target.
- 23: motion interrupted or path rejected.
- 98: controller not ready / alarm state / unknown returned by controller.

If an unknown code appears, do not hide it. Show the raw code and ask the user to check DobotStudio alarm details.

## Common Failures

Camera MFT / hardware resource error:

- Usually another Python/camera process still owns Gemini or the webcam.
- Kill old `coin_classifier` Python processes before opening UI.

UnicodeDecodeError cp950 in subprocess reader:

- The subprocess emitted bytes that cannot decode in cp950.
- Keep subprocess decoding as UTF-8 with replacement where possible.

Robot no movement but says done:

- Check dry-run mode, MG400 connection, action status file, and whether hover/lower command was skipped due to range/safety check.

Long recognition:

- Current design starts subprocess/model/camera for official recognition.
- `--fast --quality-only` helps, but persistent background worker is the real next speed improvement.

## Safety

Do not enable vacuum/DO without adding:

- Confirmed suction output channel.
- Pickup height test.
- Place location.
- Failure detection.
- Physical emergency stop procedure.

Do not lower to a new Z without first testing hover at safe Z.
