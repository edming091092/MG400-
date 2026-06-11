# CoinVision YOLO Dataset

This folder is written automatically by `dual_camera_live.py` when
`yolo_export_enabled` is true.

Default export policy:

- Only SAM3-generated boxes are exported: `sam3`, `sam3_fallback`.
- Images are saved under `images/train` or `images/val`.
- YOLO labels are saved under `labels/train` or `labels/val`.
- `data.yaml` is regenerated automatically for YOLO training.
- Optional export metadata is appended to `metadata/<split>/exports.jsonl`.

Training entry point:

```powershell
python src\train_yolov8m_visible.py --epochs 120 --imgsz 960 --batch 4
```

Change `yolo_export_split` in `src/dual_camera_config.json` to choose `train`
or `val`.
