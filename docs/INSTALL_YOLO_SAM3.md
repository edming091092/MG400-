# YOLO / SAM3 install notes

This project uses Ultralytics for both YOLO and SAM3.

## Python environment

Use a Windows conda environment with Python 3.12:

```powershell
conda create -n coin python=3.12 -y
conda activate coin
python -m pip install --upgrade pip
```

## Install packages

For an NVIDIA GPU machine, install the CUDA build of PyTorch first:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-yolo-sam3.txt
```

For CPU-only testing, install PyTorch from the normal PyPI index instead:

```powershell
python -m pip install torch torchvision
python -m pip install -r requirements-yolo-sam3.txt
```

## Model files

The pip packages are only the code libraries. The model weights are separate files.

YOLO base weights already tracked in this repository through Git LFS:

- `yolov8m.pt`
- `yolo11n.pt`
- `yolo26n.pt`

SAM3 model weight:

- Expected filename: `sam3.pt`
- Current local source found on this PC: `C:\Users\user\Desktop\專題\sam3.pt`
- Default locations checked by the code include:
  - `src\sam3.pt`
  - project root `sam3.pt`
  - `%USERPROFILE%\Desktop\sam3.pt`
  - `%USERPROFILE%\Desktop\專題\sam3.pt`

The local `sam3.pt` is about 3.45 GB, so it is not committed to the normal Git repository. Copy it to the new computer separately, or publish it as a large release/LFS asset.

## Camera SDK note

Gemini/Orbbec depth camera support imports `pyorbbecsdk` at runtime. That SDK is not installed in the current `coin` environment, so install the Orbbec SDK package separately on the target PC if Gemini depth camera features are needed.

