import os
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import __version__
from ultralytics.utils.torch_utils import unwrap_model


def _safe_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(obj, str(tmp))
    if path.exists():
        path.unlink()
    os.replace(str(tmp), str(path))


def _save_model_lightweight(self):
    """Avoid Windows write_bytes crashes by streaming a smaller YOLO checkpoint."""
    ema = deepcopy(unwrap_model(self.ema.ema)).half()
    if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
        print(f"[YOLO save] skip epoch {self.epoch}: EMA contains NaN/Inf")
        return False

    ckpt = {
        "epoch": self.epoch,
        "best_fitness": self.best_fitness,
        "model": None,
        "ema": ema,
        "updates": self.ema.updates,
        "optimizer": None,
        "scaler": None,
        "train_args": vars(self.args),
        "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    epoch_no = self.epoch + 1
    last_path = self.wdir / f"last_epoch{epoch_no:03d}.pt"
    _safe_torch_save(ckpt, last_path)
    self.last = last_path
    if self.best_fitness == self.fitness:
        best_path = self.wdir / f"best_epoch{epoch_no:03d}.pt"
        _safe_torch_save(ckpt, best_path)
        self.best = best_path
    if (self.save_period > 0) and (self.epoch % self.save_period == 0):
        _safe_torch_save(ckpt, self.wdir / f"epoch{self.epoch}.pt")
    return True


def _final_eval_no_strip(self):
    model = self.best if self.best.exists() else None
    if model:
        print(f"\nValidating {model}...")
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        self.metrics = self.validator(model=model)
        self.metrics.pop("fitness", None)
        self.run_callbacks("on_fit_epoch_end")


BaseTrainer.save_model = _save_model_lightweight
BaseTrainer.final_eval = _final_eval_no_strip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--name", default="yolov8m_960_visible")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    model = YOLO(args.model)
    model.train(
        data="src/yolo_dataset/data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(root / "runs_yolo_coin"),
        name=args.name,
        exist_ok=True,
        workers=0,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
