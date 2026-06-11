"""Local coin parameter database for fast denomination lookup."""

import json
import statistics
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).parent
PARAM_DB_FILE = HERE / "coin_param_db.json"

DEFAULT_DB = {
    "version": 1,
    "diameter_tolerance_mm": 1.35,
    "min_margin_mm": 0.45,
    "coins": {
        "1yuan": {"label_name": "1NT", "value_nt": 1, "diameter_mm": 20.0},
        "5yuan": {"label_name": "5NT", "value_nt": 5, "diameter_mm": 22.0},
        "10yuan": {"label_name": "10NT", "value_nt": 10, "diameter_mm": 26.0},
        "50yuan": {"label_name": "50NT", "value_nt": 50, "diameter_mm": 28.0},
    },
}


def ensure_coin_param_db(path=PARAM_DB_FILE):
    path = Path(path)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_DB, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_coin_param_db(path=PARAM_DB_FILE):
    path = ensure_coin_param_db(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = DEFAULT_DB
    coins = data.get("coins") or DEFAULT_DB["coins"]
    return {
        "version": data.get("version", 1),
        "diameter_tolerance_mm": float(data.get("diameter_tolerance_mm", DEFAULT_DB["diameter_tolerance_mm"])),
        "min_margin_mm": float(data.get("min_margin_mm", DEFAULT_DB["min_margin_mm"])),
        "coins": coins,
    }


def save_coin_param_db(data, path=PARAM_DB_FILE):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _coin_match_value(spec, min_samples=3):
    samples = spec.get("learned_samples") or []
    if len(samples) >= int(min_samples):
        return float(spec.get("learned_diameter_mm", spec.get("diameter_mm")))
    return float(spec.get("diameter_mm"))


def learn_diameter_sample(label, diameter_mm, confidence=1.0, source="", path=PARAM_DB_FILE, max_samples=500):
    if label in (None, "", "?") or diameter_mm is None:
        return False
    data = load_coin_param_db(path)
    coins = data.setdefault("coins", {})
    if label not in coins:
        return False
    spec = coins[label]
    sample = {
        "diameter_mm": round(float(diameter_mm), 4),
        "confidence": round(float(confidence or 0.0), 4),
        "source": str(source or ""),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    samples = list(spec.get("learned_samples") or [])
    samples.append(sample)
    spec["learned_total_count"] = int(spec.get("learned_total_count", len(samples) - 1) or 0) + 1
    samples = samples[-int(max_samples):]
    values = [float(s["diameter_mm"]) for s in samples if s.get("diameter_mm") is not None]
    if values:
        spec["learned_samples"] = samples
        spec["learned_sample_count"] = len(values)
        spec["learned_diameter_mm"] = round(float(statistics.median(values)), 4)
        if len(values) >= 2:
            spec["learned_diameter_stdev_mm"] = round(float(statistics.pstdev(values)), 4)
        spec["updated_at"] = sample["timestamp"]
        save_coin_param_db(data, path)
        return True
    return False


def classify_by_diameter(diameter_mm, db=None, tolerance_mm=None, min_margin_mm=None):
    if diameter_mm is None:
        return "?", None, 0.0, "no_diameter"
    db = db or load_coin_param_db()
    tolerance = float(tolerance_mm if tolerance_mm is not None else db["diameter_tolerance_mm"])
    margin = float(min_margin_mm if min_margin_mm is not None else db["min_margin_mm"])

    ranked = []
    for label, spec in db["coins"].items():
        diff = abs(float(diameter_mm) - _coin_match_value(spec))
        ranked.append((diff, label))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        return "?", None, 0.0, "empty_db"

    best_diff, best_label = ranked[0]
    second_diff = ranked[1][0] if len(ranked) > 1 else best_diff + tolerance + margin
    confidence = max(0.0, min(1.0, 1.0 - best_diff / max(tolerance, 1e-6)))
    if best_diff > tolerance:
        return "?", best_diff, confidence, "diameter_out_of_tolerance"
    if second_diff - best_diff < margin:
        return "?", best_diff, confidence, "low_margin"
    return best_label, best_diff, confidence, "param_db"
