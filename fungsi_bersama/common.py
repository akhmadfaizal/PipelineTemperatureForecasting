from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def make_run_dir(cfg: dict, requested: str | None = None) -> Path:
    if requested:
        run_dir = Path(requested).expanduser().resolve()
    else:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        run_dir = (Path(cfg["project"]["run_root"]) / f"candidate_{stamp}").resolve()
    for name in ["config", "dataset", "data", "models", "deployment", "results", "logs", "test_vectors", "state"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir


def _timestamps(series: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(series, format="mixed", errors="raise", utc=True)
    return parsed.dt.tz_convert(timezone)


def _interpolate_short(frame: pd.DataFrame, max_slots: int) -> tuple[pd.DataFrame, int]:
    result = frame.copy()
    filled = 0
    for column in result.columns:
        missing = result[column].isna()
        groups = (missing != missing.shift()).cumsum()
        lengths = missing.groupby(groups).transform("size")
        eligible = missing & (lengths <= max_slots)
        interpolated = result[column].interpolate(method="linear", limit=max_slots, limit_area="inside")
        can_fill = eligible & interpolated.notna()
        result.loc[can_fill, column] = interpolated.loc[can_fill]
        filled += int(can_fill.sum())
    return result, filled


def _segments(frame: pd.DataFrame, min_length: int) -> list[pd.DataFrame]:
    valid = frame.notna().all(axis=1)
    groups = (valid != valid.shift()).cumsum()
    return [part for _, part in frame.groupby(groups) if valid.loc[part.index[0]] and len(part) >= min_length]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    hour = result.index.hour + result.index.minute / 60.0
    day = result.index.dayofweek.astype(float)
    result["jam_sin"] = np.sin(2 * np.pi * hour / 24)
    result["jam_cos"] = np.cos(2 * np.pi * hour / 24)
    result["hari_sin"] = np.sin(2 * np.pi * day / 7)
    result["hari_cos"] = np.cos(2 * np.pi * day / 7)
    result["akhir_pekan"] = (result.index.dayofweek >= 5).astype(float)
    return result


def _sequences(segments: list[pd.DataFrame], features: list[str], target: str, lookback: int, horizon: int):
    xs, ys, y_celsius, target_times, issue_times = [], [], [], [], []
    for segment in segments:
        enriched = add_features(segment)
        values = enriched[features].to_numpy(dtype="float32")
        scaled_target = enriched[target].to_numpy(dtype="float32")
        raw_target = enriched[f"{target}_celsius"].to_numpy(dtype="float32")
        for start in range(len(enriched) - lookback - horizon + 1):
            issue = start + lookback - 1
            target_pos = issue + horizon
            xs.append(values[start : start + lookback])
            ys.append(scaled_target[target_pos])
            y_celsius.append(raw_target[target_pos])
            issue_times.append(enriched.index[issue].tz_localize(None).to_datetime64())
            target_times.append(enriched.index[target_pos].tz_localize(None).to_datetime64())
    shape = (0, lookback, len(features))
    return (
        np.stack(xs) if xs else np.empty(shape, dtype="float32"),
        np.asarray(ys, dtype="float32"),
        np.asarray(y_celsius, dtype="float32"),
        np.asarray(issue_times, dtype="datetime64[ns]"),
        np.asarray(target_times, dtype="datetime64[ns]"),
    )


def fit_sensor_scaler(frame: pd.DataFrame, columns: list[str], preprocessing: dict):
    """Fit observed sensor extrema on the configured information boundary."""
    mode = preprocessing.get("scaler_fit", "all")
    if mode == "train_only":
        left, right = map(pd.Timestamp, preprocessing["split"]["train"])
        fit_frame = frame.loc[(frame.index >= left) & (frame.index < right), columns]
    elif mode == "all":
        # Compatibility for immutable historical configs that used global scaling.
        fit_frame = frame[columns]
    else:
        raise ValueError(f"Mode scaler tidak didukung: {mode}")
    minimum, maximum = fit_frame.min(), fit_frame.max()
    if not np.isfinite(minimum.to_numpy()).all() or not np.isfinite(maximum.to_numpy()).all():
        raise ValueError("Data fit scaler kosong atau tidak finite.")
    if ((maximum - minimum) <= 0).any():
        raise ValueError("Rentang Min-Max nol atau negatif.")
    return minimum, maximum


def prepare_dataset(cfg: dict, run_dir: Path, allow_incomplete: bool = False) -> dict:
    data_cfg, pre = cfg["data"], cfg["preprocessing"]
    source = Path(data_cfg["csv_path"]).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {source}")
    if (run_dir / "dataset" / "metadata.json").exists():
        raise FileExistsError("Run sudah dipraproses. Pakai artefak yang ada atau direktori run baru; jangan menimpa dataset run.")
    frame = pd.read_csv(source)
    required = {data_cfg["kolom_waktu"], data_cfg["kolom_target"], data_cfg["kolom_kelembapan"]}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Kolom dataset belum lengkap: " + ", ".join(sorted(missing)))
    timezone = cfg["project"]["timezone"]
    frame["timestamp"] = _timestamps(frame[data_cfg["kolom_waktu"]], timezone)
    temperature, humidity = data_cfg["kolom_target"], data_cfg["kolom_kelembapan"]
    frame = frame[["timestamp", temperature, humidity]].copy()
    frame[[temperature, humidity]] = frame[[temperature, humidity]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["timestamp", temperature, humidity]).sort_values("timestamp")
    raw_first, raw_last = frame["timestamp"].min(), frame["timestamp"].max()
    start, end = pd.Timestamp(data_cfg["mulai"]), pd.Timestamp(data_cfg["akhir_eksklusif"])
    tolerance = pd.Timedelta(minutes=int(data_cfg["toleransi_ujung_menit"]))
    complete = raw_first <= start + tolerance and raw_last >= end - tolerance
    if data_cfg.get("kandidat_harus_lengkap", True) and not allow_incomplete and not complete:
        raise ValueError(
            f"Dataset belum menutup jendela final. Rentang tersedia {raw_first} sampai {raw_last}; "
            f"butuh cakupan {start} sampai mendekati {end}. Gunakan --allow-incomplete hanya untuk smoke test."
        )
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
    removed_artifact = 0
    for interval in data_cfg.get("artefak_ranges", []):
        left = pd.Timestamp(interval["mulai_eksklusif"])
        right = pd.Timestamp(interval["selesai_eksklusif"])
        mask = (frame["timestamp"] > left) & (frame["timestamp"] < right)
        removed_artifact += int(mask.sum())
        frame = frame.loc[~mask]
    lo, hi = data_cfg["batas_operasi_suhu"]
    invalid = ~frame[temperature].between(lo, hi)
    removed_invalid = int(invalid.sum())
    frame = frame.loc[~invalid].drop_duplicates(subset="timestamp", keep="first")
    raw_count = len(frame)
    resampled = frame.set_index("timestamp")[[temperature, humidity]].resample(pre["resample"]).mean()
    full_grid = pd.date_range(start=start, end=end, freq=pre["resample"], inclusive="left")
    resampled = resampled.reindex(full_grid)
    resampled.index.name = "timestamp"
    scaler_min, scaler_max = fit_sensor_scaler(resampled, [temperature, humidity], pre)
    span = scaler_max - scaler_min
    if (span <= 0).any():
        raise ValueError("Rentang Min-Max nol atau negatif.")
    scaled = (resampled - scaler_min) / span
    scaled[f"{temperature}_celsius"] = resampled[temperature]
    scaled[f"{humidity}_raw"] = resampled[humidity]
    split_frames, split_segments, rows = {}, {}, []
    max_slots = int(pd.Timedelta(minutes=pre["max_gap_interpolasi_menit"]) / pd.Timedelta(pre["resample"]))
    min_length = pre["lookback"] + max(pre["horizons"])
    for name, bounds in pre["split"].items():
        left, right = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        part = scaled[(scaled.index >= left) & (scaled.index < right)].copy()
        sensor_columns = [temperature, humidity, f"{temperature}_celsius", f"{humidity}_raw"]
        part[sensor_columns], filled = _interpolate_short(part[sensor_columns], max_slots)
        segments = _segments(part, min_length)
        split_frames[name], split_segments[name] = part, segments
        rows.append({"split": name, "slots": len(part), "filled_cells": filled, "segments": len(segments)})
    prepared = pd.concat([part.assign(split=name) for name, part in split_frames.items()]).reset_index()
    prepared.rename(columns={prepared.columns[0]: "timestamp"}, inplace=True)
    atomic_csv(run_dir / "data" / "prepared_series.csv", prepared)
    features = list(pre["features"])
    sequence_counts = []
    for horizon in pre["horizons"]:
        package = {"features": np.asarray(features), "lookback": pre["lookback"], "horizon": horizon}
        for name in ["train", "val", "test"]:
            x, y, yc, issue, target_time = _sequences(
                split_segments[name], features, temperature, pre["lookback"], horizon
            )
            package.update({f"X_{name}": x, f"y_{name}": y, f"ycel_{name}": yc,
                            f"issue_{name}": issue, f"t_{name}": target_time})
            actual_series = split_frames[name][f"{temperature}_celsius"].copy()
            actual_series.index = actual_series.index.tz_localize(None)
            actual_at_issue = actual_series.reindex(pd.DatetimeIndex(issue)).to_numpy(dtype="float64")
            if not np.isfinite(actual_at_issue).all():
                raise ValueError("Suhu aktual saat issue tidak tersedia untuk sekuens valid.")
            package[f"actual_at_issue_{name}"] = actual_at_issue
            sequence_counts.append({"horizon_minutes": horizon * 5, "split": name, "samples": len(y)})
        np.savez_compressed(run_dir / "data" / f"seq_h{horizon}.npz", **package)
    scaler = {"mode": "fit_on_train" if pre.get("scaler_fit", "all") == "train_only" else "fit_on_all",
              "fit_split": "train" if pre.get("scaler_fit", "all") == "train_only" else "all",
              "clip": False, "columns": [temperature, humidity],
              "min": scaler_min.tolist(), "max": scaler_max.tolist()}
    atomic_json(run_dir / "data" / "scaler.json", scaler)
    atomic_csv(run_dir / "results" / "preprocessing_splits.csv", pd.DataFrame(rows))
    atomic_csv(run_dir / "results" / "sequence_counts.csv", pd.DataFrame(sequence_counts))
    config_snapshot = run_dir / "config" / "config.yaml"
    config_snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    valid_slots = resampled[[temperature, humidity]].notna().all(axis=1)
    missing_slots = ~valid_slots
    missing_groups = (missing_slots != missing_slots.shift()).cumsum()
    gap_lengths = [len(part) for _, part in resampled.groupby(missing_groups)
                   if missing_slots.loc[part.index[0]]]
    total_slots = len(resampled)
    complete_slots = int(valid_slots.sum())
    completeness = complete_slots / total_slots if total_slots else 0.0
    meta = {
        "status": cfg["project"]["status"], "source": str(source), "source_size_bytes": source.stat().st_size,
        "raw_first": raw_first.isoformat(), "raw_last": raw_last.isoformat(), "final_window_complete": complete,
        "rows_after_filter": raw_count, "removed_artifact_rows": removed_artifact,
        "removed_outside_sensor_range": removed_invalid, "feature_order": features,
        "lookback": pre["lookback"], "horizons": pre["horizons"],
        "total_five_minute_slots": total_slots, "complete_five_minute_slots": complete_slots,
        "missing_five_minute_slots": total_slots - complete_slots,
        "completeness_fraction": completeness, "completeness_target_fraction": 0.95,
        "meets_completeness_target": completeness >= 0.95,
        "long_gap_count": len(gap_lengths), "longest_gap_slots": max(gap_lengths, default=0),
        "meets_no_gap_over_three_hours": max(gap_lengths, default=0) <= 36,
    }
    meta["fingerprint"] = stable_hash({"dataset_metadata": meta, "config": cfg, "schema": 2})
    atomic_json(run_dir / "dataset" / "metadata.json", meta)
    return meta


def load_sequence(run_dir: Path, horizon: int, splits: list[str] | None = None) -> dict[str, np.ndarray]:
    path = run_dir / "data" / f"seq_h{horizon}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Artefak sekuens belum ada: {path}")
    loaded = np.load(path, allow_pickle=False)
    if splits is None:
        keys = loaded.files
    else:
        suffixes = tuple(f"_{name}" for name in splits)
        keys = [key for key in loaded.files if key in {"features", "lookback", "horizon"} or key.endswith(suffixes)]
    return {key: loaded[key] for key in keys}


def inverse_target(run_dir: Path, values: np.ndarray) -> np.ndarray:
    scaler = json.loads((run_dir / "data" / "scaler.json").read_text())
    lo, hi = float(scaler["min"][0]), float(scaler["max"][0])
    return np.asarray(values) * (hi - lo) + lo


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = np.asarray(y_true) - np.asarray(y_pred)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mape": float(np.mean(np.abs(error / y_true)) * 100), "n": int(len(error))}
