"""Train and select direct-horizon LSTM/GRU models with immutable run evidence.

Phases: smoke checks fit/save/load; benchmark estimates duration; tune selects
shared hyperparameters using validation; final fits locked choices for each seed.
TensorFlow imports stay inside training functions so provenance can be inspected
without initializing a GPU runtime. Never resume an old run after editing source.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / "fungsi_bersama"))
sys.path.insert(0, str(PIPELINE_ROOT / "04_training"))
import common
import pc_training_state as state


def build_model(architecture: str, hp: dict, input_shape: tuple[int, int], loss: str):
    """Build one LSTM or GRU model with a scalar direct-horizon output."""
    from tensorflow import keras

    cell = keras.layers.LSTM if architecture == "lstm" else keras.layers.GRU
    model = keras.Sequential(name=f"{architecture}_{hp['config_id']}")
    model.add(keras.Input(shape=input_shape))
    for index in range(hp["layers"]):
        model.add(cell(hp["units"], return_sequences=index < hp["layers"] - 1))
        model.add(keras.layers.Dropout(hp["dropout"]))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(hp["learning_rate"]), loss=loss)
    return model


def grid(cfg: dict) -> list[dict]:
    """Enumerate configured candidates in stable order without choosing a winner."""
    values = cfg["model"]["grid"]
    result = []
    for units, layers, dropout, lr, batch in itertools.product(
        values["units"],
        values["layers"],
        values["dropout"],
        values["learning_rate"],
        values["batch_size"],
    ):
        hp = {
            "units": units,
            "layers": layers,
            "dropout": float(dropout),
            "learning_rate": float(lr),
            "batch_size": batch,
        }
        hp["config_id"] = f"u{units}_l{layers}_d{dropout}_lr{lr}_b{batch}".replace(
            ".", "p"
        )
        result.append(hp)
    return result


def environment_snapshot(run_dir: Path, cfg: dict) -> dict:
    """Verify GPU availability and record the exact environment used for fitting."""
    import importlib.metadata as metadata
    import tensorflow as tf

    packages = [
        "tensorflow",
        "tensorflow-metal",
        "keras",
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "PyYAML",
        "jupyterlab",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    versions.update(
        {
            d.metadata["Name"]: d.version
            for d in metadata.distributions()
            if d.metadata["Name"]
        }
    )
    devices = tf.config.list_physical_devices("GPU")
    if cfg.get("runtime", {}).get("require_gpu", False) and not devices:
        raise RuntimeError(
            "GPU diwajibkan untuk run PC ini; TensorFlow tidak mendeteksi GPU."
        )
    if cfg.get("runtime", {}).get("memory_growth", True):
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
    gpus = [device.name for device in devices]
    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        probe = tf.matmul(tf.ones((16, 16)), tf.ones((16, 16)))
    snapshot = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "physical_gpus": gpus,
        "probe_device": probe.device,
        "same_process_as_model_fit": True,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    import subprocess

    try:
        gpu_runtime = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=20,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        gpu_runtime = "unavailable"
    snapshot["gpu_runtime"] = gpu_runtime
    snapshot["hostname"] = platform.node()
    snapshot["machine"] = platform.machine()
    snapshot["tensorflow_build"] = tf.sysconfig.get_build_info()
    snapshot["gpu_details"] = [
        tf.config.experimental.get_device_details(d)
        for d in tf.config.list_physical_devices("GPU")
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    common.atomic_json(run_dir / "logs" / "environments" / f"{stamp}.json", snapshot)
    return snapshot


def _row_key(
    fingerprint: str, architecture: str, horizon: int, hp: dict, seed: int
) -> str:
    """Hash the run, model, horizon, hyperparameters, and seed into one fit key."""
    return common.stable_hash(
        {
            "fingerprint": fingerprint,
            "architecture": architecture,
            "horizon": horizon,
            "hp": hp,
            "seed": seed,
        }
    )


def load_progress(path: Path) -> pd.DataFrame:
    """Read recorded trials, returning an empty table for a fresh phase."""
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def train_once(
    cfg: dict,
    sequence: dict,
    architecture: str,
    hp: dict,
    seed: int,
    epochs: int | None = None,
    subset: int | None = None,
    attempt_dir: Path | None = None,
):
    """Fit only training data, select the best validation checkpoint, and return fit evidence."""
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    x_train, y_train = sequence["X_train"], sequence["y_train"]
    x_val, y_val = sequence["X_val"], sequence["y_val"]
    if subset:
        x_train, y_train = x_train[:subset], y_train[:subset]
        x_val, y_val = x_val[: max(64, subset // 4)], y_val[: max(64, subset // 4)]
    if not len(x_train) or not len(x_val):
        raise ValueError("Training dan validasi harus memiliki sampel.")
    model = build_model(architecture, hp, x_train.shape[1:], cfg["model"]["loss"])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["model"]["patience"],
            restore_best_weights=True,
            verbose=0,
        )
    ]
    if attempt_dir:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        callbacks += [
            keras.callbacks.CSVLogger(str(attempt_dir / "history.csv")),
            keras.callbacks.ModelCheckpoint(
                str(attempt_dir / "best.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            keras.callbacks.TerminateOnNaN(),
        ]
    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs or cfg["model"]["epochs"],
        batch_size=int(hp["batch_size"]),
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )
    duration = time.perf_counter() - started
    losses = np.asarray(history.history["val_loss"])
    if not np.isfinite(losses).all():
        raise RuntimeError("Loss non-finite; trial gagal dan tidak boleh dipilih.")
    # Reload the persisted best checkpoint, including optimizer state, for final artefacts.
    if attempt_dir:
        model = keras.models.load_model(attempt_dir / "best.keras")
    prediction = model.predict(x_val, verbose=0).reshape(-1)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Prediksi validasi non-finite.")
    predicted_celsius = common.inverse_target(Path(cfg["_run_dir"]), prediction)
    val_mae = common.metrics(
        sequence["ycel_val"][: len(prediction)], predicted_celsius
    )["mae"]
    details = {
        "val_mae_celsius": val_mae,
        "duration_seconds": duration,
        "epochs_ran": len(history.history["loss"]),
        "best_epoch": int(np.argmin(losses) + 1),
        "trainable_params": int(model.count_params()),
        "variable_devices": sorted(
            {
                str(getattr(getattr(v, "value", v), "device", "unknown"))
                for v in model.trainable_variables
            }
        ),
        "same_process_as_model_fit": True,
        "op_determinism_enabled": True,
        "seed": seed,
    }
    if attempt_dir:
        common.atomic_json(attempt_dir / "fit_summary.json", details)
    return model, details


def append_result(path: Path, row: dict) -> pd.DataFrame:
    """Persist the latest status for each trial without discarding other trials."""
    frame = load_progress(path)
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame = frame.drop_duplicates(subset=["trial_id"], keep="last")
    common.atomic_csv(path, frame)
    return frame


def run_trial(
    cfg: dict,
    run_dir: Path,
    fingerprint: str,
    architecture: str,
    horizon: int,
    hp: dict,
    progress_path: Path,
    force: bool = False,
    seed: int | None = None,
    phase: str = "tuning",
):
    """Reuse a verified completed fit or restart an interrupted fit from its original seed."""
    seed = int(cfg["model"]["seed_tuning"] if seed is None else seed)
    trial_id = common.stable_hash(
        {"phase": phase, "fit": _row_key(fingerprint, architecture, horizon, hp, seed)}
    )
    progress = load_progress(progress_path)
    if not force and not progress.empty:
        done = progress[
            (progress["trial_id"] == trial_id) & (progress["status"] == "success")
        ]
        if not done.empty:
            row = done.iloc[-1].to_dict()
            state.verify_artifacts(run_dir, row)
            return None, row
    # A stopped fit is rerun from the original seed. RNG/dropout and EarlyStopping
    # state are not reconstructed from a best-only checkpoint; never pretend an
    # epoch continuation is scientifically identical. Keep every attempt.
    attempt_dir = state.new_attempt(run_dir, phase, trial_id)
    sequence = common.load_sequence(run_dir, horizon, splits=["train", "val"])
    row = {
        "trial_id": trial_id,
        "fingerprint": fingerprint,
        "status": "running",
        "phase": phase,
        "architecture": architecture,
        "horizon_slots": horizon,
        "horizon_minutes": horizon * 5,
        "seed": seed,
        **hp,
        "started_at_utc": state.utc_now(),
        "attempt_dir": str(attempt_dir.relative_to(run_dir)),
        "error": "",
    }
    common.atomic_json(attempt_dir / "status.json", row)
    append_result(progress_path, row)
    try:
        model, details = train_once(
            cfg, sequence, architecture, hp, seed, attempt_dir=attempt_dir
        )
        model_path = attempt_dir / "model.keras"
        model.save(model_path)
        row.update(details)
        row.update(
            {
                "status": "success",
                "finished_at_utc": state.utc_now(),
                "model_path": str(model_path.relative_to(run_dir)),
                "model_sha256": common.sha256_file(model_path),
                "history_path": str((attempt_dir / "history.csv").relative_to(run_dir)),
                "history_sha256": common.sha256_file(attempt_dir / "history.csv"),
                "checkpoint_path": str(
                    (attempt_dir / "best.keras").relative_to(run_dir)
                ),
                "checkpoint_sha256": common.sha256_file(attempt_dir / "best.keras"),
            }
        )
        common.atomic_json(attempt_dir / "status.json", row)
        append_result(progress_path, row)
        return model, row
    except BaseException as exc:
        row.update(
            {
                "status": (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                ),
                "finished_at_utc": state.utc_now(),
                "error": repr(exc),
            }
        )
        common.atomic_json(attempt_dir / "status.json", row)
        append_result(progress_path, row)
        raise


def shared_selection(progress: pd.DataFrame, cfg: dict, fingerprint: str) -> dict:
    """Select shared LSTM/GRU hyperparameters from tuning-validation MAE only."""
    if progress.empty:
        return {}
    successful = progress[
        (progress["status"] == "success")
        & (progress["fingerprint"] == fingerprint)
        & (progress["seed"] == int(cfg["model"]["seed_tuning"]))
        & (progress["phase"] == "tuning")
    ].copy()
    selections = {}
    hp_columns = [
        "units",
        "layers",
        "dropout",
        "learning_rate",
        "batch_size",
        "config_id",
    ]
    for horizon in cfg["preprocessing"]["horizons"]:
        subset = successful[successful["horizon_slots"] == horizon]
        pivot = subset.pivot_table(
            index="config_id",
            columns="architecture",
            values="val_mae_celsius",
            aggfunc="last",
        )
        if not {"lstm", "gru"}.issubset(pivot.columns):
            continue
        pivot = pivot.dropna(subset=["lstm", "gru"])
        if pivot.empty:
            continue
        pivot["mean"] = (pivot["lstm"] + pivot["gru"]) / 2
        pivot["worst"] = pivot[["lstm", "gru"]].max(axis=1)
        best_id = (
            pivot.reset_index()
            .sort_values(["mean", "worst", "config_id"])
            .iloc[0]["config_id"]
        )
        source = subset[subset["config_id"] == best_id].iloc[0]
        hp = {
            column: (
                source[column].item()
                if hasattr(source[column], "item")
                else source[column]
            )
            for column in hp_columns
        }
        winner = (
            "lstm" if pivot.loc[best_id, "lstm"] <= pivot.loc[best_id, "gru"] else "gru"
        )
        selections[str(horizon)] = {
            "horizon_minutes": horizon * 5,
            "shared_hyperparameters": hp,
            "lstm_val_mae": float(pivot.loc[best_id, "lstm"]),
            "gru_val_mae": float(pivot.loc[best_id, "gru"]),
            "pair_mean_val_mae": float(pivot.loc[best_id, "mean"]),
            "deployment_winner": winner,
            "seed": int(cfg["model"]["seed_tuning"]),
        }
    return selections


def final_training(cfg: dict, run_dir: Path, fingerprint: str) -> None:
    """Fit locked selections for all final seeds and export auditable prediction artifacts."""
    import tensorflow as tf

    seeds = [int(seed) for seed in cfg["model"]["final_seeds"]]
    if seeds != [1, 2, 3]:
        raise ValueError("Protokol final membutuhkan final_seeds tepat [1, 2, 3].")
    path = run_dir / "results" / "selected_shared_configs.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("run_fingerprint") != fingerprint:
        raise RuntimeError(
            "Selection berasal dari run/runtime berbeda; jangan reuse tuning Metal."
        )
    selections = selection["selections"]
    if set(selections) != {str(h) for h in cfg["preprocessing"]["horizons"]}:
        raise RuntimeError("Selection belum lengkap untuk seluruh horizon.")
    lock_path = run_dir / "state" / "locked_final_selection.json"
    state.lock_json(lock_path, selection)
    progress_path = run_dir / "results" / "final_training_progress.csv"
    manifest = []
    predictions = []
    for horizon in cfg["preprocessing"]["horizons"]:
        hp = selections[str(horizon)]["shared_hyperparameters"]
        for seed in seeds:
            payload = {}
            for architecture in cfg["model"]["architectures"]:
                model, row = run_trial(
                    cfg,
                    run_dir,
                    fingerprint,
                    architecture,
                    horizon,
                    hp,
                    progress_path,
                    seed=seed,
                    phase="final",
                )
                if model is None:
                    model = tf.keras.models.load_model(run_dir / row["model_path"])
                # Test data is loaded only after the fit and config choice are complete.
                sequence = common.load_sequence(run_dir, horizon)
                payload.update(
                    {
                        "y_true": sequence["ycel_test"],
                        "t": sequence["t_test"],
                        "issue": sequence["issue_test"],
                        "y_true_val": sequence["ycel_val"],
                        "t_val": sequence["t_val"],
                        "issue_val": sequence["issue_val"],
                        "run_fingerprint": np.asarray(fingerprint),
                        "seed": np.asarray(seed),
                    }
                )
                for split in ["val", "test"]:
                    payload[f"actual_at_issue_{split}"] = sequence[
                        f"actual_at_issue_{split}"
                    ]
                    pred = model.predict(sequence[f"X_{split}"], verbose=0).reshape(-1)
                    if not np.isfinite(pred).all():
                        raise RuntimeError("Prediksi non-finite; evaluasi dihentikan.")
                    payload[f"{architecture}_{split}"] = common.inverse_target(
                        run_dir, pred
                    )
                canonical = (
                    run_dir
                    / "models"
                    / f"{architecture}_h{horizon * 5}_seed{seed}.keras"
                )
                state.copy_verified(
                    run_dir / row["model_path"], canonical, row["model_sha256"]
                )
                manifest.append(
                    {
                        "architecture": architecture,
                        "horizon_slots": horizon,
                        "horizon_minutes": horizon * 5,
                        "seed": seed,
                        "model_path": str(canonical.relative_to(run_dir)),
                        "model_sha256": row["model_sha256"],
                        "trial_id": row["trial_id"],
                        "val_mae_celsius": row["val_mae_celsius"],
                        "duration_seconds": row["duration_seconds"],
                        "history_path": row["history_path"],
                        "attempt_dir": row["attempt_dir"],
                    }
                )
                del model, sequence
                tf.keras.backend.clear_session()
                gc.collect()
            prediction_path = (
                run_dir / "results" / f"neural_predictions_h{horizon}_seed{seed}.npz"
            )
            state.atomic_npz(prediction_path, payload)
            predictions.append(
                {
                    "path": str(prediction_path.relative_to(run_dir)),
                    "sha256": common.sha256_file(prediction_path),
                    "horizon_slots": horizon,
                    "seed": seed,
                }
            )
    common.atomic_json(
        run_dir / "results" / "final_models_manifest.json",
        {
            "run_fingerprint": fingerprint,
            "seeds": seeds,
            "models": manifest,
            "predictions": predictions,
            "deployment_selection": "PENDING_PREDECLARED_SEED_OR_ENSEMBLE_POLICY",
            "selection_basis": "shared_configuration_from_tuning_validation_only",
        },
    )


def smoke(cfg: dict, run_dir: Path) -> None:
    """Check a small fit and save/load cycle before running the full experiment."""
    import tensorflow as tf

    hp = dict(cfg["model"]["smoke"])
    epochs = int(hp.pop("epochs"))
    hp["config_id"] = "smoke_u32_l1_d0p1"
    sequence = common.load_sequence(
        run_dir, max(cfg["preprocessing"]["horizons"]), splits=["train", "val"]
    )
    results = []
    for architecture in cfg["model"]["architectures"]:
        attempt_dir = state.new_attempt(run_dir, "smoke", architecture)
        model, row = train_once(
            cfg,
            sequence,
            architecture,
            hp,
            cfg["model"]["seed_tuning"],
            epochs=epochs,
            subset=256,
            attempt_dir=attempt_dir,
        )
        path = run_dir / "state" / f"smoke_{architecture}.keras"
        model.save(path)
        tf.keras.models.load_model(path)
        results.append({"architecture": architecture, **row, "model_save_load": True})
        tf.keras.backend.clear_session()
    common.atomic_json(run_dir / "logs" / "smoke_test.json", results)


def benchmark(cfg: dict, run_dir: Path, fingerprint: str, progress_path: Path) -> dict:
    """Estimate total tuning duration from representative, reusable tuning trials."""
    candidates = grid(cfg)
    ids = {hp["config_id"]: hp for hp in candidates}
    selected = list(
        {
            hp["config_id"]: hp
            for hp in [candidates[len(candidates) // 2], candidates[-1]]
        }.values()
    )
    durations = []
    for horizon in cfg["preprocessing"]["horizons"]:
        for hp in selected:
            for architecture in cfg["model"]["architectures"]:
                model, row = run_trial(
                    cfg, run_dir, fingerprint, architecture, horizon, hp, progress_path
                )
                durations.append(float(row["duration_seconds"]))
                if model is not None:
                    del model
                gc.collect()
    estimate = float(
        np.mean(durations)
        * len(candidates)
        * len(cfg["model"]["architectures"])
        * len(cfg["preprocessing"]["horizons"])
        * cfg["model"]["benchmark_safety_factor"]
        / 3600
    )
    result = {
        "sample_trials": len(durations),
        "mean_seconds": float(np.mean(durations)),
        "safety_factor": cfg["model"]["benchmark_safety_factor"],
        "estimated_total_hours": estimate,
        "limit_hours": cfg["model"]["benchmark_max_hours"],
        "may_continue": estimate <= cfg["model"]["benchmark_max_hours"],
    }
    common.atomic_json(run_dir / "logs" / "benchmark.json", result)
    return result


def tune(cfg: dict, run_dir: Path, fingerprint: str, progress_path: Path) -> None:
    """Complete the configured grid and lock one shared configuration per horizon."""
    for horizon in cfg["preprocessing"]["horizons"]:
        for hp in grid(cfg):
            for architecture in cfg["model"]["architectures"]:
                model, _ = run_trial(
                    cfg, run_dir, fingerprint, architecture, horizon, hp, progress_path
                )
                if model is not None:
                    del model
                gc.collect()
    progress = load_progress(progress_path)
    selections = shared_selection(progress, cfg, fingerprint)
    if len(selections) != len(cfg["preprocessing"]["horizons"]):
        raise RuntimeError(
            "Tuning belum lengkap; konfigurasi bersama belum tersedia untuk semua horizon."
        )
    state.lock_json(
        run_dir / "results" / "selected_shared_configs.json",
        {
            "run_fingerprint": fingerprint,
            "selection_basis": "tuning_validation_pair_mean_mae",
            "selections": selections,
        },
    )


def main() -> None:
    """Validate prepared inputs, bind immutable run identity, and dispatch one phase."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--mode", choices=["smoke", "benchmark", "tune", "final"], required=True
    )
    args = parser.parse_args()
    cfg = common.load_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    cfg["_run_dir"] = str(run_dir)
    prepared_cfg = common.load_config(run_dir / "config" / "config.yaml")
    stable_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    prepared_cfg = {k: v for k, v in prepared_cfg.items() if not k.startswith("_")}
    if common.stable_hash(stable_cfg) != common.stable_hash(prepared_cfg):
        raise RuntimeError(
            "Config saat ini berbeda dari config praproses; buat run baru."
        )
    metadata = json.loads((run_dir / "dataset" / "metadata.json").read_text())
    expected = cfg["data"].get("expected_sha256")
    if expected and metadata["sha256"] != expected:
        raise RuntimeError("Dataset run tidak cocok checksum final.")
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG harus deterministik sebelum TensorFlow import."
        )
    if not metadata.get("final_window_complete", False) and args.mode != "smoke":
        raise RuntimeError(
            "Data incomplete tidak dapat dipakai untuk training final/tuning."
        )
    fingerprint = state.bind_run_identity(
        run_dir, cfg, metadata, environment_snapshot(run_dir, cfg)
    )
    progress_path = run_dir / "results" / "tuning_progress.csv"
    if args.mode == "smoke":
        smoke(cfg, run_dir)
    elif args.mode == "benchmark":
        print(json.dumps(benchmark(cfg, run_dir, fingerprint, progress_path), indent=2))
    elif args.mode == "tune":
        tune(cfg, run_dir, fingerprint, progress_path)
    else:
        final_training(cfg, run_dir, fingerprint)


if __name__ == "__main__":
    with state.process_lock():
        main()
