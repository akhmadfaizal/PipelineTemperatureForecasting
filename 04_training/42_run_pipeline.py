"""Run the Mac-authored 42-day pipeline as a detached WSL process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fungsi_bersama"))
import common


def validate_config(cfg):
    """Require the locked 42-day split, final seeds, and GPU policy before launch."""
    import pandas as pd

    splits = cfg["preprocessing"]["split"]
    expected = {
        "train": ("2026-08-20", "2026-09-17", 28),
        "val": ("2026-09-17", "2026-09-24", 7),
        "test": ("2026-09-24", "2026-10-01", 7),
    }
    for name, (left, right, days) in expected.items():
        actual = [pd.Timestamp(x) for x in splits[name]]
        required = [pd.Timestamp(x, tz="Asia/Jakarta") for x in (left, right)]
        if actual != required or (actual[1] - actual[0]).days != days:
            raise ValueError("Config is not the locked 42-day split.")
    if pd.Timestamp(cfg["data"]["mulai"]) != pd.Timestamp(splits["train"][0]):
        raise ValueError("Dataset start differs from train start.")
    if pd.Timestamp(cfg["data"]["akhir_eksklusif"]) != pd.Timestamp(splits["test"][1]):
        raise ValueError("Dataset end differs from test end.")
    if cfg["model"]["final_seeds"] != [1, 2, 3]:
        raise ValueError("Final seeds must be 1, 2, 3.")
    if not cfg.get("runtime", {}).get("require_gpu"):
        raise ValueError("GPU must be required for this PC experiment.")


def write_status(run, status, **fields):
    """Persist and print the current controller status with UTC provenance."""
    result = {
        "status": status,
        "pid": os.getpid(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run),
        "code_authority": "Mac",
        **fields,
    }
    common.atomic_json(run / "logs" / "pipeline_status.json", result)
    print(json.dumps(result), flush=True)


def worker(run, config):
    """Run preparation and training phases under one lock, then verify the final manifest."""
    os.chdir(ROOT)
    cfg = common.load_config(config)
    validate_config(cfg)
    run.mkdir(parents=True, exist_ok=True)
    with (run / "pipeline.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = run / "logs" / "pipeline_status.json"
        if (
            status_path.exists()
            and json.loads(status_path.read_text()).get("status")
            == "FINAL_TRAINING_COMPLETED"
        ):
            raise RuntimeError("This run is already complete; do not launch it again.")
        environment = os.environ.copy()
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["PYTHONHASHSEED"] = "1"
        try:
            if not (run / "dataset" / "metadata.json").exists():
                write_status(run, "RUNNING_PREPROCESSING")
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "03_praproses/02_prepare.py",
                        "--config",
                        str(config),
                        "--run-dir",
                        str(run),
                    ],
                    env=environment,
                    check=True,
                )
            for phase in ("smoke", "benchmark", "tune", "final"):
                write_status(run, "RUNNING_" + phase.upper(), phase=phase)
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "04_training/04_train.py",
                        "--config",
                        str(config),
                        "--run-dir",
                        str(run),
                        "--mode",
                        phase,
                    ],
                    env=environment,
                    check=True,
                )
                if phase == "benchmark":
                    estimate = json.loads((run / "logs" / "benchmark.json").read_text())
                    if not estimate["may_continue"]:
                        write_status(run, "WAITING_DURATION_REVIEW", benchmark=estimate)
                        return
            manifest = json.loads(
                (run / "results" / "final_models_manifest.json").read_text()
            )
            if len(manifest["models"]) != 18:
                raise RuntimeError("Final manifest does not contain 18 models.")
            for item in manifest["models"]:
                path = run / item["model_path"]
                if common.sha256_file(path) != item["model_sha256"]:
                    raise RuntimeError(
                        "Final model integrity check failed: " + str(path)
                    )
            write_status(
                run,
                "FINAL_TRAINING_COMPLETED",
                models=18,
                evaluation="NOT_STARTED",
                deployment="NOT_CHANGED",
            )
        except BaseException as exc:
            write_status(run, "FAILED", error=repr(exc))
            raise


def main():
    """Start a fresh detached run, execute its worker, or read its saved status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "worker", "status"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    run = Path(args.run_dir).resolve()
    config = Path(args.config).resolve()
    if args.action == "status":
        print((run / "logs" / "pipeline_status.json").read_text())
    elif args.action == "worker":
        worker(run, config)
    else:
        validate_config(common.load_config(config))
        # A fresh run is reserved atomically, so repeated clicks cannot start it twice.
        run.mkdir(parents=True, exist_ok=False)
        (run / "logs").mkdir()
        write_status(run, "STARTING")
        with (run / "logs" / "pipeline_console.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "worker",
                    "--config",
                    str(config),
                    "--run-dir",
                    str(run),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(
            json.dumps({"controller_pid": process.pid, "run_dir": str(run)}, indent=2)
        )


if __name__ == "__main__":
    main()
