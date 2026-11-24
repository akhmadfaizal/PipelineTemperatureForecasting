"""Training provenance and safe resume; no TensorFlow import needed to audit state."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / "fungsi_bersama"))
sys.path.insert(0, str(PIPELINE_ROOT / "04_training"))
import common


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp for training provenance."""
    return datetime.now(timezone.utc).isoformat()


def lock_json(path: Path, value: dict) -> None:
    """An existing identity/selection is immutable, never silently replaced."""
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if common.stable_hash(old) != common.stable_hash(value):
            raise RuntimeError(
                f"Identitas/selection berubah: {path}. Gunakan run baru."
            )
    else:
        common.atomic_json(path, value)


def bind_run_identity(
    run_dir: Path, cfg: dict, metadata: dict, environment: dict
) -> str:
    """Lock config, prepared files, source hashes, and runtime into one run fingerprint."""
    source_dir = PIPELINE_ROOT
    sources = {
        str(p.relative_to(source_dir)): common.sha256_file(p)
        for folder in [
            "03_praproses",
            "04_training",
            "05_baseline",
            "06_evaluasi",
            "fungsi_bersama",
        ]
        for p in sorted((source_dir / folder).glob("*.py"))
    }
    prepared = {
        str(p.relative_to(run_dir)): common.sha256_file(p)
        for p in sorted((run_dir / "data").glob("*"))
        if p.is_file()
    }
    stable_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    stable_env = {
        k: v
        for k, v in environment.items()
        if k not in {"captured_at_utc", "same_process_as_model_fit"}
    }
    payload = {
        "schema": 2,
        "dataset_source": metadata.get("source"),
        "dataset_size_bytes": metadata.get("source_size_bytes"),
        "preprocessing_fingerprint": metadata["fingerprint"],
        "prepared_files": prepared,
        "config": stable_cfg,
        "sources": sources,
        "environment": stable_env,
        "resume_policy": "completed_fit_hash_verified; interrupted_fit_restart_seed",
    }
    fingerprint = common.stable_hash(payload)
    identity = {"fingerprint": fingerprint, **payload}
    lock_json(run_dir / "state" / "training_identity.json", identity)
    snapshot = run_dir / "logs" / "training_source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, digest in sources.items():
        copy_verified(source_dir / name, snapshot / name, digest)
    return fingerprint


def new_attempt(run_dir: Path, phase: str, trial_id: str) -> Path:
    """Create a unique attempt directory so interrupted fits retain their evidence."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = (
        run_dir / "logs" / "fits" / phase / trial_id / f"{stamp}_{uuid.uuid4().hex[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def verify_artifacts(run_dir: Path, row: dict) -> None:
    """Reject missing, escaped, or changed artifacts before reusing a completed fit."""
    for prefix in ["model", "history", "checkpoint"]:
        relative = Path(str(row.get(f"{prefix}_path", "")))
        if not str(relative) or relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Path artefak progress tidak valid.")
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()) or not path.is_file():
            raise RuntimeError(f"Artefak completed fit hilang: {path}")
        if common.sha256_file(path) != row.get(f"{prefix}_sha256"):
            raise RuntimeError(f"Checksum artefak berubah: {path}; gunakan run baru.")


def copy_verified(source: Path, target: Path, digest: str) -> None:
    """Copy an artifact atomically without replacing a different existing artifact."""
    if common.sha256_file(source) != digest:
        raise RuntimeError(f"Sumber berubah: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if common.sha256_file(target) != digest:
            raise RuntimeError(f"Artefak tujuan berubah: {target}; tidak ditimpa.")
        return
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    if common.sha256_file(temporary) != digest:
        temporary.unlink()
        raise RuntimeError("Copy artefak gagal checksum.")
    os.replace(temporary, target)


def atomic_npz(path: Path, payload: dict) -> None:
    """Write compressed prediction arrays through an atomic file replacement."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


@contextmanager
def process_lock():
    """OS releases flock after a crash; only one writer per run (Linux/WSL)."""
    import fcntl

    parser_run = next(
        (
            sys.argv[i + 1]
            for i, value in enumerate(sys.argv[:-1])
            if value == "--run-dir"
        ),
        None,
    )
    if parser_run is None:
        parser_run = next(
            (
                value.split("=", 1)[1]
                for value in sys.argv
                if value.startswith("--run-dir=")
            ),
            None,
        )
    if parser_run is None:
        yield  # argparse will explain the missing required argument.
        return
    run_dir = Path(parser_run).resolve()
    (run_dir / "state").mkdir(parents=True, exist_ok=True)
    with (run_dir / "state" / "training.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Proses training lain sedang menulis run ini.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
