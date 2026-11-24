"""Detached WSL training jobs with explicit phase/status; no automatic next phase."""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / "fungsi_bersama"))
sys.path.insert(0, str(PIPELINE_ROOT / "04_training"))
import common

PHASES = {"smoke", "benchmark", "tune", "final"}


def now():
    """Return a timezone-aware UTC timestamp for job status records."""
    return datetime.now(timezone.utc).isoformat()


def process_token(pid):
    """Identify a live Linux process by boot ID and start time, avoiding PID reuse."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        if stat[0] == "Z":
            return None
        return (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + stat[19]
        )
    except (FileNotFoundError, ProcessLookupError):
        return None


def start(phase, run_dir, config):
    """Launch one detached phase and preserve its console output in a new job directory."""
    if phase not in PHASES:
        raise ValueError("Unknown training phase")
    jobs = run_dir / "logs/jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    for path in jobs.glob("*/status.json"):
        record = json.loads(path.read_text())
        if (
            record.get("status") == "running"
            and record.get("process_token")
            and process_token(record["pid"]) == record["process_token"]
        ):
            raise RuntimeError("A training job is already running for this run.")
    job = jobs / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + phase
        + "_"
        + uuid.uuid4().hex[:8]
    )
    job.mkdir()
    common.atomic_json(
        job / "status.json",
        {"status": "starting", "phase": phase, "requested_at": now()},
    )
    with (job / "console.log").open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--phase",
                phase,
                "--run-dir",
                str(run_dir),
                "--config",
                str(config),
                "--job-dir",
                str(job),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    result = {
        "phase": phase,
        "worker_pid": process.pid,
        "job_dir": str(job),
        "console_log": str(job / "console.log"),
    }
    print(json.dumps(result, indent=2))
    return result


def worker(phase, run_dir, config, job):
    """Run one training phase and always persist its terminal status."""
    record = {
        "status": "running",
        "phase": phase,
        "pid": os.getpid(),
        "process_token": process_token(os.getpid()),
        "started_at": now(),
    }
    common.atomic_json(job / "status.json", record)
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.setdefault("PYTHONHASHSEED", "1")
    try:
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("04_train.py")),
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
            "--mode",
            phase,
        ]
        result = subprocess.run(command, cwd=PIPELINE_ROOT, env=environment)
        record.update(
            status="completed" if result.returncode == 0 else "failed",
            exit_code=result.returncode,
            finished_at=now(),
        )
    except BaseException as exc:
        record.update(status="failed", error=repr(exc), finished_at=now())
        raise
    finally:
        common.atomic_json(job / "status.json", record)


def main():
    """Dispatch a phase launcher or its internal worker from command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "worker"])
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--job-dir")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    config = Path(args.config).resolve()
    if args.action == "start":
        start(args.phase, run, config)
    else:
        worker(args.phase, run, config, Path(args.job_dir).resolve())


if __name__ == "__main__":
    main()
