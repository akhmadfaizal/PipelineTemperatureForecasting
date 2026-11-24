"""Small Jupyter controls; no preprocessing, model fitting or selection logic."""

from __future__ import annotations

import csv
import io
import shlex
import subprocess
from pathlib import Path


def show_status(run_dir: Path, log_characters: int = 4500) -> None:
    """Read run status and fit counts without starting or changing a job."""
    status = run_dir / "logs/pipeline_status.json"
    print(status.read_text() if status.exists() else "Run belum dimulai")
    for name in ("sequence_counts.csv", "tuning_progress.csv", "final_training_progress.csv"):
        path = run_dir / "results" / name
        if not path.exists():
            continue
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        if name == "sequence_counts.csv":
            print(name, "jumlah baris split/horizon:", len(rows))
        else:
            successes = sum(row.get("status") == "success" for row in rows)
            print(name, "fit sukses:", successes, "/", len(rows))
    benchmark = run_dir / "logs/benchmark.json"
    if benchmark.exists():
        print("Benchmark:", benchmark.read_text())
    console = run_dir / "logs/pipeline_console.log"
    if console.exists():
        print(console.read_text(encoding="utf-8", errors="replace")[-log_characters:])


def run_wsl_script(
    pipeline_dir: str, python_path: str, script: str, *arguments: str, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Invoke the explicit WSL interpreter and surface failures to Jupyter."""
    command = "cd " + shlex.quote(pipeline_dir) + " && " + shlex.join(
        [python_path, "-u", script, *arguments]
    )
    result = subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-c", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Perintah WSL gagal ({result.returncode}): {result.stderr[-6000:]}")
    return result
