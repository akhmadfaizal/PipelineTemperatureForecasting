from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / 'fungsi_bersama'))
sys.path.insert(0, str(PIPELINE_ROOT / '04_training'))
import common


def main() -> None:
    parser = argparse.ArgumentParser(description="Praproses PC final; dataset terkunci dan run tidak ditimpa.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--allow-incomplete", action="store_true", help="Hanya untuk smoke test.")
    args = parser.parse_args()
    cfg = common.load_config(args.config)
    run_dir = common.make_run_dir(cfg, args.run_dir)
    meta = common.prepare_dataset(cfg, run_dir, allow_incomplete=args.allow_incomplete)
    print(f"RUN_DIR={run_dir}")
    print(f"Dataset: {meta['source']}")
    print(f"Rentang: {meta['raw_first']} sampai {meta['raw_last']}")
    print(f"Jendela final lengkap: {meta['final_window_complete']}")
    print("Jumlah sampel:")
    print((run_dir / "results" / "sequence_counts.csv").read_text())


if __name__ == "__main__":
    main()
