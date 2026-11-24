"""Verify and extract a Mac execution package into a new PC directory.

Usage: python unpack_runtime_pc.py --archive PACKAGE.zip --target NEW_DIRECTORY
Extraction never launches training or overwrites an existing execution directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


def safe_member(name: str) -> bool:
    """Accept relative portable paths and reject traversal or Windows drives."""
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and ":" not in name and "\\" not in name


def unpack(archive_path: Path, target: Path) -> int:
    """Validate source checksums before writing an immutable execution copy."""
    if target.exists():
        raise FileExistsError(f"Folder eksekusi sudah ada: {target}")
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if not safe_member(entry.filename) or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Path arsip tidak aman: {entry.filename}")
        manifest = json.loads(archive.read("package_manifest.json"))
        for relative, expected in manifest.items():
            if not safe_member(relative):
                raise ValueError(f"Path manifest tidak aman: {relative}")
            if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                raise RuntimeError(f"Checksum kode/config tidak cocok: {relative}")
        target.mkdir(parents=True)
        archive.extractall(target)
    return len(manifest)


def main() -> None:
    """Parse explicit package locations and report extraction without fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    count = unpack(args.archive, args.target)
    print(f"Salinan eksekusi terverifikasi: {args.target} ({count} file kode/config)")
    print("Training belum dimulai. Buka 02_pengendali/02_pengendali_training.ipynb.")


if __name__ == "__main__":
    main()
