#!/usr/bin/env python3
"""
s08_package_and_audit.py - provenance diff, artifact inventory and ZIP packaging (rewritten in remediation).

* manifests/provenance_changes.json : every file of the ORIGINAL output tree classified as unchanged / modified /
  superseded(moved) / removed, plus files added by the remediation (SHA-256 based; original hashes in
  manifests/original_output_sha256.txt were taken before any change).
* manifests/artifacts_inventory.json : path, bytes, sha256 of every packaged file.
* <zip>: complete package in the spec layout (source/ config/ tests/ results/ predictions/ models/ figures/ reports/
  manifests/ logs/ + README.md, phase5_run.py, requirements.txt, audit_summary.json).  Large immutable Phase 2-4
  inputs and the resumable checkpoints/ directory are NOT included.
audit_summary.json is produced by s11 (from result files); this script does not write any metric.
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EXCLUDE_DIRS = {"checkpoints", "__pycache__", ".pytest_cache", "backup_pre_remediation"}
EXCLUDE_FILES = {".DS_Store"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def walk(root: Path):
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in EXCLUDE_DIRS)
        for f in sorted(files):
            if f in EXCLUDE_FILES or f.endswith((".pyc", ".zip")):
                continue
            yield Path(d) / f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--zip-path", default=None)
    a = ap.parse_args(argv)
    root = Path(a.output_dir).resolve()
    zip_path = Path(a.zip_path) if a.zip_path else root.parent / "Phase5_SVM_Kernel_RFF_Results.zip"
    (root / "manifests").mkdir(exist_ok=True)

    # ---- provenance diff vs original tree
    orig = {}
    f = root / "manifests" / "original_output_sha256.txt"
    if f.exists():
        for line in f.read_text().splitlines():
            h, p = line.split(maxsplit=1)
            orig[p.strip().lstrip("./")] = h
    cur = {str(p.relative_to(root)): sha(p) for p in walk(root)
           if not str(p.relative_to(root)).startswith(("superseded_pre_audit/",))}
    sup = {str(p.relative_to(root / "superseded_pre_audit")) for p in walk(root / "superseded_pre_audit")} if (root / "superseded_pre_audit").exists() else set()
    changes = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "unchanged": [], "modified": [], "added": [], "removed_or_moved": []}
    for p, h in sorted(cur.items()):
        if p == "manifests/original_output_sha256.txt":
            continue
        if p in orig:
            (changes["unchanged"] if orig[p] == h else changes["modified"]).append(p)
        else:
            changes["added"].append(p)
    for p in sorted(orig):
        if p not in cur and not p.startswith("checkpoints/"):
            changes["removed_or_moved"].append(p)
    changes["note"] = ("Originals of modified/removed files are kept under superseded_pre_audit/. 'added' includes regenerated files whose name did not exist "
                       "before. checkpoints/ (original Stage A/C pickles) are untouched in the working tree but not packaged.")
    changes["counts"] = {k: len(v) for k, v in changes.items() if isinstance(v, list)}
    (root / "manifests" / "provenance_changes.json").write_text(json.dumps(changes, indent=2))

    # ---- inventory
    def build_inventory():
        return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha(p),
                 "category": p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else "root"}
                for p in walk(root) if p.name != "artifacts_inventory.json"]
    inv = build_inventory()
    (root / "manifests" / "artifacts_inventory.json").write_text(json.dumps(inv, indent=1))

    # ---- zip (inventory file itself included; its listed files are all present)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in walk(root):
            zf.write(p, arcname=str(p.relative_to(root)))
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    if bad:
        print("[ERROR] corrupted member:", bad)
        return 1
    print(f"ZIP written: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {len(names)} files, CRC ok); changes: {changes['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
