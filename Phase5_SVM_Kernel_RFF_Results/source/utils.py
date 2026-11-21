"""
utils.py – Shared utilities for Phase 5 integrity gate and downstream scripts.
"""
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class GateLogger:
    """Simple logger that writes to both stdout and an optional log file."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self._file = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(log_path, "w", buffering=1)

    def _write(self, line: str):
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        full = f"[{ts}] {line}"
        print(full, flush=True)
        if self._file:
            self._file.write(full + "\n")

    def info(self, msg: str):
        self._write(f"INFO  {msg}")

    def ok(self, msg: str):
        self._write(f"PASS  {msg}")

    def warn(self, msg: str):
        self._write(f"WARN  {msg}")

    def error(self, msg: str):
        self._write(f"FAIL  {msg}")

    def section(self, title: str):
        bar = "=" * 70
        self._write(bar)
        self._write(f"SECTION  {title}")
        self._write(bar)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# SHA-256 utilities
# ---------------------------------------------------------------------------

def compute_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Return lowercase hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# NPZ helpers
# ---------------------------------------------------------------------------

def load_npz_metadata_only(
    path: str,
    meta_keys: Tuple[str, ...] = ("y", "target_timestamp", "current_state", "current_pm25"),
) -> Dict[str, np.ndarray]:
    """
    Load only the specified metadata arrays from an NPZ file, skipping large
    feature arrays.  Uses NpzFile lazy-loading to avoid reading 'F' into RAM.
    """
    with np.load(path, allow_pickle=True) as data:
        result = {}
        for key in meta_keys:
            if key in data:
                result[key] = data[key]
            else:
                raise KeyError(f"Key '{key}' not found in {path}; available: {list(data.keys())}")
    return result


def load_npz_all_keys(path: str) -> Dict[str, np.ndarray]:
    """Load all arrays from an NPZ; use sparingly for large files."""
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def npz_key_list(path: str) -> List[str]:
    """Return list of keys in an NPZ without loading array data."""
    with np.load(path, allow_pickle=True) as data:
        return list(data.files)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def to_datetime64_ns(arr: np.ndarray) -> np.ndarray:
    """Coerce an array to datetime64[ns]."""
    if arr.dtype == np.dtype("datetime64[ns]"):
        return arr
    return arr.astype("datetime64[ns]")


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def build_provenance_manifest(paths: List[str], base_dir: str) -> Dict:
    """Compute SHA-256 for each file and return a provenance dict."""
    manifest = {}
    for p in paths:
        rel = os.path.relpath(p, base_dir)
        size = os.path.getsize(p)
        sha = compute_sha256(p)
        manifest[rel] = {"sha256": sha, "bytes": size, "path": p}
    return manifest
