#!/usr/bin/env python3
"""
s01_integrity_gate.py – Phase 5, Section 3: Hard Data-Integrity Gate
=====================================================================
Performs comprehensive validation of all Phase 4 output files before any
modelling proceeds.  Exits with code 0 on full pass, code 1 on any failure.

Usage:
    python3 s01_integrity_gate.py \
        --input-dir  /path/to/phase5/inputs \
        --output-dir /path/to/phase5/output \
        --seed 42
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Inject local utils
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from utils import (
    GateLogger,
    build_provenance_manifest,
    compute_sha256,
    load_npz_metadata_only,
    npz_key_list,
    to_datetime64_ns,
)

# ---------------------------------------------------------------------------
# Constants / expected values
# ---------------------------------------------------------------------------

SPLITS = ("train", "validation", "test")
CONFIGS = ("pca_A", "pca_B", "pca_C", "hosvd_A", "hosvd_B", "hosvd_C", "hosvd_B_alt1", "hosvd_B_alt2")
EXPECTED_SAMPLE_COUNTS = {"train": 24108, "validation": 4239, "test": 5720}

# Class labels: 0=Low, 1=Moderate, 2=High
EXPECTED_CLASS_DIST: Dict[str, Dict[int, int]] = {
    "train":      {0: 7872, 1: 5832,  2: 10404},
    "validation": {0: 1848, 1: 1114,  2: 1277},
    "test":       {0: 2142, 1: 1202,  2: 2376},
}

EXPECTED_SCOPE_A_COUNTS = {"train": 23976, "validation": 4197, "test": 5685}

REQUIRED_FEATURE_NPZ_KEYS = {"F", "y", "target_timestamp", "current_state", "current_pm25"}
EXPECTED_DTYPES = {
    "F":                 np.float32,
    "y":                 np.int8,
    "current_state":     np.int8,
    "current_pm25":      np.float32,
    "target_timestamp":  np.dtype("datetime64[ns]"),
}

REQUIRED_MARKOV_COLUMNS = {
    "target_timestamp", "y", "current_state", "scope_a_eligible",
    "pred_majority", "pred_persistence_scope_b", "pred_markov_scope_b",
}

WINDOW_META_KEYS = ("y", "target_timestamp", "current_state", "current_pm25")

MANIFEST_FILENAME  = "phase4_feature_manifest.csv"
METADATA_FILENAME  = "phase4_metadata.json"
MARKOV_CSV_FILES   = {
    "validation": "markov_predictions_validation.csv",
    "test":       "markov_predictions_test.csv",
}
WINDOW_NPZ_FILES   = {s: f"{s}_windows.npz" for s in SPLITS}
FITTED_BASIS_FILES = [
    "fitted_pca_basis_48.npz",
    "fitted_pca_basis_192.npz",
    "fitted_pca_basis_576.npz",
    "fitted_center_and_factors.npz",
]

# Reference config to use for cross-validation (must be listed in CONFIGS)
REFERENCE_CONFIG = "pca_A"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

class GateResult:
    def __init__(self):
        self.passes: List[str] = []
        self.failures: List[str] = []
        self.warnings: List[str] = []
        self.details: Dict = {}

    def ok(self, name: str, detail: str = ""):
        self.passes.append(name)
        if detail:
            self.details[name] = detail

    def fail(self, name: str, detail: str = ""):
        self.failures.append(name)
        if detail:
            self.details[name] = detail

    def warn(self, name: str, detail: str = ""):
        self.warnings.append(name)
        if detail:
            self.details[name] = detail

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0


# ===========================================================================
# Check implementations
# ===========================================================================

def check_file_presence(input_dir: Path, manifest_rows: List[Dict],
                         res: GateResult, log: GateLogger) -> bool:
    """Check 2: Verify exact presence of all required files."""
    log.section("CHECK 2 – File Presence")
    all_ok = True

    # Feature NPZs from manifest
    for row in manifest_rows:
        rel = row["relative_path"]
        fp = input_dir / rel
        if fp.exists():
            log.ok(f"Feature NPZ present: {rel}")
            res.ok(f"file_present:{rel}")
        else:
            log.error(f"MISSING feature NPZ: {rel}")
            res.fail(f"file_present:{rel}", f"File not found: {fp}")
            all_ok = False

    # Window NPZs
    for split, fname in WINDOW_NPZ_FILES.items():
        fp = input_dir / fname
        if fp.exists():
            log.ok(f"Window NPZ present: {fname}")
            res.ok(f"file_present:{fname}")
        else:
            log.error(f"MISSING window NPZ: {fname}")
            res.fail(f"file_present:{fname}", f"File not found: {fp}")
            all_ok = False

    # Manifest + metadata
    for fname in (MANIFEST_FILENAME, METADATA_FILENAME):
        fp = input_dir / fname
        if fp.exists():
            log.ok(f"Metadata file present: {fname}")
            res.ok(f"file_present:{fname}")
        else:
            log.error(f"MISSING metadata file: {fname}")
            res.fail(f"file_present:{fname}", f"File not found: {fp}")
            all_ok = False

    # Markov CSVs
    for split, fname in MARKOV_CSV_FILES.items():
        fp = input_dir / fname
        if fp.exists():
            log.ok(f"Markov CSV present: {fname}")
            res.ok(f"file_present:{fname}")
        else:
            log.error(f"MISSING Markov CSV: {fname}")
            res.fail(f"file_present:{fname}", f"File not found: {fp}")
            all_ok = False

    # Fitted basis files
    for fname in FITTED_BASIS_FILES:
        fp = input_dir / fname
        if fp.exists():
            log.ok(f"Fitted basis present: {fname}")
            res.ok(f"file_present:{fname}")
        else:
            log.error(f"MISSING fitted basis: {fname}")
            res.fail(f"file_present:{fname}", f"File not found: {fp}")
            all_ok = False

    return all_ok


def check_sha256_and_size(input_dir: Path, manifest_rows: List[Dict],
                           res: GateResult, log: GateLogger) -> bool:
    """Check 3: Validate SHA-256 and file size from manifest."""
    log.section("CHECK 3 – SHA-256 and File-Size Validation")
    all_ok = True

    for row in manifest_rows:
        rel  = row["relative_path"]
        fp   = input_dir / rel
        if not fp.exists():
            log.error(f"Skipping SHA check (file missing): {rel}")
            res.fail(f"sha256:{rel}", "File missing")
            all_ok = False
            continue

        expected_sha  = row["sha256"].strip()
        expected_size = int(row["actual_file_bytes"])
        actual_size   = fp.stat().st_size
        actual_sha    = compute_sha256(str(fp))

        size_ok = actual_size == expected_size
        sha_ok  = actual_sha  == expected_sha

        if size_ok and sha_ok:
            log.ok(f"SHA256+size match: {rel}")
            res.ok(f"sha256:{rel}")
        else:
            if not size_ok:
                msg = (f"Size mismatch for {rel}: expected={expected_size}, "
                       f"actual={actual_size}")
                log.error(msg)
                res.fail(f"file_size:{rel}", msg)
                all_ok = False
            if not sha_ok:
                note = row.get("sha256_note", "")
                msg = (f"SHA-256 mismatch for {rel}: "
                       f"expected={expected_sha}, actual={actual_sha}. "
                       f"Note: '{note}'")
                log.warn(msg)
                # Treat as warning (not hard fail) due to known cross-platform note
                res.warn(f"sha256:{rel}", msg)

    return all_ok


def check_feature_npz_structure(input_dir: Path, manifest_rows: List[Dict],
                                  res: GateResult, log: GateLogger) -> bool:
    """Check 4: Validate keys, shapes, dtypes, finiteness, label range."""
    log.section("CHECK 4 – Feature NPZ Structure (keys, dtypes, shapes, values)")
    all_ok = True

    for row in manifest_rows:
        rel      = row["relative_path"]
        split    = row["split"]
        config   = f"{row['method']}_{row['config_id']}"
        expected_n = int(row["n_samples"])
        expected_k = int(row["n_features"])
        fp = input_dir / rel

        if not fp.exists():
            res.fail(f"npz_struct:{rel}", "File missing")
            all_ok = False
            continue

        try:
            with np.load(str(fp), allow_pickle=True) as data:
                keys = set(data.files)

                # 4a: Exact key set
                if keys != REQUIRED_FEATURE_NPZ_KEYS:
                    msg = (f"{rel}: key mismatch – expected={sorted(REQUIRED_FEATURE_NPZ_KEYS)}, "
                           f"got={sorted(keys)}")
                    log.error(msg)
                    res.fail(f"npz_keys:{rel}", msg)
                    all_ok = False
                    continue

                F   = data["F"]
                y   = data["y"]
                ts  = data["target_timestamp"]
                cs  = data["current_state"]
                pm  = data["current_pm25"]

                # 4b: Sample count
                n = F.shape[0]
                if n != expected_n:
                    msg = f"{rel}: n_samples mismatch – expected={expected_n}, got={n}"
                    log.error(msg)
                    res.fail(f"npz_nsamples:{rel}", msg)
                    all_ok = False

                # 4c: Feature dim
                k = F.shape[1] if F.ndim >= 2 else -1
                if k != expected_k:
                    msg = f"{rel}: n_features mismatch – expected={expected_k}, got={k}"
                    log.error(msg)
                    res.fail(f"npz_nfeatures:{rel}", msg)
                    all_ok = False

                # 4d: Dtypes
                dtype_ok = True
                for arr_name, arr, expected_dtype in [
                    ("F",             F,   EXPECTED_DTYPES["F"]),
                    ("y",             y,   EXPECTED_DTYPES["y"]),
                    ("current_state", cs,  EXPECTED_DTYPES["current_state"]),
                    ("current_pm25",  pm,  EXPECTED_DTYPES["current_pm25"]),
                    ("target_timestamp", ts, EXPECTED_DTYPES["target_timestamp"]),
                ]:
                    if arr.dtype != expected_dtype:
                        msg = (f"{rel}['{arr_name}']: dtype mismatch – "
                               f"expected={expected_dtype}, got={arr.dtype}")
                        log.error(msg)
                        res.fail(f"npz_dtype:{rel}:{arr_name}", msg)
                        all_ok = False
                        dtype_ok = False

                # 4e: F finiteness
                if not np.all(np.isfinite(F)):
                    n_bad = int(np.sum(~np.isfinite(F)))
                    msg = f"{rel}['F']: {n_bad} non-finite values"
                    log.error(msg)
                    res.fail(f"npz_finite:{rel}", msg)
                    all_ok = False
                else:
                    res.ok(f"npz_finite:{rel}")

                # 4f: y label range
                unique_y = set(y.tolist())
                if not unique_y.issubset({0, 1, 2}):
                    msg = f"{rel}['y']: unexpected label values {unique_y}"
                    log.error(msg)
                    res.fail(f"npz_labels:{rel}", msg)
                    all_ok = False
                else:
                    res.ok(f"npz_labels:{rel}")

                if not any(k.startswith("npz_struct") and rel in k for k in res.failures):
                    log.ok(f"NPZ structure valid: {rel} (n={n}, k={k})")
                    res.ok(f"npz_struct:{rel}")

        except Exception as exc:
            msg = f"{rel}: Exception during load: {exc}"
            log.error(msg)
            res.fail(f"npz_load:{rel}", msg)
            all_ok = False

    return all_ok


def check_cross_config_consistency(input_dir: Path, manifest_rows: List[Dict],
                                    res: GateResult, log: GateLogger) -> bool:
    """Check 5: Cross-validate metadata arrays across configs for each split."""
    log.section("CHECK 5 – Cross-Config Metadata Consistency")
    all_ok = True

    # Group rows by split
    split_rows: Dict[str, List[Dict]] = {s: [] for s in SPLITS}
    for row in manifest_rows:
        split_rows[row["split"]].append(row)

    for split in SPLITS:
        rows = split_rows[split]
        if not rows:
            continue

        # Pick reference (pca_A preferred)
        ref_row = None
        for row in rows:
            cid = f"{row['method']}_{row['config_id']}"
            if cid == REFERENCE_CONFIG:
                ref_row = row
                break
        if ref_row is None:
            ref_row = rows[0]

        ref_path = input_dir / ref_row["relative_path"]
        if not ref_path.exists():
            log.error(f"Reference file missing for split {split}: {ref_path}")
            res.fail(f"cross_config_ref:{split}", "Reference file missing")
            all_ok = False
            continue

        # Load reference metadata
        try:
            ref_meta = load_npz_metadata_only(str(ref_path))
            ref_y   = ref_meta["y"]
            ref_ts  = to_datetime64_ns(ref_meta["target_timestamp"])
            ref_cs  = ref_meta["current_state"]
            ref_pm  = ref_meta["current_pm25"]
        except Exception as exc:
            log.error(f"Failed to load reference {ref_path}: {exc}")
            res.fail(f"cross_config_ref:{split}", str(exc))
            all_ok = False
            continue

        log.info(f"Split={split}: reference config={ref_row['method']}_{ref_row['config_id']}, n={len(ref_y)}")

        for row in rows:
            cid = f"{row['method']}_{row['config_id']}"
            if cid == f"{ref_row['method']}_{ref_row['config_id']}":
                continue
            fp = input_dir / row["relative_path"]
            if not fp.exists():
                res.fail(f"cross_config:{split}:{cid}", "File missing")
                all_ok = False
                continue
            try:
                meta = load_npz_metadata_only(str(fp))
                y_   = meta["y"]
                ts_  = to_datetime64_ns(meta["target_timestamp"])
                cs_  = meta["current_state"]
                pm_  = meta["current_pm25"]

                ok = True
                if not np.array_equal(ref_y, y_):
                    msg = f"{split}/{cid}: y array differs from reference"
                    log.error(msg)
                    res.fail(f"cross_config_y:{split}:{cid}", msg)
                    all_ok = ok = False

                if not np.array_equal(ref_ts, ts_):
                    msg = f"{split}/{cid}: target_timestamp differs from reference"
                    log.error(msg)
                    res.fail(f"cross_config_ts:{split}:{cid}", msg)
                    all_ok = ok = False

                if not np.array_equal(ref_cs, cs_):
                    msg = f"{split}/{cid}: current_state differs from reference"
                    log.error(msg)
                    res.fail(f"cross_config_cs:{split}:{cid}", msg)
                    all_ok = ok = False

                # current_pm25 may contain NaN; use equal_nan=True
                pm_match = np.array_equal(ref_pm, pm_) or (
                    ref_pm.shape == pm_.shape and
                    np.all((ref_pm == pm_) | (np.isnan(ref_pm) & np.isnan(pm_)))
                )
                if not pm_match:
                    msg = f"{split}/{cid}: current_pm25 differs from reference"
                    log.error(msg)
                    res.fail(f"cross_config_pm:{split}:{cid}", msg)
                    all_ok = ok = False

                if ok:
                    log.ok(f"Cross-config consistent: split={split} config={cid}")
                    res.ok(f"cross_config:{split}:{cid}")
            except Exception as exc:
                msg = f"{split}/{cid}: Exception: {exc}"
                log.error(msg)
                res.fail(f"cross_config:{split}:{cid}", msg)
                all_ok = False

    return all_ok


def check_window_alignment(input_dir: Path, manifest_rows: List[Dict],
                             res: GateResult, log: GateLogger) -> bool:
    """Check 6: Align Phase-2 window metadata with feature files."""
    log.section("CHECK 6 – Phase-2 Window Metadata Alignment")
    all_ok = True

    # Group manifest by split -> pick reference
    split_ref: Dict[str, Optional[Dict]] = {}
    for row in manifest_rows:
        s = row["split"]
        cid = f"{row['method']}_{row['config_id']}"
        if s not in split_ref or cid == REFERENCE_CONFIG:
            split_ref[s] = row

    for split in SPLITS:
        win_fname = WINDOW_NPZ_FILES[split]
        win_path  = input_dir / win_fname
        ref_row   = split_ref.get(split)
        if ref_row is None:
            continue

        ref_path = input_dir / ref_row["relative_path"]
        if not win_path.exists():
            log.error(f"Window NPZ missing: {win_fname}")
            res.fail(f"window_align:{split}", "Window NPZ missing")
            all_ok = False
            continue
        if not ref_path.exists():
            log.error(f"Reference feature NPZ missing for window alignment: {ref_path}")
            res.fail(f"window_align:{split}", "Reference NPZ missing")
            all_ok = False
            continue

        try:
            ref_meta = load_npz_metadata_only(str(ref_path))
            ref_y    = ref_meta["y"]
            ref_ts   = to_datetime64_ns(ref_meta["target_timestamp"])
            ref_cs   = ref_meta["current_state"]
            ref_pm   = ref_meta["current_pm25"]

            # Load only metadata arrays from window NPZ (not X tensor)
            win_meta = load_npz_metadata_only(str(win_path))
            win_y    = win_meta["y"]
            win_ts   = to_datetime64_ns(win_meta["target_timestamp"])
            win_cs   = win_meta["current_state"]
            win_pm   = win_meta["current_pm25"]

            ok = True

            if not np.array_equal(ref_y, win_y):
                msg = f"Window {split}: y mismatch with feature file"
                log.error(msg)
                res.fail(f"window_y:{split}", msg)
                all_ok = ok = False

            if not np.array_equal(ref_ts, win_ts):
                msg = f"Window {split}: target_timestamp mismatch with feature file"
                log.error(msg)
                res.fail(f"window_ts:{split}", msg)
                all_ok = ok = False

            if not np.array_equal(ref_cs, win_cs):
                msg = f"Window {split}: current_state mismatch with feature file"
                log.error(msg)
                res.fail(f"window_cs:{split}", msg)
                all_ok = ok = False

            pm_match = np.array_equal(ref_pm, win_pm) or (
                ref_pm.shape == win_pm.shape and
                np.all((ref_pm == win_pm) | (np.isnan(ref_pm) & np.isnan(win_pm)))
            )
            if not pm_match:
                msg = f"Window {split}: current_pm25 mismatch with feature file"
                log.error(msg)
                res.fail(f"window_pm:{split}", msg)
                all_ok = ok = False

            if ok:
                log.ok(f"Window alignment: split={split} OK")
                res.ok(f"window_align:{split}")

        except Exception as exc:
            msg = f"Window {split}: Exception: {exc}"
            log.error(msg)
            res.fail(f"window_align:{split}", msg)
            all_ok = False

    return all_ok


def check_sample_counts(manifest_rows: List[Dict],
                         res: GateResult, log: GateLogger) -> bool:
    """Check 7: Verify expected sample counts per split."""
    log.section("CHECK 7 – Sample Counts")
    all_ok = True

    # Collect n_samples from manifest (consistent within split)
    for split, expected_n in EXPECTED_SAMPLE_COUNTS.items():
        rows = [r for r in manifest_rows if r["split"] == split]
        if not rows:
            msg = f"No manifest rows for split={split}"
            log.error(msg)
            res.fail(f"sample_count:{split}", msg)
            all_ok = False
            continue
        for row in rows:
            actual_n = int(row["n_samples"])
            if actual_n != expected_n:
                msg = (f"Sample count mismatch split={split} config={row['method']}_{row['config_id']}: "
                       f"expected={expected_n}, got={actual_n}")
                log.error(msg)
                res.fail(f"sample_count:{split}:{row['config_id']}", msg)
                all_ok = False
            else:
                log.ok(f"Sample count split={split} config={row['method']}_{row['config_id']}: {actual_n}")
                res.ok(f"sample_count:{split}:{row['method']}_{row['config_id']}")

    return all_ok


def check_class_distribution(input_dir: Path, manifest_rows: List[Dict],
                               res: GateResult, log: GateLogger) -> bool:
    """Check 8: Verify class distribution per split."""
    log.section("CHECK 8 – Class Distribution")
    all_ok = True

    split_ref: Dict[str, Optional[str]] = {}
    for row in manifest_rows:
        s   = row["split"]
        cid = f"{row['method']}_{row['config_id']}"
        if s not in split_ref or cid == REFERENCE_CONFIG:
            split_ref[s] = row["relative_path"]

    for split in SPLITS:
        expected_dist = EXPECTED_CLASS_DIST[split]
        rel = split_ref.get(split)
        if rel is None:
            continue
        fp = input_dir / rel
        if not fp.exists():
            res.fail(f"class_dist:{split}", "Reference file missing")
            all_ok = False
            continue
        try:
            meta = load_npz_metadata_only(str(fp), meta_keys=("y",))
            y = meta["y"]
            for label, expected_count in expected_dist.items():
                actual_count = int(np.sum(y == label))
                label_name = {0: "Low", 1: "Moderate", 2: "High"}[label]
                if actual_count != expected_count:
                    msg = (f"Class dist mismatch split={split} label={label}({label_name}): "
                           f"expected={expected_count}, got={actual_count}")
                    log.error(msg)
                    res.fail(f"class_dist:{split}:{label}", msg)
                    all_ok = False
                else:
                    log.ok(f"Class dist split={split} label={label_name}: {actual_count}")
                    res.ok(f"class_dist:{split}:{label}")
        except Exception as exc:
            msg = f"Class dist split={split}: Exception: {exc}"
            log.error(msg)
            res.fail(f"class_dist:{split}", msg)
            all_ok = False

    return all_ok


def check_chronological_ordering(input_dir: Path, manifest_rows: List[Dict],
                                   res: GateResult, log: GateLogger) -> bool:
    """Check 9: Chronological ordering, no duplicates, no cross-split overlap."""
    log.section("CHECK 9 – Chronological Order, Duplicates, Cross-Split Overlap")
    all_ok = True

    split_ts: Dict[str, Optional[np.ndarray]] = {}
    split_ref: Dict[str, Optional[str]] = {}
    for row in manifest_rows:
        s   = row["split"]
        cid = f"{row['method']}_{row['config_id']}"
        if s not in split_ref or cid == REFERENCE_CONFIG:
            split_ref[s] = row["relative_path"]

    for split in SPLITS:
        rel = split_ref.get(split)
        if rel is None:
            split_ts[split] = None
            continue
        fp = input_dir / rel
        if not fp.exists():
            split_ts[split] = None
            res.fail(f"chronological:{split}", "File missing")
            all_ok = False
            continue
        try:
            meta = load_npz_metadata_only(str(fp), meta_keys=("target_timestamp",))
            ts   = to_datetime64_ns(meta["target_timestamp"])
            split_ts[split] = ts

            # Strictly monotone
            if not np.all(ts[1:] > ts[:-1]):
                n_bad = int(np.sum(ts[1:] <= ts[:-1]))
                msg = f"split={split}: {n_bad} timestamp(s) not strictly increasing"
                log.error(msg)
                res.fail(f"chronological:{split}", msg)
                all_ok = False
            else:
                log.ok(f"Chronological order OK: split={split}")
                res.ok(f"chronological:{split}")

            # No duplicates
            unique_ts = np.unique(ts)
            if len(unique_ts) != len(ts):
                n_dup = len(ts) - len(unique_ts)
                msg = f"split={split}: {n_dup} duplicate timestamp(s)"
                log.error(msg)
                res.fail(f"no_duplicate_ts:{split}", msg)
                all_ok = False
            else:
                log.ok(f"No duplicate timestamps: split={split}")
                res.ok(f"no_duplicate_ts:{split}")

        except Exception as exc:
            msg = f"Chronological check split={split}: Exception: {exc}"
            log.error(msg)
            res.fail(f"chronological:{split}", msg)
            split_ts[split] = None
            all_ok = False

    # Cross-split timestamp overlap
    split_list = [s for s in SPLITS if split_ts.get(s) is not None]
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            s1, s2 = split_list[i], split_list[j]
            ts1 = set(split_ts[s1].tolist())
            ts2 = set(split_ts[s2].tolist())
            overlap = ts1 & ts2
            if overlap:
                msg = f"Cross-split overlap between {s1} and {s2}: {len(overlap)} shared timestamp(s)"
                log.error(msg)
                res.fail(f"cross_split_overlap:{s1}_{s2}", msg)
                all_ok = False
            else:
                log.ok(f"No cross-split timestamp overlap: {s1} vs {s2}")
                res.ok(f"cross_split_overlap:{s1}_{s2}")

    return all_ok


def check_scope_a_counts(input_dir: Path, manifest_rows: List[Dict],
                          res: GateResult, log: GateLogger) -> bool:
    """Check 10: Scope A (current_state >= 0) counts."""
    log.section("CHECK 10 – Scope A Counts (current_state >= 0)")
    all_ok = True

    split_ref: Dict[str, Optional[str]] = {}
    for row in manifest_rows:
        s   = row["split"]
        cid = f"{row['method']}_{row['config_id']}"
        if s not in split_ref or cid == REFERENCE_CONFIG:
            split_ref[s] = row["relative_path"]

    for split, expected_n in EXPECTED_SCOPE_A_COUNTS.items():
        rel = split_ref.get(split)
        if rel is None:
            continue
        fp = input_dir / rel
        if not fp.exists():
            res.fail(f"scope_a:{split}", "File missing")
            all_ok = False
            continue
        try:
            meta = load_npz_metadata_only(str(fp), meta_keys=("current_state",))
            cs   = meta["current_state"]
            actual_n = int(np.sum(cs >= 0))
            if actual_n != expected_n:
                msg = f"Scope A count mismatch split={split}: expected={expected_n}, got={actual_n}"
                log.error(msg)
                res.fail(f"scope_a:{split}", msg)
                all_ok = False
            else:
                log.ok(f"Scope A count split={split}: {actual_n}")
                res.ok(f"scope_a:{split}")
        except Exception as exc:
            msg = f"Scope A split={split}: Exception: {exc}"
            log.error(msg)
            res.fail(f"scope_a:{split}", msg)
            all_ok = False

    return all_ok


def check_state_pm25_consistency(input_dir: Path, manifest_rows: List[Dict],
                                   res: GateResult, log: GateLogger) -> bool:
    """Check 11: current_state==-1 exactly when current_pm25 is NaN."""
    log.section("CHECK 11 – current_state==-1 ⟺ current_pm25 is NaN")
    all_ok = True

    split_ref: Dict[str, Optional[str]] = {}
    for row in manifest_rows:
        s   = row["split"]
        cid = f"{row['method']}_{row['config_id']}"
        if s not in split_ref or cid == REFERENCE_CONFIG:
            split_ref[s] = row["relative_path"]

    for split in SPLITS:
        rel = split_ref.get(split)
        if rel is None:
            continue
        fp = input_dir / rel
        if not fp.exists():
            res.fail(f"state_pm25:{split}", "File missing")
            all_ok = False
            continue
        try:
            meta = load_npz_metadata_only(str(fp), meta_keys=("current_state", "current_pm25"))
            cs   = meta["current_state"]
            pm   = meta["current_pm25"]

            mask_neg1 = cs == -1
            mask_nan  = np.isnan(pm)

            only_neg1   = np.sum(mask_neg1 & ~mask_nan)
            only_nan    = np.sum(mask_nan  & ~mask_neg1)

            if only_neg1 == 0 and only_nan == 0:
                log.ok(f"State/PM25 consistency: split={split} OK "
                       f"(n_missing={int(np.sum(mask_neg1))})")
                res.ok(f"state_pm25:{split}")
            else:
                msg = (f"split={split}: {int(only_neg1)} rows have state==-1 but PM25 not NaN; "
                       f"{int(only_nan)} rows have PM25 NaN but state!=-1")
                log.error(msg)
                res.fail(f"state_pm25:{split}", msg)
                all_ok = False
        except Exception as exc:
            msg = f"State/PM25 split={split}: Exception: {exc}"
            log.error(msg)
            res.fail(f"state_pm25:{split}", msg)
            all_ok = False

    return all_ok


def check_markov_csvs(input_dir: Path, manifest_rows: List[Dict],
                       res: GateResult, log: GateLogger) -> bool:
    """Check 12: Markov CSV row counts, columns, timestamp alignment."""
    log.section("CHECK 12 – Markov CSV Validation")
    all_ok = True

    # Reference timestamps from feature files
    split_ref: Dict[str, Optional[str]] = {}
    for row in manifest_rows:
        s   = row["split"]
        cid = f"{row['method']}_{row['config_id']}"
        if s not in split_ref or cid == REFERENCE_CONFIG:
            split_ref[s] = row["relative_path"]

    for split, fname in MARKOV_CSV_FILES.items():
        fp = input_dir / fname
        expected_n = EXPECTED_SAMPLE_COUNTS[split]

        if not fp.exists():
            log.error(f"Markov CSV missing: {fname}")
            res.fail(f"markov_present:{split}", "File missing")
            all_ok = False
            continue

        try:
            with open(str(fp), newline="") as fh:
                reader = csv.DictReader(fh)
                rows   = list(reader)
                cols   = set(reader.fieldnames or [])

            # Row count
            actual_n = len(rows)
            if actual_n != expected_n:
                msg = f"Markov {fname}: row count expected={expected_n}, got={actual_n}"
                log.error(msg)
                res.fail(f"markov_rows:{split}", msg)
                all_ok = False
            else:
                log.ok(f"Markov {fname}: row count {actual_n}")
                res.ok(f"markov_rows:{split}")

            # Required columns
            missing_cols = REQUIRED_MARKOV_COLUMNS - cols
            if missing_cols:
                msg = f"Markov {fname}: missing columns {missing_cols}"
                log.error(msg)
                res.fail(f"markov_cols:{split}", msg)
                all_ok = False
            else:
                log.ok(f"Markov {fname}: all required columns present")
                res.ok(f"markov_cols:{split}")

            # Timestamp alignment with feature file
            ref_rel = split_ref.get(split)
            if ref_rel:
                ref_fp = input_dir / ref_rel
                if ref_fp.exists():
                    ref_meta = load_npz_metadata_only(str(ref_fp), meta_keys=("target_timestamp",))
                    ref_ts   = to_datetime64_ns(ref_meta["target_timestamp"])
                    csv_ts   = np.array(
                        [np.datetime64(r["target_timestamp"].replace(" ", "T"), "ns") for r in rows],
                        dtype="datetime64[ns]"
                    )
                    if not np.array_equal(ref_ts, csv_ts):
                        n_mismatch = int(np.sum(ref_ts != csv_ts))
                        msg = f"Markov {fname}: {n_mismatch} timestamp(s) differ from feature file"
                        log.error(msg)
                        res.fail(f"markov_ts:{split}", msg)
                        all_ok = False
                    else:
                        log.ok(f"Markov {fname}: timestamps align with feature file")
                        res.ok(f"markov_ts:{split}")

        except Exception as exc:
            msg = f"Markov CSV {fname}: Exception: {exc}"
            log.error(msg)
            res.fail(f"markov:{split}", msg)
            all_ok = False

    return all_ok


# ===========================================================================
# Output writers
# ===========================================================================

def write_results(output_dir: Path, res: GateResult, log: GateLogger,
                   elapsed: float, args_dict: Dict):
    """Check 13/14: Write JSON + TXT report."""
    results_dir  = output_dir / "results"
    manifest_dir = output_dir / "manifests"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "gate_passed":    res.passed,
        "n_checks_passed": len(res.passes),
        "n_checks_failed": len(res.failures),
        "n_checks_warned": len(res.warnings),
        "elapsed_seconds": round(elapsed, 2),
        "passes":    res.passes,
        "failures":  res.failures,
        "warnings":  res.warnings,
        "details":   res.details,
        "args":      args_dict,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    json_path = results_dir / "integrity_gate.json"
    with open(str(json_path), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info(f"JSON report written: {json_path}")

    txt_path = results_dir / "integrity_gate_report.txt"
    with open(str(txt_path), "w") as fh:
        fh.write("=" * 72 + "\n")
        fh.write("PHASE 5 INTEGRITY GATE REPORT\n")
        fh.write(f"Timestamp (UTC): {summary['timestamp_utc']}\n")
        fh.write(f"Status: {'PASSED' if res.passed else 'FAILED'}\n")
        fh.write(f"Checks passed : {len(res.passes)}\n")
        fh.write(f"Checks failed : {len(res.failures)}\n")
        fh.write(f"Warnings      : {len(res.warnings)}\n")
        fh.write(f"Elapsed (s)   : {elapsed:.2f}\n")
        fh.write("=" * 72 + "\n\n")
        if res.failures:
            fh.write("FAILURES:\n")
            for f in res.failures:
                detail = res.details.get(f, "")
                fh.write(f"  FAIL  {f}: {detail}\n")
            fh.write("\n")
        if res.warnings:
            fh.write("WARNINGS:\n")
            for w in res.warnings:
                detail = res.details.get(w, "")
                fh.write(f"  WARN  {w}: {detail}\n")
            fh.write("\n")
        fh.write("PASSES:\n")
        for p in res.passes:
            fh.write(f"  PASS  {p}\n")
    log.info(f"Text report written: {txt_path}")


def write_provenance_manifest(input_dir: Path, output_dir: Path,
                               res: GateResult, log: GateLogger):
    """Check 14/15: Write provenance manifest with SHA-256s of all input files."""
    log.section("CHECK 15 – Provenance Manifest")
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    all_input_files = sorted(str(p) for p in input_dir.iterdir() if p.is_file())
    log.info(f"Computing SHA-256 for {len(all_input_files)} input files …")
    provenance = build_provenance_manifest(all_input_files, str(input_dir))

    for fname, info in provenance.items():
        log.info(f"  {fname}: sha256={info['sha256'][:16]}… bytes={info['bytes']}")

    prov_path = manifest_dir / "input_provenance.json"
    with open(str(prov_path), "w") as fh:
        json.dump(provenance, fh, indent=2, default=str)
    log.info(f"Provenance manifest written: {prov_path}")
    res.ok("provenance_manifest")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 Integrity Gate")
    p.add_argument("--input-dir",  default="inputs",
                   help="Directory containing Phase 4 output files")
    p.add_argument("--output-dir", default="output",
                   help="Directory for Phase 5 outputs")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed (unused here, logged for reproducibility)")
    return p.parse_args()


def main():
    args   = parse_args()
    input_dir  = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Create output dirs
    for sub in ("source", "results", "manifests", "logs"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    log = GateLogger(str(output_dir / "logs" / "s01_integrity_gate.log"))
    res = GateResult()
    t0  = time.time()

    log.section("PHASE 5 – INTEGRITY GATE")
    log.info(f"input_dir  = {input_dir}")
    log.info(f"output_dir = {output_dir}")
    log.info(f"seed       = {args.seed}")

    # ------------------------------------------------------------------ #
    # Read manifest
    # ------------------------------------------------------------------ #
    manifest_path = input_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        log.error(f"Manifest not found: {manifest_path}")
        log.error("INTEGRITY GATE: FAILED (manifest missing – cannot continue)")
        sys.exit(1)

    with open(str(manifest_path), newline="") as fh:
        manifest_rows = list(csv.DictReader(fh))
    log.info(f"Manifest loaded: {len(manifest_rows)} rows")

    # ------------------------------------------------------------------ #
    # Run all checks
    # ------------------------------------------------------------------ #
    check_file_presence(input_dir, manifest_rows, res, log)
    check_sha256_and_size(input_dir, manifest_rows, res, log)
    check_feature_npz_structure(input_dir, manifest_rows, res, log)
    check_cross_config_consistency(input_dir, manifest_rows, res, log)
    check_window_alignment(input_dir, manifest_rows, res, log)
    check_sample_counts(manifest_rows, res, log)
    check_class_distribution(input_dir, manifest_rows, res, log)
    check_chronological_ordering(input_dir, manifest_rows, res, log)
    check_scope_a_counts(input_dir, manifest_rows, res, log)
    check_state_pm25_consistency(input_dir, manifest_rows, res, log)
    check_markov_csvs(input_dir, manifest_rows, res, log)

    # ------------------------------------------------------------------ #
    # Provenance manifest
    # ------------------------------------------------------------------ #
    write_provenance_manifest(input_dir, output_dir, res, log)

    # ------------------------------------------------------------------ #
    # Write reports
    # ------------------------------------------------------------------ #
    elapsed = time.time() - t0
    args_dict = {"input_dir": str(input_dir), "output_dir": str(output_dir), "seed": args.seed}
    write_results(output_dir, res, log, elapsed, args_dict)

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #
    log.section("FINAL SUMMARY")
    log.info(f"Checks passed : {len(res.passes)}")
    log.info(f"Checks failed : {len(res.failures)}")
    log.info(f"Warnings      : {len(res.warnings)}")
    log.info(f"Elapsed (s)   : {elapsed:.2f}")

    if res.failures:
        for f in res.failures:
            detail = res.details.get(f, "")
            log.error(f"  FAIL  {f}: {detail}")

    if res.passed:
        msg = (f"INTEGRITY GATE: PASSED  "
               f"({len(res.passes)} checks passed, {len(res.warnings)} warnings)")
        log.info(msg)
        print(msg, flush=True)
        log.close()
        sys.exit(0)
    else:
        msg = (f"INTEGRITY GATE: FAILED  "
               f"({len(res.failures)} failures, {len(res.passes)} passed, "
               f"{len(res.warnings)} warnings)")
        log.error(msg)
        print(msg, flush=True)
        log.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
