#!/usr/bin/env python3
"""
phase5_run.py - master pipeline runner for Phase 5 (remediated).

One command (inputs supplied via --input-dir, everything written under --output-dir = this package root by default):

    python phase5_run.py --input-dir /path/to/Phase2-4_inputs --stage all

Stage order for ``--stage all``:
    gate -> qp -> qp_real -> stage_a -> stage_b -> stage_c -> eval -> comparative -> revision -> tables -> report
         -> test -> finalize -> package

Resume behaviour: a stage whose primary outputs already exist is SKIPPED (recorded as 'skipped_existing' in
logs/run_registry.jsonl) so completed results are never overwritten.  Use --force to recompute a stage.
Stage B and the revision stage additionally resume per grid cell / per seed checkpoint.
Nothing in this runner reads or writes the Phase 2-4 inputs except through the stage scripts, which open them read-only.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE_ORDER = ["gate", "qp", "qp_real", "stage_a", "stage_b", "stage_c", "eval", "comparative", "revision", "tables",
               "report", "test", "finalize", "package"]

# primary output(s) whose existence marks a stage as complete
MARKERS = {
    "gate": ["results/integrity_gate.json"],
    "qp": ["results/qp_verification.json"],
    "qp_real": ["results/qp_real_subset.json"],
    "stage_a": ["results/stage_a_all_candidates.csv", "results/stage_a_best_per_config.csv"],
    "stage_b": ["manifests/STAGE_B_SELECTION_validation_only.json", "results/stage_b_all_candidates.csv"],
    "stage_c": ["results/stage_c_full_experiment.csv"],
    "eval": ["manifests/LOCKED_SELECTION.json", "results/final_evaluation_scopeAB.csv"],
    "comparative": ["figures/pareto_performance_vs_runtime.png"],
    "revision": ["manifests/REVISED_SELECTION_post_audit.json", "results/rff_seed_sweep_descriptive.csv"],
}


def build_plan(stage: str):
    """Ordered list of stages to run for --stage <stage> (exposed for tests)."""
    if stage == "all":
        return list(STAGE_ORDER)
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage}")
    return [stage]


def stage_commands(a, py):
    inp, out, seed = str(a.input_dir), str(a.output_dir), str(a.seed)
    S = Path(a.output_dir) / "source"
    common = ["--input-dir", inp, "--output-dir", out]
    return {
        "gate": [py, S / "s01_integrity_gate.py", *common, "--seed", seed],
        "qp": [py, S / "s02_qp_verification.py", "--output-dir", out, "--seed", seed],
        "qp_real": [py, S / "s02b_qp_real_subset.py", *common],
        "stage_a": [py, S / "s03_stage_a_linear_svm.py", *common, "--seed", seed],
        "stage_b": [py, S / "s04_stage_b_rbf_svc.py", *common, "--seed", seed, "--threads", str(a.threads),
                    "--max-ram-gb", str(a.max_ram_gb), "--rbf-max-seconds", str(a.rbf_max_seconds)] + (["--full-exact"] if a.full_exact else []),
        "stage_c": [py, S / "s05_stage_c_rff.py", *common, "--seed", seed],
        "eval": [py, S / "s06_selection_and_evaluation.py", *common, "--seed", seed],
        "comparative": [py, S / "s07_comparative_studies.py", *common, "--seed", seed],
        "revision": [py, S / "s09_post_audit_revision.py", *common, "--seed", seed, "--threads", str(a.threads)],
        "tables": [py, S / "s10_tables_and_bootstrap.py", *common, "--seed", seed],
        "report": [py, S / "s11_report_and_audit.py", *common],
        "finalize": [py, S / "s11_report_and_audit.py", *common],
        "package": [py, S / "s08_package_and_audit.py", "--output-dir", out],
    }


def run_tests(a, py):
    """Run pytest, keep the raw log, and write results/test_summary.json from the JUnit XML (actual counts)."""
    out = Path(a.output_dir)
    (out / "logs").mkdir(exist_ok=True)
    xml = out / "logs" / "pytest_junit.xml"
    cmd = [py, "-m", "pytest", str(out / "tests"), "-v", "-rs", "--tb=short", f"--junitxml={xml}", "--output-dir", str(out)]
    if a.input_dir and Path(a.input_dir).exists():
        cmd += ["--input-dir", str(a.input_dir)]
    t0 = time.time()
    with open(out / "logs" / "pytest_log.txt", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=out).returncode
    suites = ET.parse(xml).getroot()
    suite = suites if suites.tag == "testsuite" else suites.find("testsuite")
    tot = int(suite.get("tests"))
    fail = int(suite.get("failures")) + int(suite.get("errors"))
    skip = int(suite.get("skipped"))
    skipped = [{"test": tc.get("classname", "") + "::" + tc.get("name"), "reason": tc.find("skipped").get("message")}
               for tc in suite.iter("testcase") if tc.find("skipped") is not None]
    summ = {"command": " ".join(cmd), "returncode": rc, "total": tot, "passed": tot - fail - skip, "failed": fail, "skipped": skip,
            "skipped_tests": skipped, "seconds": round(time.time() - t0, 1)}
    (out / "results" / "test_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"TESTS total={tot} passed={summ['passed']} failed={fail} skipped={skip}")
    return rc


def main():
    p = argparse.ArgumentParser(description="Phase 5 master pipeline runner")
    p.add_argument("--input-dir", default=".", help="Directory containing the Phase 2-4 input files (read-only)")
    p.add_argument("--output-dir", default=str(ROOT), help="Package root where results/, models/, ... are written")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=1, help="BLAS threads (OMP_NUM_THREADS)")
    p.add_argument("--max-ram-gb", type=float, default=2.5, help="Stage B RAM guard (estimate-based, not an OS cap)")
    p.add_argument("--rbf-max-seconds", type=float, default=600.0, help="Enforced per-fit wall-time limit for exact RBF fits")
    p.add_argument("--full-exact", action="store_true", help="Also pilot full-train exact RBF fits (bounded by --rbf-max-seconds)")
    p.add_argument("--stage", default="all", choices=["all"] + STAGE_ORDER)
    p.add_argument("--force", action="store_true", help="Recompute stages even if their outputs exist")
    a = p.parse_args()
    a.input_dir, a.output_dir = Path(a.input_dir).resolve(), Path(a.output_dir).resolve()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    sys.path.insert(0, str(a.output_dir / "source"))
    from common import RunRegistry
    reg = RunRegistry(a.output_dir / "logs" / "run_registry.jsonl")
    py = sys.executable
    cmds = stage_commands(a, py)
    for stg in build_plan(a.stage):
        if stg in MARKERS and not a.force and all((a.output_dir / m).exists() for m in MARKERS[stg]) and stg != "stage_b":
            print(f"[SKIP] {stg}: outputs already exist (use --force to recompute)")
            reg.log(stage=stg, status="skipped_existing")
            continue
        print("=" * 70, f"\nSTAGE {stg}", flush=True)
        t0 = time.time()
        if stg == "test":
            rc = run_tests(a, py)       # a failing suite is recorded, not hidden; the pipeline still finalises
            reg.log(stage=stg, status="tests_passed" if rc == 0 else "tests_FAILED", seconds=round(time.time() - t0, 1))
            continue
        rc = subprocess.run([str(c) for c in cmds[stg]], cwd=a.output_dir).returncode
        reg.log(stage=stg, status="done" if rc == 0 else f"failed_rc{rc}", seconds=round(time.time() - t0, 1))
        if rc != 0:
            print(f"[ERROR] stage {stg} failed (rc={rc})")
            sys.exit(rc)
    print("Requested stages finished.")


if __name__ == "__main__":
    main()
