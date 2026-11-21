"""
conftest.py - portable test configuration (no absolute paths).

Locations are resolved, in order, from
  1. pytest options   --input-dir / --output-dir
  2. environment      PHASE5_INPUT_DIR / PHASE5_OUTPUT_DIR
  3. defaults         output-dir = package root (the folder that contains source/ and tests/);
                      input-dir  = unset  -> tests that need the large Phase 2-4 inputs are SKIPPED with a reason.
"""
import os
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "source"))


def pytest_addoption(parser):
    parser.addoption("--input-dir", action="store", default=None, help="Directory with Phase 2-4 inputs")
    parser.addoption("--output-dir", action="store", default=None, help="Package root holding results/, models/, ...")


@pytest.fixture(scope="session")
def out_dir(request) -> Path:
    p = request.config.getoption("--output-dir") or os.environ.get("PHASE5_OUTPUT_DIR") or str(PKG)
    return Path(p).resolve()


@pytest.fixture(scope="session")
def in_dir(request):
    p = request.config.getoption("--input-dir") or os.environ.get("PHASE5_INPUT_DIR")
    if not p or not (Path(p) / "phase4_feature_manifest.csv").exists():
        pytest.skip("Phase 2-4 input directory not supplied (use --input-dir or PHASE5_INPUT_DIR)")
    return Path(p).resolve()


@pytest.fixture(scope="session")
def need_results(out_dir):
    def _need(*rel):
        for r in rel:
            if not (out_dir / r).exists():
                pytest.skip(f"required result artifact not present: {r}")
        return out_dir
    return _need
