"""Shared fixtures.

The fixtures are deliberately small. A test that needs five thousand simulated
days to make its point is testing the simulation, not the claim, and it puts a
minute on every full-suite run. The heavy end-to-end checks live in
``test_pipeline.py`` behind the ``slow`` marker, where they can be deselected.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketrisk import config as cfg
from marketrisk import data as data_module

#: Where the scratch-directory fixture puts things. Inside the project so that the
#: tests do not depend on the system temporary directory being writable, and
#: ignored by git.
SCRATCH_ROOT = Path(__file__).resolve().parents[1] / ".pytest_tmp"


@pytest.fixture(scope="session")
def scratch():
    """A factory for clean scratch directories.

    ``tmp_path`` is the idiomatic choice and would be the obvious one, but it fails
    outright in a sandbox that cannot create its base directory, and a test that
    errors for an unrelated environmental reason is worse than no test. This builds
    directories under the project root, which is where the pipeline writes anyway.

    Two details are deliberate, and both cost a debugging session to learn:

    * the directories live under a **per-process** root, so a run never has to delete
      a tree left behind by an earlier one. With a fixed root, the first ``make()``
      call after any previous run had to remove several dozen files at once - which is
      the exact pattern a bulk-delete guard exists to block, and a guard that aborts
      the process takes the test session down with it. The failure looks like a broken
      test and is really a housekeeping collision;
    * nothing is removed at teardown either, for the same reason. The end-to-end
      pipeline legitimately writes a couple of dozen files, so "clean up" means "delete
      all of them at once", and a suite that dies while tidying is worse than one that
      leaves a gitignored directory behind. ``make clean`` removes it.
    """
    root = SCRATCH_ROOT / f"run-{os.getpid()}"

    def make(name: str) -> Path:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    yield make


@pytest.fixture(scope="session")
def market():
    """A synthetic market long enough to fit a GARCH and short enough to be quick.

    Ten years gives roughly 2,500 trading days, which is enough for the 1,000-day
    initial window plus about 1,500 out-of-sample days - the same shape as the
    study, at a fifth of the cost.
    """
    return data_module.load_market(data_mode="synthetic", start="2010-01-04", end="2019-12-31", seed=cfg.SEED)


@pytest.fixture(scope="session")
def panel(market):
    """The primary asset panel."""
    return market.primary


@pytest.fixture(scope="session")
def returns(panel) -> pd.Series:
    return panel.returns


@pytest.fixture(scope="session")
def features(panel) -> pd.DataFrame:
    return panel.features


@pytest.fixture(scope="session")
def realised(panel) -> pd.Series:
    return panel.realised


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh generator, so a test that draws cannot affect the next one."""
    return np.random.default_rng(cfg.SEED)


@pytest.fixture(scope="session")
def garch_fit(returns, features):
    """One GJR fit, reused. Fitting is the slow part of most of these tests."""
    from marketrisk import volmodels

    model = volmodels.GARCH(kind="gjr")
    model.fit(returns.to_numpy(dtype=float), features, len(returns))
    return model


@pytest.fixture(scope="session")
def normal_shocks() -> np.ndarray:
    return np.random.default_rng(7).standard_normal(4_000)
