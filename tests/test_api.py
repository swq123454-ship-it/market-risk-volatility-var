"""Tests for the serving layer.

Two contracts are being defended, and both are about *omission* rather than
computation.

1. **The engine holds history, not just parameters.** A request that supplies a
   fresh close must move the risk number, because the filter is run forward over
   the appended observation. An engine that answers from stored parameters alone
   cannot do that, and its risk number is stale by however many days have passed.
2. **Missing is not zero.** A JSON payload cannot carry a NaN, so a day with no
   observation arrives as an absent element rather than as ``0.0``. Substituting
   zero for a missing return silently biases every volatility estimate downwards,
   which is the direction that understates risk. The service rejects such payloads.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from marketrisk import api
from marketrisk import config as cfg
from marketrisk import data as data_module


@pytest.fixture(scope="module")
def engine() -> api.RiskEngine:
    market = data_module.load_market(data_mode="synthetic", start="2012-01-03", end="2019-12-31", seed=cfg.SEED)
    panel = market.primary
    return api.RiskEngine.fit(panel.returns, panel.features, asset=panel.key)


@pytest.fixture(scope="module")
def client(engine) -> TestClient:
    return TestClient(api.create_app(engine))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def test_engine_records_the_data_it_was_fitted_on(engine):
    assert engine.observations > 1_000
    assert engine.asset == cfg.PRIMARY_ASSET
    assert engine.reference_window == cfg.HS_WINDOWS[0]
    assert engine.sample_end


def test_risk_returns_every_method_with_positive_numbers(engine):
    payload = engine.risk(level=0.99, horizon=1)
    assert set(payload["methods"]) == {"hs", "parametric_normal", "parametric_t", "fhs"}
    for name, values in payload["methods"].items():
        assert values["var"] > 0.0, name
        assert values["es"] >= values["var"] - 1e-12, name


def test_risk_reports_the_reference_window_it_used(engine):
    """Contract one: the response says how much history is behind the number."""
    payload = engine.risk()
    assert payload["reference_window"] == engine.reference_window
    assert payload["observations"] == engine.observations
    assert "window_worst_return" in payload["diagnostics"]
    assert payload["diagnostics"]["expected_breaches_per_year"] == pytest.approx(0.01 * cfg.TRADING_DAYS)


def test_a_fresh_close_moves_the_conditional_volatility(engine):
    """Contract one, stated as a test: the filter is run forward, not read off."""
    calm = np.zeros(1)
    stressed = np.array([-0.08])  # an eight-percent down day

    baseline = engine.risk(extra=calm)["conditional_vol_daily"]
    after_shock = engine.risk(extra=np.concatenate([calm, stressed]))["conditional_vol_daily"]
    assert after_shock > baseline


def test_extra_history_does_not_mutate_the_stored_returns(engine):
    before = engine.returns.copy()
    engine.risk(extra=np.array([-0.05, -0.04]))
    assert np.array_equal(engine.returns, before)


def test_a_longer_horizon_scales_the_numbers_up(engine):
    one = engine.risk(horizon=1)
    ten = engine.risk(horizon=10)
    for name in one["methods"]:
        assert ten["methods"][name]["var"] > one["methods"][name]["var"], name


def test_a_higher_confidence_level_gives_a_larger_var(engine):
    ninety = engine.risk(level=0.95)
    ninety_nine = engine.risk(level=0.99)
    for name in ninety["methods"]:
        assert ninety_nine["methods"][name]["var"] > ninety["methods"][name]["var"], name


def test_fat_tailed_methods_exceed_the_gaussian_one_at_the_same_scale(engine):
    payload = engine.risk(level=0.99)
    # If the fitted degrees of freedom had come out above 200 the t and the
    # normal would coincide by construction, so the comparison below would be
    # vacuous. Assert the premise before testing the consequence.
    assert payload["degrees_of_freedom"] is not None
    assert payload["degrees_of_freedom"] < 60.0
    assert payload["methods"]["parametric_t"]["var"] > payload["methods"]["parametric_normal"]["var"]


def test_info_reports_a_persistence_and_a_half_life(engine):
    info = engine.info()
    assert 0.0 < info["persistence"] < 1.0
    assert info["half_life_days"] > 0.0
    assert info["parameters"]["alpha"] is not None
    assert info["observations"] == engine.observations


def test_info_turns_a_non_finite_dof_into_null(engine):
    """JSON has no infinity; the field is null rather than a string or a sentinel."""
    info = engine.info()
    dof = info["parameters"]["dof"]
    assert dof is None or (isinstance(dof, float) and np.isfinite(dof))


def test_engine_rejects_a_history_that_is_too_short():
    tiny = api.RiskEngine(
        returns=np.zeros(10),
        garch=type("M", (), {"params": {}})(),
        ewma=type("M", (), {"chosen": 0.94})(),
        rolling=type("M", (), {"window": 21})(),
    )
    with pytest.raises(ValueError, match="at least 60 observations"):
        tiny.risk()


def test_coverage_history_is_unavailable_without_a_var_history(engine):
    assert engine.coverage_history() == {"available": False}


def test_coverage_history_reports_a_zone_when_it_has_one(engine):
    import copy

    import pandas as pd

    local = copy.copy(engine)
    n = len(local.returns)
    local.var_history = pd.DataFrame({str(cfg.PRIMARY_LEVEL): np.full(n, 0.01)}, index=range(n))
    payload = local.coverage_history(level=cfg.PRIMARY_LEVEL, window=250)
    assert payload["available"] is True
    assert payload["zone"] in {"green", "yellow", "red"}
    assert payload["plus_factor"] >= 0.0


# ---------------------------------------------------------------------------
# Bundle round trip
# ---------------------------------------------------------------------------
def test_bundle_round_trip_preserves_the_numbers(engine, scratch):
    path = scratch("api_bundle") / "engine.joblib"
    api.save_bundle(engine, path)
    restored = api.load_bundle(path)
    assert restored.version == engine.version
    assert restored.observations == engine.observations
    a = engine.risk()
    b = restored.risk()
    for name in a["methods"]:
        assert b["methods"][name]["var"] == pytest.approx(a["methods"][name]["var"], rel=1e-12)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_route(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    assert "persistence" in response.json()


def test_risk_route(client):
    response = client.get("/risk", params={"level": 0.99, "horizon": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["level"] == 0.99
    assert set(body["methods"]) == {"hs", "parametric_normal", "parametric_t", "fhs"}


def test_risk_route_rejects_a_level_outside_the_open_unit_interval(client):
    assert client.get("/risk", params={"level": 1.5}).status_code == 422
    assert client.get("/risk", params={"level": 0.2}).status_code == 422


def test_conditional_route_accepts_a_fresh_observation(client):
    response = client.post("/risk/conditional", json={"returns": [-0.03, 0.01], "level": 0.99})
    assert response.status_code == 200
    assert response.json()["observations"] > 0


def test_conditional_route_rejects_a_sentinel_for_a_missing_day(client):
    """Contract two: a NaN in the payload is an error, not a zero.

    Silently treating a missing return as zero pulls every volatility estimate
    down, and a risk number that is too low because a day was rounded to nothing is
    the worst possible failure for this service.
    """
    response = client.post("/risk/conditional", json={"returns": [0.01, None, -0.02]})
    assert response.status_code == 422


def test_conditional_route_rejects_an_infinite_value(client):
    """Sent raw, because a JSON *serialiser* will not emit ``Infinity``.

    The payload is built by hand rather than with ``json=``: most clients refuse to
    encode a non-finite float at all, so the only way to reach the server's check is
    to send the literal that ``json.loads`` accepts.
    """
    response = client.post(
        "/risk/conditional",
        content='{"returns": [0.01, Infinity]}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_conditional_route_accepts_an_empty_payload(client):
    """An empty list means "no new data", which is a legitimate request."""
    response = client.post("/risk/conditional", json={})
    assert response.status_code == 200


def test_coverage_route(client):
    assert client.get("/coverage").status_code == 200


# ---------------------------------------------------------------------------
# The standalone window helper
# ---------------------------------------------------------------------------
def test_risk_from_window_needs_no_filter(returns):
    r = returns.to_numpy(dtype=float)
    out = api.risk_from_window(r, 0.99, 1, 250, 5.0)
    assert {"hs", "parametric_normal", "parametric_t", "fhs"} <= set(out)
    for values in out.values():
        assert np.isfinite(values["var"])
        assert values["var"] > 0.0


def test_tick_loss_diagnostics_reports_a_breach_rate(returns):
    r = returns.to_numpy(dtype=float)
    var_path = np.full(r.size, 0.02)
    payload = api.tick_loss_diagnostics(r, var_path, 0.99)
    assert 0.0 <= payload["breach_rate"] <= 1.0
    assert np.isfinite(payload["tick_loss_bp"])
