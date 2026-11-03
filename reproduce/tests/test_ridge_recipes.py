"""
Keep recorded regression settings unregularized and label ridge searches explicitly.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "ridge_recipe_runner", ROOT / "reproduce/run.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
STRENGTHS = [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-2]


def recipe_for(method):
    presets = runner.load_presets()
    identity, recipe = next(
        (identity, recipe)
        for identity, recipe in presets["recipes"].items()
        if presets["fit_settings"][recipe["fit"]]["method"] == method
    )
    return presets, identity, recipe


def test_historical_regression_recipes_explicitly_pin_zero():
    presets = runner.load_presets()
    assert len(presets["recipes"]) == 399
    for source in presets["sources"].values():
        assert set(source) == {"campaign_id", "sha256", "row_count"}
    for settings in presets["fit_settings"].values():
        if settings["method"] in ("derivative", "midpoint"):
            assert settings["ridge"] == 0.0
        else:
            assert "ridge" not in settings


@pytest.mark.parametrize("method", ["derivative", "midpoint"])
def test_ridge_recipe_variant_keeps_identity_and_deadline(method, capsys, monkeypatch):
    presets, identity, recipe = recipe_for(method)
    monkeypatch.setitem(sys.modules, "phsi", None)
    runner.main([identity, "--dry-run"])
    historical = json.loads(capsys.readouterr().out)
    runner.main([identity, "--dry-run", "--ridge-search"])
    ridge = json.loads(capsys.readouterr().out)
    assert historical["fit"]["ridge"] == 0.0
    assert ridge["fit"]["ridge"] == STRENGTHS
    assert ridge["recipe_variant"] == "ridge_selection_20261102"
    assert ridge["recipe_id"] == identity
    assert ridge["identity"] == historical["identity"] == recipe
    assert ridge["candidate_count"] == 6 * historical["candidate_count"]
    assert ridge["fit"]["time_limit"] == historical["fit"]["time_limit"] == 3000
    assert ridge["model"] == historical["model"]
    assert presets["fit_settings"][recipe["fit"]]["ridge"] == 0.0


def test_ridge_recipe_requires_one_regression_recipe():
    _, identity, _ = recipe_for("integration_scheme")
    for arguments in (
        ["--dry-run", "--ridge-search"],
        [identity, "--dry-run", "--ridge-search"],
    ):
        with pytest.raises(SystemExit):
            runner.main(arguments)
