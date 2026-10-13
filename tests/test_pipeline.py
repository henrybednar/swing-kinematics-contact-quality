"""End-to-end smoke tests on synthetic data shaped like the Savant extract."""

import numpy as np
import pandas as pd
import pytest

import figures
import swing_kinematics_model as skm
from config import OUTCOMES, PREDICTORS
from report import build_report


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n_players, seasons = 120, (2023, 2024, 2025)
    pid = np.repeat(np.arange(n_players), len(seasons))
    raw = pd.DataFrame({
        "last_name, first_name": [f"Player, {i}" for i in pid],
        "player_id": pid,
        "year": np.tile(seasons, n_players),
        **{p: rng.normal(size=pid.size) for p in PREDICTORS},
    })
    raw["exit_velocity_avg"] = 89 + 1.8 * raw["avg_swing_speed"] + rng.normal(size=pid.size)
    raw["xwobacon"] = 0.39 + 0.04 * raw["avg_swing_speed"] + rng.normal(0, 0.03, pid.size)
    zs = {p + "_z": (raw[p] - raw[p].mean()) / raw[p].std()
          for p in PREDICTORS + ["exit_velocity_avg"]}
    years = {f"year_{y}": (raw["year"] == y).astype(int) for y in (2024, 2025)}
    return raw.assign(**zs, **years)


@pytest.fixture(scope="module")
def models(synthetic_df):
    return skm.fit_models(synthetic_df)


def test_fit_models_returns_one_clustered_model_per_outcome(models):
    assert set(models) == set(OUTCOMES)
    assert all(m.cov_type == "cluster" for m in models.values())


def test_build_report_contains_every_section(synthetic_df, models, tmp_path):
    data_path = tmp_path / "data.csv"
    synthetic_df.to_csv(data_path, index=False)
    text = build_report(synthetic_df, models, data_path, tmp_path / "missing.meta.json")
    for heading in ["Standard errors", "Predictor importance", "Out-of-sample fit",
                    "Mixed model", "Sensitivity", "Nonlinearity", "Model 3",
                    "Holm-adjusted"]:
        assert heading in text
    assert "unknown (no .meta.json" in text
    assert "N = 360 player-seasons from 120 unique players" in text


def test_figures_are_written(synthetic_df, models, tmp_path, monkeypatch):
    monkeypatch.setattr(skm, "FIG_DIR", tmp_path / "figures")
    monkeypatch.setattr(skm, "OUTPUT_DIR", tmp_path)
    skm.make_figures(synthetic_df, models)
    written = sorted(p.name for p in (tmp_path / "figures").iterdir())
    assert written == ["figure1_scatter.png", "figure2_coefficients_exit_velocity_avg.png",
                       "figure3_coefficients_xwobacon.png", "figure4_comparison.png"]


def test_p_text_formats_small_and_large_values():
    assert figures.p_text(0.0004) == "p < .001"
    assert figures.p_text(0.0412) == "p = 0.041"


def test_load_data_drops_missing_rows_and_adds_model_columns(synthetic_df, tmp_path,
                                                             monkeypatch):
    raw = synthetic_df[["last_name, first_name", "player_id", "year"]
                       + PREDICTORS + list(OUTCOMES)].copy()
    raw.loc[0, "attack_angle"] = np.nan
    path = tmp_path / "data.csv"
    raw.to_csv(path, index=False)
    monkeypatch.setattr(skm, "DATA_PATH", path)
    df = skm.load_data()
    assert len(df) == len(raw) - 1
    assert {"avg_swing_speed_z", "exit_velocity_avg_z", "year_2024", "year_2025"} <= set(df)
    assert df["avg_swing_speed_z"].mean() == pytest.approx(0, abs=1e-12)


def test_load_data_exits_with_a_hint_when_csv_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(skm, "DATA_PATH", tmp_path / "nope.csv")
    with pytest.raises(SystemExit, match="get_data.py"):
        skm.load_data()


def test_console_sections_run(synthetic_df, models, capsys):
    skm.print_descriptives(synthetic_df)
    skm.print_diagnostics(synthetic_df, models)
    out = capsys.readouterr().out
    assert "descriptive only" in out
    assert "VIF" in out and "Total unique players: 120" in out


def test_sensitivity_refits_use_the_right_samples(synthetic_df):
    from report import sensitivity_section
    text = "\n".join(sensitivity_section(synthetic_df, "exit_velocity_avg"))
    assert "full: N = 360" in text
    assert "no_2023: N = 240" in text  # 120 players x 2 seasons, fit with year_2025 only
    low_n = int(text.split("low_influence: N = ")[1].split(",")[0])
    assert 300 < low_n < 360


def test_ev_control_model_adds_exit_velocity_only_to_model3(synthetic_df, models):
    from report import ev_control_section
    text = "\n".join(ev_control_section(synthetic_df, models))
    ev_row = next(line for line in text.splitlines() if line.startswith("| Exit Velocity |"))
    cells = [c.strip() for c in ev_row.strip("|").split("|")]
    assert cells[1:3] == ["—", "—"]           # absent from Model 2
    assert all(c != "—" for c in cells[3:5])  # estimated in Model 3


def test_provenance_reads_pull_date_and_hashes_file(tmp_path):
    import hashlib
    import json

    from report import provenance_lines
    data = tmp_path / "d.csv"
    data.write_bytes(b"a,b\n1,2\n")
    meta = tmp_path / "d.meta.json"
    meta.write_text(json.dumps({"pulled_at_utc": "2026-10-13T04:28:55+00:00"}))
    line = provenance_lines(data, meta)[0]
    assert "2026-10-13T04:28:55+00:00" in line
    assert hashlib.sha256(b"a,b\n1,2\n").hexdigest()[:16] in line
