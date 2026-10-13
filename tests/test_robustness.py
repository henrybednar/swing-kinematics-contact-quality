import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

import robustness as rb


def make_panel(n_players=150, seasons=3, player_sd=1.0, noise_sd=0.5, seed=0):
    """Repeated player-seasons where both x and the error share a player component."""
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_players), seasons)
    player_x = rng.normal(size=n_players)[pid]
    player_err = rng.normal(scale=player_sd, size=n_players)[pid]
    x = player_x + rng.normal(scale=0.3, size=pid.size)
    noise = rng.normal(size=pid.size)
    y = 2.0 * x + player_err + rng.normal(scale=noise_sd, size=pid.size)
    return pd.DataFrame({"player_id": pid, "x": x, "noise": noise, "y": y})


def test_clustered_se_exceeds_hc3_when_errors_share_a_player_component():
    df = make_panel()
    table = rb.se_comparison(df, "y ~ x", "player_id", ["x"])
    assert table.loc["x", "SE_ratio"] > 1.2


def test_clustered_se_close_to_hc3_when_errors_are_independent():
    df = make_panel(n_players=1000, player_sd=0.0)
    table = rb.se_comparison(df, "y ~ x", "player_id", ["x"])
    assert table.loc["x", "SE_ratio"] == pytest.approx(1.0, abs=0.1)


def test_fit_ols_rejects_missing_cluster_ids():
    df = make_panel().astype({"player_id": float})
    df.loc[0, "player_id"] = np.nan
    with pytest.raises(ValueError, match="drop NAs"):
        rb.fit_ols(df, "y ~ x", "player_id")


def test_delta_r2_is_large_for_true_predictor_and_near_zero_for_noise():
    df = make_panel()
    d = rb.delta_r2(df, "y", ["x", "noise"])
    assert d["x"] > 0.3
    assert 0 <= d["noise"] < 0.01


def test_grouped_cv_never_puts_a_player_in_train_and_test():
    df = make_panel()
    for train, test in GroupKFold(5).split(df, groups=df["player_id"]):
        assert set(df["player_id"].iloc[train]).isdisjoint(df["player_id"].iloc[test])


def test_grouped_cv_r2_reports_five_folds_and_sensible_r2():
    df = make_panel()
    res = rb.grouped_cv_r2(df, "y", ["x"], "player_id")
    assert len(res["fold_r2"]) == 5
    assert 0.5 < res["pooled_oof_r2"] < 1.0


def test_mixed_model_recovers_slope_and_detects_player_variance():
    df = make_panel()
    m = rb.fit_mixed(df, "y ~ x", "player_id")
    assert m.params["x"] == pytest.approx(2.0, abs=0.2)
    assert rb.intraclass_correlation(m) > 0.5


def test_to_markdown_renders_header_and_rows():
    frame = pd.DataFrame({"a": [1.23456, 1e-9]}, index=pd.Index(["r1", "r2"], name="term"))
    md = rb.to_markdown(frame).splitlines()
    assert md[0] == "| term | a |"
    assert md[2] == "| r1 | 1.2346 |"
    assert md[3] == "| r2 | 1.00e-09 |"


def test_holm_adjust_matches_hand_computation_and_keeps_index():
    p = pd.Series([0.01, 0.04, 0.03], index=["a", "b", "c"])
    adj = rb.holm_adjust(p)
    # sorted: a .01*3=.03, c .03*2=.06, b max(.06, .04*1)=.06
    assert adj.to_dict() == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_low_influence_rows_drops_a_planted_outlier():
    df = make_panel()
    df.loc[0, "y"] += 50
    keep = rb.low_influence_rows(df, "y ~ x")
    assert not keep.loc[0]
    assert keep.mean() > 0.9


def test_to_markdown_renders_missing_values_as_dash():
    frame = pd.DataFrame({"a": [np.nan]}, index=["r1"])
    assert rb.to_markdown(frame).splitlines()[2] == "| r1 | — |"
