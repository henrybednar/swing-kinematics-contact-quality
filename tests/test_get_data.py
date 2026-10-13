from datetime import datetime, timezone

import pandas as pd
import pytest

import get_data as gd


def test_parse_csv_text_rejects_html_and_empty_bodies():
    with pytest.raises(ValueError, match="HTML or empty"):
        gd.parse_csv_text("<!DOCTYPE html><html></html>", "url")
    with pytest.raises(ValueError, match="HTML or empty"):
        gd.parse_csv_text("   ", "url")


def test_parse_csv_text_rejects_header_only_csv():
    with pytest.raises(ValueError, match="no rows"):
        gd.parse_csv_text("a,b\n", "url")


def test_select_columns_names_missing_fields():
    frame = pd.DataFrame({c: [1] for c in gd.ID_COLS + gd.SELECTIONS[:-1]})
    with pytest.raises(ValueError, match="xwobacon"):
        gd.select_columns(frame, 2024)


def test_select_columns_keeps_only_model_fields_in_order():
    frame = pd.DataFrame({c: [1] for c in ["extra"] + gd.SELECTIONS[::-1] + gd.ID_COLS})
    assert list(gd.select_columns(frame, 2024).columns) == gd.ID_COLS + gd.SELECTIONS


def test_coverage_for_sample_counts_only_sampled_player_seasons():
    sample = pd.DataFrame({"player_id": [1, 2, 1], "year": [2023, 2023, 2024]})
    tracked = pd.DataFrame({
        "id": [1, 2, 3, 1],
        "swings_competitive": [100, 300, 9999, 800],
        "year": [2023, 2023, 2023, 2024],
    })
    cov = gd.coverage_for_sample(sample, tracked).set_index("year")
    assert cov.loc[2023, "player_seasons"] == 2
    assert cov.loc[2023, "median_swings"] == 200
    assert cov.loc[2023, "min_swings"] == 100
    assert cov.loc[2024, "median_swings"] == 800


def test_coverage_for_sample_reports_unmatched_players():
    sample = pd.DataFrame({"player_id": [1, 2], "year": [2023, 2023]})
    tracked = pd.DataFrame({"id": [1], "swings_competitive": [100], "year": [2023]})
    cov = gd.coverage_for_sample(sample, tracked).iloc[0]
    assert (cov["player_seasons"], cov["matched"]) == (2, 1)


def test_build_metadata_records_pull_time_and_rows():
    data = pd.DataFrame({"player_id": range(5)})
    meta = gd.build_metadata(data, datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc))
    assert meta["pulled_at_utc"] == "2026-10-12T15:00:00+00:00"
    assert meta["rows"] == 5
    assert meta["seasons"] == [2023, 2024, 2025]
