"""
Download the 2023-2025 bat-tracking dataset from Baseball Savant.

Writes Swing_Metrics_2023-2025.csv (one row per qualified player-season), a
.meta.json sidecar recording when it was pulled, and prints how many swings
were tracked per season for the hitters in the analysis sample.

    python get_data.py
"""

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "Swing_Metrics_2023-2025.csv"
META_PATH = ROOT / "Swing_Metrics_2023-2025.meta.json"

SEASONS = (2023, 2024, 2025)
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT_S = 60

SELECTIONS = [
    "avg_swing_speed", "avg_swing_length", "attack_angle", "attack_direction",
    "ideal_angle_rate", "vertical_swing_path", "exit_velocity_avg", "xwobacon",
]
ID_COLS = ["last_name, first_name", "player_id", "year"]

# Custom leaderboard, qualified batters (min=q), one season per request.
CUSTOM_URL = (
    "https://baseballsavant.mlb.com/leaderboard/custom"
    "?year={year}&type=batter&filter=&min=q&selections={selections}&csv=true"
)
# Bat-tracking leaderboard for every hitter with at least one tracked swing,
# used only to count the swings behind each sampled player-season.
BAT_TRACKING_URL = (
    "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
    "?gameType=Regular&minSwings=1&minGroupSwings=1"
    "&seasonStart={year}&seasonEnd={year}&type=batter&csv=true"
)


def parse_csv_text(text: str, source: str) -> pd.DataFrame:
    """Parse a Savant CSV body, rejecting the HTML/empty pages Savant returns on bad params."""
    if not text.strip() or text.lstrip().startswith("<"):
        raise ValueError(f"Expected CSV, got HTML or empty body from {source}")
    frame = pd.read_csv(io.StringIO(text))
    if frame.empty:
        raise ValueError(f"CSV had no rows: {source}")
    return frame


def read_csv_url(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_S)
    resp.raise_for_status()
    return parse_csv_text(resp.content.decode("utf-8-sig"), url)


def select_columns(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    missing = set(ID_COLS + SELECTIONS) - set(frame.columns)
    if missing:
        raise ValueError(f"{year}: Savant CSV is missing columns {sorted(missing)}")
    return frame[ID_COLS + SELECTIONS]


def pull_season(year: int) -> pd.DataFrame:
    url = CUSTOM_URL.format(year=year, selections=",".join(SELECTIONS))
    return select_columns(read_csv_url(url), year)


def pull_tracked_swings(year: int) -> pd.DataFrame:
    frame = read_csv_url(BAT_TRACKING_URL.format(year=year))
    return frame[["id", "swings_competitive"]].assign(year=year)


def coverage_for_sample(sample: pd.DataFrame, tracked: pd.DataFrame) -> pd.DataFrame:
    """Tracked competitive swings per season for the player-seasons in `sample`."""
    merged = sample[["player_id", "year"]].merge(
        tracked.rename(columns={"id": "player_id"}), on=["player_id", "year"], how="left")
    return merged.groupby("year")["swings_competitive"].agg(
        player_seasons="size", matched="count", median_swings="median", min_swings="min",
    ).reset_index()


def build_metadata(data: pd.DataFrame, pulled_at: datetime) -> dict:
    return {
        "pulled_at_utc": pulled_at.isoformat(timespec="seconds"),
        "source": CUSTOM_URL.split("?")[0],
        "seasons": list(SEASONS),
        "rows": len(data),
    }


def main() -> None:
    # Fetch everything before writing, so a failed request never leaves a
    # half-updated CSV/meta pair on disk.
    data = pd.concat([pull_season(y) for y in SEASONS], ignore_index=True)
    tracked = pd.concat([pull_tracked_swings(y) for y in SEASONS], ignore_index=True)

    data.to_csv(OUT_PATH, index=False)
    meta = build_metadata(data, datetime.now(timezone.utc))
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUT_PATH.name}: {len(data)} player-seasons, "
          f"{data['player_id'].nunique()} unique players (pulled {meta['pulled_at_utc']})")

    print("\nTracked competitive swings behind each sampled player-season "
          "(bat tracking was rolled out partway through 2023):")
    print(coverage_for_sample(data, tracked).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, ValueError, KeyError) as err:
        sys.exit(f"Download failed: {err}")
