"""Shared paths, variable names, and model formulas."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Swing_Metrics_2023-2025.csv"
META_PATH = ROOT / "Swing_Metrics_2023-2025.meta.json"
OUTPUT_DIR = ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "robustness_report.md"

CLUSTER_COL = "player_id"
CV_FOLDS = 5
VIF_FLAG = 5

PREDICTORS = [
    "avg_swing_speed",       # Bat speed (mph)
    "avg_swing_length",      # Swing length (ft)
    "attack_angle",          # Attack angle (°)
    "attack_direction",      # Attack direction (°)
    "ideal_angle_rate",      # Ideal attack angle rate (%)
    "vertical_swing_path",   # Swing path tilt (°)
]
PREDS_Z = [p + "_z" for p in PREDICTORS]

# Season fixed effects: year dummies (2023 = reference category) absorb
# league-wide shifts across seasons (ball, measurement, swing trends).
YEAR_TERMS = ["year_2024", "year_2025"]

OUTCOMES = {
    "exit_velocity_avg": "Average Exit Velocity (mph)",
    "xwobacon":          "Expected Weighted On-Base Average on Contact",
}
SHORT_OUTCOME = {"exit_velocity_avg": "Exit Velocity", "xwobacon": "xwOBAcon"}

LABELS = {
    "avg_swing_speed":     "Bat Speed",
    "avg_swing_length":    "Swing Length",
    "attack_angle":        "Attack Angle",
    "attack_direction":    "Attack Direction",
    "ideal_angle_rate":    "Ideal Angle %",
    "vertical_swing_path": "Swing Path Tilt",
}
TERM_LABELS = {
    **{p + "_z": label for p, label in LABELS.items()},
    "year_2024": "Season 2024",
    "year_2025": "Season 2025",
    "exit_velocity_avg_z": "Exit Velocity",
    "I(avg_swing_speed_z ** 2)": "Bat Speed²",
}


def model_formula(outcome: str, extra_terms: tuple[str, ...] = (),
                  year_terms: tuple[str, ...] = tuple(YEAR_TERMS)) -> str:
    """Primary specification: six z-scored kinematics + season dummies (+ any extras)."""
    return f"{outcome} ~ " + " + ".join(PREDS_Z + list(year_terms) + list(extra_terms))
