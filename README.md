# Swing Kinematics & Contact Quality

Sport analytics capstone: how six swing kinematics predict contact quality in MLB hitters.

Uses Statcast bat-tracking data for all qualified hitters from 2023-2025 (408 player-seasons). Bat speed, swing length, attack angle, attack direction, ideal attack angle rate, and swing path tilt are used to predict two outcomes — average exit velocity and expected weighted on-base average on contact (xwOBAcon) — via multiple linear regression with heteroscedasticity-consistent standard errors and year as a covariate.

Bat speed is the dominant predictor of both outcomes, but swing path tilt and attack angle matter specifically for whether hard contact turns into run value, while contributing little to raw exit velocity — suggesting swing plane governs *where* the ball goes more than *how hard* it leaves the bat.

## Contents

- **`writeup.pdf`** — full write-up: methodology, results, and discussion.
- **`swing_kinematics_model.py`** — the full analysis: descriptive stats, Pearson correlations, two HC3-robust OLS regressions (standardized six kinematics + year covariate) for exit velocity and xwOBAcon, a diagnostics suite (VIF, condition number, Shapiro-Wilk residual normality, Breusch-Pagan heteroscedasticity, Cook's D influential-observation checks, repeated-measures structure), and four figures (bat speed vs. exit velocity scatter, per-model standardized-coefficient bar charts, and a side-by-side comparison).
- **`requirements.txt`** — `pandas`, `scipy`, `statsmodels`, `matplotlib`.

## Data

Baseball Savant's [Bat Tracking leaderboard](https://baseballsavant.mlb.com/leaderboard/bat-tracking), filtered to qualified batters, pulled separately for the 2023, 2024, and 2025 seasons and stacked into one 408-row player-season table (expected as `Swing_Metrics_2023-2025.csv`). Not included in this repo.

## Tech stack

Python, `pandas`, `scipy`, `statsmodels`, `matplotlib`.
