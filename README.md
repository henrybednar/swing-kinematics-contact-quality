# Swing Kinematics & Contact Quality

Sport analytics capstone: how six swing kinematics predict contact quality in MLB hitters.

Uses Statcast bat-tracking data for all qualified hitters from 2023-2025 (408 player-seasons). Bat speed, swing length, attack angle, attack direction, ideal attack angle rate, and swing path tilt are used to predict two outcomes — average exit velocity and expected weighted on-base average on contact (xwOBAcon) — via multiple linear regression with heteroscedasticity-consistent standard errors and year as a covariate.

Bat speed is the dominant predictor of both outcomes, but swing path tilt and attack angle matter specifically for whether hard contact turns into run value, while contributing little to raw exit velocity — suggesting swing plane governs *where* the ball goes more than *how hard* it leaves the bat.

## Contents

- **`writeup.pdf`** — full write-up: methodology, results, and discussion.
- **`swing_kinematics_model.py`** — fits both regressions: standardizes the six kinematics, adds year as a categorical covariate, runs HC3-robust OLS for each outcome, prints the standardized coefficients/R²/F-stats, and plots a standardized-coefficient comparison (Figure 1).
- **`requirements.txt`** — `pandas`, `statsmodels`, `matplotlib`.

## Data

Baseball Savant's [Bat Tracking leaderboard](https://baseballsavant.mlb.com/leaderboard/bat-tracking), filtered to qualified batters, pulled separately for the 2023, 2024, and 2025 seasons and stacked into one 408-row player-season table. Not included in this repo — see the script's docstring for the exact columns expected.

## Tech stack

Python, `pandas`, `statsmodels`, `matplotlib`.
