# Swing Kinematics & Contact Quality

Sport analytics capstone: how six swing kinematics predict contact quality in MLB hitters.

Uses Statcast bat-tracking data for all qualified hitters from 2023-2025 (408 player-seasons). Bat speed, swing length, attack angle, attack direction, ideal attack angle rate, and swing path tilt are used to predict two outcomes — average exit velocity and expected weighted on-base average on contact (xwOBAcon) — via multiple linear regression with heteroscedasticity-consistent standard errors and year as a covariate.

Bat speed is the dominant predictor of both outcomes, but swing path tilt and attack angle matter specifically for whether hard contact turns into run value, while contributing little to raw exit velocity — suggesting swing plane governs *where* the ball goes more than *how hard* it leaves the bat.

## Contents

- **`writeup.pdf`** — full write-up: methodology, results, and discussion.

## Status

Analysis code to follow.
