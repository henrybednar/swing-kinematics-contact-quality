"""
Swing Kinematics as Predictors of Contact Quality in MLB Hitters
==================================================================

Two multiple linear regressions testing how six swing kinematics predict
contact quality for qualified MLB hitters, 2023-2025 (408 player-seasons):

  Model 1 (Exit Velocity)  -- avg_exit_velocity ~ 6 kinematics + year
  Model 2 (xwOBAcon)       -- xwobacon           ~ 6 kinematics + year

Both models use heteroscedasticity-consistent (HC3) standard errors and
report standardized coefficients, so predictors are z-scored before fitting.
Year is entered as a categorical covariate (2 dummies against a reference
season), which is why each model has 8 model degrees of freedom
(6 kinematics + 2 year dummies) against 399 residual df on N = 408.

Data source
-----------
Baseball Savant's Bat Tracking leaderboard:
  https://baseballsavant.mlb.com/leaderboard/bat-tracking
Filtered to Batter / qualified, pulled separately for the 2023, 2024, and
2025 seasons and stacked into one player-season table. Columns used:

  avg_bat_speed            -- bat speed
  avg_swing_length         -- swing length
  attack_angle             -- attack angle
  attack_direction         -- attack direction
  ideal_attack_angle_rate  -- ideal attack angle rate
  vertical_swing_path      -- swing path tilt
  exit_velocity_avg        -- Model 1 outcome
  xwobacon                 -- Model 2 outcome
  year                     -- season (2023 / 2024 / 2025)

This script expects the three season exports already merged into
`data/bat_tracking_2023_2025.csv` (one row per qualified batter-season).
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

KINEMATICS = [
    "avg_bat_speed",
    "avg_swing_length",
    "attack_angle",
    "attack_direction",
    "ideal_attack_angle_rate",
    "vertical_swing_path",
]

PRETTY_NAMES = {
    "avg_bat_speed": "Bat Speed",
    "avg_swing_length": "Swing Length",
    "attack_angle": "Attack Angle",
    "attack_direction": "Attack Direction",
    "ideal_attack_angle_rate": "Ideal Attack Angle Rate",
    "vertical_swing_path": "Swing Path Tilt",
}


def load_data(path="data/bat_tracking_2023_2025.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=KINEMATICS + ["exit_velocity_avg", "xwobacon", "year"])

    # Standardize the six kinematics so coefficients are directly comparable
    # (the write-up reports standardized/beta coefficients).
    for col in KINEMATICS:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()

    df["year"] = df["year"].astype(str).astype("category")
    return df


def fit_model(df, outcome):
    z_terms = " + ".join(c + "_z" for c in KINEMATICS)
    formula = f"{outcome} ~ {z_terms} + C(year)"
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    return model


def summarize(model, label):
    print(f"\n{label}")
    print(f"R^2 = {model.rsquared:.3f}, F({int(model.df_model)}, {int(model.df_resid)}) "
          f"= {model.fvalue:.2f}, p = {model.f_pvalue:.3g}")
    for col in KINEMATICS:
        term = col + "_z"
        beta = model.params[term]
        p = model.pvalues[term]
        print(f"  {PRETTY_NAMES[col]:<24} beta = {beta:>6.3f}   p = {p:.3g}")


def plot_coefficients(ev_model, con_model, out_path="figure1_standardized_coefficients.png"):
    labels = [PRETTY_NAMES[c] for c in KINEMATICS]
    ev_betas = [ev_model.params[c + "_z"] for c in KINEMATICS]
    con_betas = [con_model.params[c + "_z"] for c in KINEMATICS]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], ev_betas, width, label="Exit Velocity")
    ax.bar([i + width / 2 for i in x], con_betas, width, label="xwOBAcon")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Standardized coefficient (beta)")
    ax.set_title("Swing Kinematics as Predictors of Contact Quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    df = load_data()

    ev_model = fit_model(df, "exit_velocity_avg")
    con_model = fit_model(df, "xwobacon")

    summarize(ev_model, "Model 1: Exit Velocity")
    summarize(con_model, "Model 2: xwOBAcon")

    plot_coefficients(ev_model, con_model)
