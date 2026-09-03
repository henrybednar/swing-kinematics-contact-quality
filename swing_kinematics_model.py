"""
Swing Kinematics as Predictors of Contact Quality in Major League Baseball Hitters
Henry Bednar | University of Mississippi | Sport Analytics Capstone 2025

Data: Baseball Savant bat tracking leaderboard (2023-2025)
      CSV downloaded from: https://baseballsavant.mlb.com/leaderboard/bat-tracking

Requirements:
    pip install pandas scipy statsmodels matplotlib
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as oi
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── 0. Configuration ──────────────────────────────────────────────────────────

DATA_PATH = "Swing_Metrics_2023-2025.csv"

PREDICTORS = [
    "avg_swing_speed",       # Bat speed (mph)
    "avg_swing_length",      # Swing length (ft)
    "attack_angle",          # Attack angle (°)
    "attack_direction",      # Attack direction (°)
    "ideal_angle_rate",      # Ideal attack angle rate (%)
    "vertical_swing_path",   # Swing path tilt (°)
]

OUTCOMES = {
    "exit_velocity_avg": "Average Exit Velocity (mph)",
    "xwobacon":          "Expected Weighted On-Base Average on Contact",
}

LABELS = {
    "avg_swing_speed":    "Bat Speed",
    "avg_swing_length":   "Swing Length",
    "attack_angle":       "Attack Angle",
    "attack_direction":   "Attack Direction",
    "ideal_angle_rate":   "Ideal Angle %",
    "vertical_swing_path":"Swing Path Tilt",
}

# Plot palette
BLUE   = "#3A7DC9"
ORANGE = "#E07B39"
DARK   = "#0F1923"
LGRAY  = "#8A9BAD"
WHITE  = "#FFFFFF"
DGRAY  = "#1C2B3A"


# ── 1. Load data ──────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Seasons: {sorted(df['year'].unique())}")
print(f"N (player-seasons): {len(df)}")


# ── 2. Standardize predictors (z-score) ───────────────────────────────────────
# Standardizing before regression:
#   - reduces multicollinearity (lowers condition number)
#   - makes standardized coefficients directly comparable across predictors
#   - since SD_x = 1 after z-scoring, b_standardized = b_unstandardized / SD_y

for p in PREDICTORS:
    df[p + "_z"] = (df[p] - df[p].mean()) / df[p].std()

PREDS_Z = [p + "_z" for p in PREDICTORS]

# Add year dummies (2023 = reference category)
df["year_2024"] = (df["year"] == 2024).astype(int)
df["year_2025"] = (df["year"] == 2025).astype(int)

formula_base = (
    "avg_swing_speed_z + avg_swing_length_z + attack_angle_z + "
    "attack_direction_z + ideal_angle_rate_z + vertical_swing_path_z + "
    "year_2024 + year_2025"
)


# ── 3. Descriptive statistics ─────────────────────────────────────────────────

print("\n=== DESCRIPTIVE STATISTICS ===")
desc_cols = PREDICTORS + list(OUTCOMES.keys())
desc = df[desc_cols].describe().loc[["mean", "std", "min", "max"]]
print(desc.round(3).to_string())


# ── 4. Pearson correlations ───────────────────────────────────────────────────

print("\n=== PEARSON CORRELATIONS WITH OUTCOMES ===")
for outcome, label in OUTCOMES.items():
    print(f"\n  {label}:")
    for p in PREDICTORS:
        r, pval = stats.pearsonr(df[p], df[outcome])
        sig = "***" if pval < .001 else ("**" if pval < .01 else ("*" if pval < .05 else ""))
        print(f"    {LABELS[p]:20s}  r = {r:6.3f}{sig:4s}  p = {pval:.4g}")


# ── 5. Multiple linear regressions ────────────────────────────────────────────
# HC3 heteroscedasticity-consistent standard errors correct for unequal
# residual variance identified by the Breusch-Pagan test (see diagnostics).
# Year included as a fixed covariate to account for pooled three-season structure.

models = {}
for outcome in OUTCOMES:
    formula = f"{outcome} ~ {formula_base}"
    m = smf.ols(formula, data=df).fit(cov_type="HC3")
    models[outcome] = m

    print(f"\n{'='*60}")
    print(f"MODEL: {outcome}")
    print(f"{'='*60}")
    print(f"  R²          = {m.rsquared:.3f}")
    print(f"  Adjusted R² = {m.rsquared_adj:.3f}")
    print(f"  F({int(m.df_model)}, {int(m.df_resid)}) = {m.fvalue:.2f},  p = {m.f_pvalue:.2e}")
    print(f"  N           = {int(m.nobs)}")
    print()

    sd_y = df[outcome].std()
    print(f"  {'Predictor':<25}  {'b':>8}  {'SE':>7}  {'β':>7}  {'p':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}")
    for pz in PREDS_Z + ["year_2024", "year_2025"]:
        b    = m.params[pz]
        se   = m.bse[pz]
        pval = m.pvalues[pz]
        beta = b / sd_y          # standardized coefficient
        sig  = "***" if pval < .001 else ("**" if pval < .01 else ("*" if pval < .05 else ""))
        name = LABELS.get(pz.replace("_z", ""), pz)
        print(f"  {name:<25}  {b:8.4f}  {se:7.4f}  {beta:7.3f}  {pval:8.4g} {sig}")


# ── 6. Diagnostics ────────────────────────────────────────────────────────────

print("\n=== DIAGNOSTICS ===")

# 6a. Variance Inflation Factors
print("\n  VIF (predictors only, standardized):")
X = df[PREDS_Z].copy()
for i, pz in enumerate(PREDS_Z):
    vif = oi.variance_inflation_factor(X.values, i)
    flag = " ← HIGH" if vif > 10 else ""
    print(f"    {LABELS[pz.replace('_z','')]:20s}  VIF = {vif:.2f}{flag}")

# 6b. Condition number (after standardization)
m_plain = smf.ols(f"exit_velocity_avg ~ {formula_base}", data=df).fit()
print(f"\n  Condition number (standardized): {m_plain.condition_number:.1f}")

# 6c. Residual normality (Shapiro-Wilk)
print("\n  Shapiro-Wilk normality test on residuals:")
for outcome, m in models.items():
    stat, pval = stats.shapiro(m.resid)
    sk = float(m.resid.skew())
    ku = float(m.resid.kurtosis())
    print(f"    {outcome}: W = {stat:.4f}, p = {pval:.4g}, skew = {sk:.3f}, kurt = {ku:.3f}")

# 6d. Heteroscedasticity (Breusch-Pagan)
print("\n  Breusch-Pagan heteroscedasticity test:")
for outcome, m in models.items():
    lm, lm_p, _, _ = het_breuschpagan(m.resid, m.model.exog)
    print(f"    {outcome}: LM = {lm:.3f}, p = {lm_p:.4g}")

# 6e. Influential observations (Cook's D)
print("\n  Influential observations (Cook's D > 4/n):")
threshold = 4 / len(df)
for outcome, m in models.items():
    cooks = m.get_influence().cooks_distance[0]
    flagged = np.where(cooks > threshold)[0]
    print(f"    {outcome}: {len(flagged)} observations flagged")
    for i in flagged[:3]:
        print(f"      idx={i}, {df.iloc[i]['last_name, first_name']}, "
              f"{df.iloc[i]['year']}, D={cooks[i]:.4f}")

# 6f. Repeated-measures structure
print("\n  Repeated-measures structure:")
counts = df["player_id"].value_counts()
for n in [1, 2, 3]:
    print(f"    Players appearing {n}x: {(counts == n).sum()}")
print(f"    Total unique players: {counts.shape[0]}")


# ── 7. Figures ────────────────────────────────────────────────────────────────

def style_ax(ax):
    ax.set_facecolor(DARK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(LGRAY)
    ax.tick_params(colors=WHITE, labelsize=11)
    ax.yaxis.label.set_color(WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)


# Figure 1 — Bat speed vs exit velocity scatter
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
fig.patch.set_facecolor(DARK)
color_map = {2023: BLUE, 2024: "#C8A951", 2025: ORANGE}
for yr, grp in df.groupby("year"):
    ax.scatter(grp["avg_swing_speed"], grp["exit_velocity_avg"],
               s=22, alpha=0.55, color=color_map[yr], edgecolor="none", label=str(yr))

m_ev = models["exit_velocity_avg"]
xs = np.linspace(df["avg_swing_speed"].min(), df["avg_swing_speed"].max(), 200)
xs_z = (xs - df["avg_swing_speed"].mean()) / df["avg_swing_speed"].std()
yhat = m_ev.params["Intercept"] + m_ev.params["avg_swing_speed_z"] * xs_z
ax.plot(xs, yhat, color=WHITE, lw=2, label="Regression line")

ax.set_xlabel("Average Bat Speed (mph)", fontsize=13)
ax.set_ylabel("Average Exit Velocity (mph)", fontsize=13)
ax.set_title("Figure 1. Bat Speed vs. Exit Velocity  |  r = .727, p < .001",
             fontsize=13, fontweight="bold", color=WHITE)
ax.legend(fontsize=10, frameon=False, labelcolor=WHITE)
style_ax(ax)
fig.tight_layout()
fig.savefig("figure1_scatter.png", bbox_inches="tight", facecolor=DARK)
plt.close()
print("\nFigure 1 saved: figure1_scatter.png")


# Figures 2 & 3 — Standardized coefficients per model
for outcome, label, color, fignum in [
    ("exit_velocity_avg", "Exit Velocity",    BLUE,   2),
    ("xwobacon",          "xwOBAcon",         ORANGE, 3),
]:
    m = models[outcome]
    sd_y = df[outcome].std()
    betas = [m.params[pz] / sd_y for pz in PREDS_Z]
    sigs  = [m.pvalues[pz] < .05 for pz in PREDS_Z]
    xlabs = [LABELS[p] for p in PREDICTORS]
    x = np.arange(len(xlabs))

    r2 = m.rsquared
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    fig.patch.set_facecolor(DARK)
    bars = ax.bar(x, betas,
                  color=[color if s else "#3A4A5A" for s in sigs],
                  edgecolor="none", width=0.58)
    for bar, sig in zip(bars, sigs):
        if not sig:
            bar.set_hatch("////")
            bar.set_edgecolor(LGRAY)
    ax.axhline(0, color=LGRAY, lw=1)
    for bar, val in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.03 if val >= 0 else -0.075),
                f"{val:.2f}", ha="center", color=WHITE, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabs, fontsize=12, color=WHITE)
    ax.set_ylabel("Standardized Coefficient (β)", fontsize=12)
    ax.set_title(f"Figure {fignum}. Model {fignum-1}: {label}  |  R² = {r2:.3f}",
                 fontsize=13, fontweight="bold", color=WHITE)
    ax.legend(
        handles=[Patch(facecolor="#3A4A5A", edgecolor=LGRAY, hatch="////",
                       label="Non-significant (p ≥ .05)")],
        fontsize=11, frameon=False, labelcolor=WHITE, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    fname = f"figure{fignum}_coefficients_{outcome}.png"
    fig.savefig(fname, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"Figure {fignum} saved: {fname}")


# Figure 4 — Side-by-side comparison
sd_ev = df["exit_velocity_avg"].std()
sd_xw = df["xwobacon"].std()
std_ev = [models["exit_velocity_avg"].params[pz] / sd_ev for pz in PREDS_Z]
std_xw = [models["xwobacon"].params[pz] / sd_xw for pz in PREDS_Z]
sig_ev = [models["exit_velocity_avg"].pvalues[pz] < .05 for pz in PREDS_Z]
sig_xw = [models["xwobacon"].pvalues[pz] < .05 for pz in PREDS_Z]
xlabs  = [LABELS[p] for p in PREDICTORS]
x = np.arange(len(xlabs))
ymin = min(min(std_ev), min(std_xw)) - 0.12
ymax = max(max(std_ev), max(std_xw)) + 0.16

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
fig.patch.set_facecolor(DARK)
for ax, title, betas, sigs, col, r2 in zip(
    axes,
    ["Model 1: Exit Velocity", "Model 2: xwOBAcon"],
    [std_ev, std_xw],
    [sig_ev, sig_xw],
    [BLUE, ORANGE],
    [models["exit_velocity_avg"].rsquared, models["xwobacon"].rsquared],
):
    bars = ax.bar(x, betas,
                  color=[col if s else "#3A4A5A" for s in sigs],
                  edgecolor="none", width=0.58)
    for bar, sig in zip(bars, sigs):
        if not sig:
            bar.set_hatch("////")
            bar.set_edgecolor(LGRAY)
    ax.axhline(0, color=LGRAY, lw=1)
    for bar, val in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.025 if val >= 0 else -0.065),
                f"{val:.2f}", ha="center", color=WHITE, fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabs, fontsize=9.5, color=WHITE)
    ax.set_ylabel("Standardized Coefficient (β)", fontsize=11, color=WHITE)
    ax.set_title(f"{title}  |  R² = {r2:.3f}", fontsize=11, fontweight="bold", color=WHITE)
    ax.set_ylim(ymin, ymax)
    style_ax(ax)

fig.legend(
    handles=[Patch(facecolor="white", edgecolor=LGRAY, hatch="////",
                   label="Non-significant (p ≥ .05)")],
    fontsize=8.5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Figure 4. Standardized Coefficients by Predictor and Outcome (2023–2025)",
             fontsize=10.5, fontweight="bold", color=WHITE, y=1.02)
fig.tight_layout(pad=2.5)
fig.savefig("figure4_comparison.png", bbox_inches="tight", facecolor=DARK)
plt.close()
print("Figure 4 saved: figure4_comparison.png")

print("\nDone. All models and figures complete.")
