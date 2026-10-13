"""
Swing Kinematics as Predictors of Contact Quality in Major League Baseball Hitters
Henry Bednar | University of Mississippi | Sport Analytics Capstone 2025

Data: Baseball Savant custom leaderboard, bat-tracking fields (2023-2025).
      Run `python get_data.py` first to download Swing_Metrics_2023-2025.csv.

Requirements:
    pip install -r requirements.txt

Outputs (under output/): figures/*.png and robustness_report.md.
"""

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as oi
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

import figures
import robustness as rb
from config import (
    CLUSTER_COL,
    DATA_PATH,
    FIG_DIR,
    LABELS,
    META_PATH,
    OUTCOMES,
    OUTPUT_DIR,
    PREDICTORS,
    PREDS_Z,
    REPORT_PATH,
    SHORT_OUTCOME,
    TERM_LABELS,
    VIF_FLAG,
    YEAR_TERMS,
    model_formula,
)
from report import Models, build_report


def sig_stars(p: float) -> str:
    return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH.name} not found. Run `python get_data.py` first.")
    raw = pd.read_csv(DATA_PATH)
    print(f"Loaded: {raw.shape[0]} rows × {raw.shape[1]} columns")

    # Drop NAs on every model column BEFORE fitting, so the cluster groups passed to
    # statsmodels stay row-aligned with the design matrix. All models share one sample.
    model_cols = PREDICTORS + list(OUTCOMES) + ["year", CLUSTER_COL]
    df = raw.dropna(subset=model_cols).reset_index(drop=True)
    print(f"Dropped for missing values: {len(raw) - len(df)}")
    print(f"Seasons: {sorted(df['year'].unique())}")
    print(f"N (player-seasons): {len(df)}")
    print(f"Unique players (clusters): {df[CLUSTER_COL].nunique()}")

    # z-scoring puts every coefficient on a per-SD scale so they can be compared,
    # and lowers the condition number. It does not change VIFs among the predictors.
    zscore = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    z_cols = {p + "_z": zscore(df[p]) for p in PREDICTORS + ["exit_velocity_avg"]}
    years = {f"year_{y}": (df["year"] == y).astype(int) for y in (2024, 2025)}
    return df.assign(**z_cols, **years)


# ── 2. Descriptives and correlations ──────────────────────────────────────────

def print_descriptives(df: pd.DataFrame) -> None:
    print("\n=== DESCRIPTIVE STATISTICS ===")
    desc = df[PREDICTORS + list(OUTCOMES)].describe().loc[["mean", "std", "min", "max"]]
    print(desc.round(3).to_string())

    print("\n=== PEARSON CORRELATIONS WITH OUTCOMES ===")
    print("  (descriptive only: p-values treat player-seasons as independent)")
    for outcome, label in OUTCOMES.items():
        print(f"\n  {label}:")
        for p in PREDICTORS:
            r, pval = stats.pearsonr(df[p], df[outcome])
            print(f"    {LABELS[p]:20s}  r = {r:6.3f}{sig_stars(pval):4s}  p = {pval:.4g}")


# ── 3. Multiple linear regressions ────────────────────────────────────────────
# Up to three seasons per player share that player's talent, so residuals are
# correlated within player. Standard errors are clustered by player_id, which is
# robust to both that correlation and unequal variance. HC3 (heteroscedasticity
# only) is kept for comparison in the robustness report.

def fit_models(df: pd.DataFrame) -> Models:
    models = {o: rb.fit_ols(df, model_formula(o), CLUSTER_COL) for o in OUTCOMES}
    for outcome, m in models.items():
        print(f"\n{'=' * 60}\nMODEL: {outcome}\n{'=' * 60}")
        print(f"  R²          = {m.rsquared:.3f}")
        print(f"  Adjusted R² = {m.rsquared_adj:.3f}")
        print(f"  F({int(m.df_model)}, {int(m.df_resid)}) = {m.fvalue:.2f},  p = {m.f_pvalue:.2e}")
        print(f"  N           = {int(m.nobs)}  ({df[CLUSTER_COL].nunique()} player clusters)\n")

        sd_y = df[outcome].std()
        print(f"  {'Predictor':<25}  {'b':>8}  {'SE':>7}  {'β':>7}  {'p':>8}")
        print(f"  {'-' * 25}  {'-' * 8}  {'-' * 7}  {'-' * 7}  {'-' * 8}")
        for term in PREDS_Z + YEAR_TERMS:
            b, se, pval = m.params[term], m.bse[term], m.pvalues[term]
            print(f"  {TERM_LABELS[term]:<25}  {b:8.4f}  {se:7.4f}  {b / sd_y:7.3f}  "
                  f"{pval:8.4g} {sig_stars(pval)}")
    return models


# ── 4. Diagnostics ────────────────────────────────────────────────────────────

def print_diagnostics(df: pd.DataFrame, models: Models) -> None:
    print("\n=== DIAGNOSTICS ===")
    print("\n  VIF (predictors only, standardized):")
    X = df[PREDS_Z].to_numpy()
    for i, pz in enumerate(PREDS_Z):
        vif = oi.variance_inflation_factor(X, i)
        flag = f" ← above {VIF_FLAG}" if vif > VIF_FLAG else ""
        print(f"    {TERM_LABELS[pz]:20s}  VIF = {vif:.2f}{flag}")

    cond = smf.ols(model_formula("exit_velocity_avg"), data=df).fit().condition_number
    print(f"\n  Condition number (standardized): {cond:.1f}")

    print("\n  Shapiro-Wilk normality test on residuals:")
    for outcome, m in models.items():
        w, pval = stats.shapiro(m.resid)
        print(f"    {outcome}: W = {w:.4f}, p = {pval:.4g}, "
              f"skew = {m.resid.skew():.3f}, kurt = {m.resid.kurtosis():.3f}")

    print("\n  Breusch-Pagan heteroscedasticity test:")
    for outcome, m in models.items():
        lm, lm_p, _, _ = het_breuschpagan(m.resid, m.model.exog)
        print(f"    {outcome}: LM = {lm:.3f}, p = {lm_p:.4g}")

    print("\n  Influential observations (Cook's D > 4/n; refit without them in the report):")
    for outcome, m in models.items():
        cooks = m.get_influence().cooks_distance[0]
        flagged = np.where(cooks > 4 / len(df))[0]
        print(f"    {outcome}: {len(flagged)} observations flagged")
        for i in flagged[:3]:
            row = df.iloc[i]
            print(f"      idx={i}, {row['last_name, first_name']}, {row['year']}, D={cooks[i]:.4f}")

    print("\n  Repeated-measures structure:")
    counts = df[CLUSTER_COL].value_counts()
    for n in (1, 2, 3):
        print(f"    Players appearing {n}x: {(counts == n).sum()}")
    print(f"    Total unique players: {counts.size}")


# ── 5. Figures ────────────────────────────────────────────────────────────────

def make_figures(df: pd.DataFrame, models: Models) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [LABELS[p] for p in PREDICTORS]
    colors = {"exit_velocity_avg": figures.BLUE, "xwobacon": figures.ORANGE}

    path = FIG_DIR / "figure1_scatter.png"
    figures.plot_scatter(df, "avg_swing_speed", "exit_velocity_avg", path)
    print(f"\nFigure 1 saved: {path.relative_to(OUTPUT_DIR.parent)}")

    panels = []
    for num, (outcome, m) in enumerate(models.items(), start=2):
        panel = {
            "betas": list(m.params[PREDS_Z] / df[outcome].std()),
            "sigs": list(m.pvalues[PREDS_Z] < .05),
            "color": colors[outcome],
            "title": f"Model {num - 1}: {SHORT_OUTCOME[outcome]}  |  R² = {m.rsquared:.3f}",
        }
        panels.append(panel)
        path = FIG_DIR / f"figure{num}_coefficients_{outcome}.png"
        figures.plot_coefficients(panel["betas"], panel["sigs"], labels, panel["color"],
                                  f"Figure {num}. {panel['title']}", path)
        print(f"Figure {num} saved: {path.relative_to(OUTPUT_DIR.parent)}")

    seasons = sorted(df["year"].unique())
    path = FIG_DIR / "figure4_comparison.png"
    title = ("Figure 4. Standardized Coefficients by Predictor and Outcome "
             f"({seasons[0]}–{seasons[-1]})")
    figures.plot_comparison(panels, labels, title, path)
    print(f"Figure 4 saved: {path.relative_to(OUTPUT_DIR.parent)}")


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    # Windows consoles default to cp1252, which cannot print β, R², or ΔR².
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_data()
    print_descriptives(df)
    models = fit_models(df)
    print_diagnostics(df, models)
    make_figures(df, models)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(df, models, DATA_PATH, META_PATH), encoding="utf-8")
    print(f"\nRobustness report saved: {REPORT_PATH.relative_to(OUTPUT_DIR.parent)}")
    print("\nDone. All models and figures complete.")


if __name__ == "__main__":
    main()
