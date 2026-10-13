"""Builds output/robustness_report.md and echoes each table to the console."""

import hashlib
import json
import warnings
from pathlib import Path

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper

import robustness as rb
from config import (
    CLUSTER_COL,
    CV_FOLDS,
    OUTCOMES,
    PREDS_Z,
    SHORT_OUTCOME,
    TERM_LABELS,
    YEAR_TERMS,
    model_formula,
)

Models = dict[str, RegressionResultsWrapper]
EV_Z = "exit_velocity_avg_z"
BAT_SPEED_SQ = "I(avg_swing_speed_z ** 2)"


def labelled(frame: pd.DataFrame, index_name: str) -> pd.DataFrame:
    out = frame.rename(index=TERM_LABELS)
    out.index.name = index_name
    return out


def emit(heading: str, frame: pd.DataFrame, note: str = "") -> list[str]:
    """Print a table to the console and return it as markdown lines."""
    print(f"\n  {heading}:")
    print(frame.round(4).to_string())
    return ["", f"### {heading}", ""] + ([note, ""] if note else []) + [rb.to_markdown(frame)]


def provenance_lines(data_path: Path, meta_path: Path) -> list[str]:
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    pulled = "unknown (no .meta.json; re-run get_data.py)"
    if meta_path.exists():
        pulled = json.loads(meta_path.read_text(encoding="utf-8"))["pulled_at_utc"]
    return [f"Data: `{data_path.name}`, pulled {pulled}, SHA-256 `{digest[:16]}`. "
            "Savant revises its data, so a later pull can shift these numbers slightly."]


def sample_lines(df: pd.DataFrame) -> list[str]:
    counts = df[CLUSTER_COL].value_counts()
    seasons = ", ".join(str(y) for y in sorted(df["year"].unique()))
    reps = " / ".join(str((counts == k).sum()) for k in (1, 2, 3))
    return [f"N = {len(df)} player-seasons from {counts.size} unique players (seasons {seasons}); "
            f"players appearing 1x / 2x / 3x: {reps}.", "",
            "All models include season fixed effects (2023 = reference); predictors are z-scored; "
            "p-values use player-clustered SEs unless labeled HC3."]


def se_section(df: pd.DataFrame, outcome: str) -> list[str]:
    table = rb.se_comparison(df, model_formula(outcome), CLUSTER_COL, PREDS_Z + YEAR_TERMS)
    return emit("Standard errors: HC3 vs. clustered by player", labelled(table, "Term"))


def importance_section(df: pd.DataFrame, model: RegressionResultsWrapper,
                       outcome: str) -> list[str]:
    imp = labelled(pd.DataFrame({
        "beta": model.params[PREDS_Z] / df[outcome].std(),
        "delta_R2": rb.delta_r2(df, outcome, PREDS_Z, YEAR_TERMS),
    }), "Predictor").sort_values("delta_R2", ascending=False)
    note = (f"Full-model R² = {model.rsquared:.3f}. ΔR² is the drop in R² when that predictor "
            "alone is removed (squared semipartial correlation).")
    return emit("Predictor importance", imp, note)


def cv_section(df: pd.DataFrame, model: RegressionResultsWrapper, outcome: str) -> list[str]:
    # LinearRegression is scale-invariant, so z-scoring on the full sample
    # does not leak into the out-of-fold predictions.
    cv = rb.grouped_cv_r2(df, outcome, PREDS_Z + YEAR_TERMS, CLUSTER_COL, CV_FOLDS)
    folds = ", ".join(f"{r:.3f}" for r in cv["fold_r2"])
    print(f"\n  Player-grouped {CV_FOLDS}-fold CV: in-sample R² {model.rsquared:.3f}, "
          f"mean fold R² {cv['mean_r2']:.3f} ({cv['sd_r2']:.3f}), "
          f"pooled out-of-fold R² {cv['pooled_oof_r2']:.3f}")
    return ["", f"### Out-of-sample fit (GroupKFold by player, k = {CV_FOLDS})", "",
            "| In-sample R² | Mean fold R² (SD) | Pooled out-of-fold R² | Fold R² |",
            "|---|---|---|---|",
            f"| {model.rsquared:.3f} | {cv['mean_r2']:.3f} ({cv['sd_r2']:.3f}) | "
            f"{cv['pooled_oof_r2']:.3f} | {folds} |"]


def mixed_section(df: pd.DataFrame, model: RegressionResultsWrapper, outcome: str) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mm = rb.fit_mixed(df, model_formula(outcome), CLUSTER_COL)
    fit_warnings = sorted({str(w.message) for w in caught})
    table = labelled(pd.DataFrame({
        "b_OLS": model.params[PREDS_Z],
        "SE_OLS_cluster": model.bse[PREDS_Z],
        "b_mixed": mm.params[PREDS_Z],
        "SE_mixed": mm.bse[PREDS_Z],
        "p_mixed": mm.pvalues[PREDS_Z],
    }), "Predictor")
    note = (f"Share of residual variance between players (ICC) = "
            f"{rb.intraclass_correlation(mm):.3f}. Optimizer converged: {mm.converged}."
            + "".join(f" statsmodels warning: {w}" for w in fit_warnings))
    return emit("Mixed model: random intercept per player (REML)", table, note)


def sensitivity_section(df: pd.DataFrame, outcome: str) -> list[str]:
    specs = {
        "full": (df, tuple(YEAR_TERMS)),
        "no_2023": (df[df["year"] > 2023].reset_index(drop=True), ("year_2025",)),
        "low_influence": (df[rb.low_influence_rows(df, model_formula(outcome))]
                          .reset_index(drop=True), tuple(YEAR_TERMS)),
    }
    sd_y = df[outcome].std()
    cols, fits = {}, []
    for name, (sub, years) in specs.items():
        m = rb.fit_ols(sub, model_formula(outcome, year_terms=years), CLUSTER_COL)
        cols[f"beta_{name}"] = m.params[PREDS_Z] / sd_y
        cols[f"p_{name}"] = m.pvalues[PREDS_Z]
        fits.append(f"{name}: N = {int(m.nobs)}, R² = {m.rsquared:.3f}")
    note = ("`no_2023` drops the partially tracked 2023 season; `low_influence` drops rows with "
            "Cook's D > 4/n. β uses the full-sample SD of the outcome. " + "; ".join(fits) + ".")
    return emit("Sensitivity: partial 2023 season and influential points",
                labelled(pd.DataFrame(cols), "Predictor"), note)


def nonlinearity_section(df: pd.DataFrame, model: RegressionResultsWrapper,
                         outcome: str) -> list[str]:
    m = rb.fit_ols(df, model_formula(outcome, (BAT_SPEED_SQ,)), CLUSTER_COL)
    table = labelled(pd.DataFrame({
        "b": [m.params[BAT_SPEED_SQ]], "SE": [m.bse[BAT_SPEED_SQ]],
        "p": [m.pvalues[BAT_SPEED_SQ]], "delta_R2": [m.rsquared - model.rsquared],
    }, index=[BAT_SPEED_SQ]), "Term")
    return emit("Nonlinearity: squared bat-speed term added", table,
                "A positive coefficient means each extra mph of bat speed is worth more "
                "at the top of the range.")


def outcome_sections(df: pd.DataFrame, models: Models, outcome: str) -> list[str]:
    print(f"\n{'=' * 60}\nROBUSTNESS: {outcome}\n{'=' * 60}")
    m = models[outcome]
    return (["", f"## {OUTCOMES[outcome]} (`{outcome}`)"]
            + se_section(df, outcome) + importance_section(df, m, outcome)
            + cv_section(df, m, outcome) + mixed_section(df, m, outcome)
            + sensitivity_section(df, outcome) + nonlinearity_section(df, m, outcome))


def holm_section(models: Models) -> list[str]:
    pvals = pd.concat({SHORT_OUTCOME[o]: m.pvalues[PREDS_Z] for o, m in models.items()})
    pvals.index = [f"{o}: {TERM_LABELS[t]}" for o, t in pvals.index]
    table = pd.DataFrame({"p_cluster": pvals, "p_holm": rb.holm_adjust(pvals)})
    table.index.name = "Test"
    return ["", "## Multiple comparisons"] + emit(
        "Holm-adjusted p-values across all 12 kinematic coefficients", table,
        "Family = six kinematics × two primary models.")


def ev_control_section(df: pd.DataFrame, models: Models) -> list[str]:
    """Model 3: xwOBAcon with exit velocity held fixed, i.e. contact value beyond raw speed."""
    base = models["xwobacon"]
    m3 = rb.fit_ols(df, model_formula("xwobacon", (EV_Z,)), CLUSTER_COL)
    terms = PREDS_Z + [EV_Z]
    sd_y = df["xwobacon"].std()
    table = labelled(pd.DataFrame({
        "beta_model2": base.params.reindex(terms) / sd_y,
        "p_model2": base.pvalues.reindex(terms),
        "beta_model3": m3.params[terms] / sd_y,
        "p_model3": m3.pvalues[terms],
    }), "Predictor")
    share = 1 - m3.params["avg_swing_speed_z"] / base.params["avg_swing_speed_z"]
    note = (f"Model 2 R² = {base.rsquared:.3f}; Model 3 R² = {m3.rsquared:.3f}. Holding exit "
            f"velocity fixed shrinks the bat-speed coefficient by {share:.0%}, the share of its "
            "xwOBAcon association that runs through how hard the ball is hit.")
    return ["", "## Model 3: xwOBAcon controlling for exit velocity"] + emit(
        "Model 2 vs. Model 3", table, note)


def build_report(df: pd.DataFrame, models: Models, data_path: Path, meta_path: Path) -> str:
    lines = ["# Robustness report", "",
             "Generated by `swing_kinematics_model.py`; do not edit by hand.", ""]
    lines += provenance_lines(data_path, meta_path) + [""] + sample_lines(df)
    for outcome in OUTCOMES:
        lines += outcome_sections(df, models, outcome)
    lines += ev_control_section(df, models) + holm_section(models)
    return "\n".join(lines) + "\n"
