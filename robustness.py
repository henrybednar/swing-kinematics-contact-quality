"""
Robustness checks for the swing-kinematics regressions.

Each function takes a dataframe that has ALREADY had missing values dropped on
every column the model uses. That keeps the cluster groups row-aligned with the
design matrix; passing groups from an un-dropped frame would either fail on
length or silently misalign players.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper
from statsmodels.stats.multitest import multipletests


def fit_ols(df: pd.DataFrame, formula: str,
            cluster_col: str | None = None) -> RegressionResultsWrapper:
    """OLS with player-clustered SEs if `cluster_col` is given, else HC3."""
    model = smf.ols(formula, data=df)
    if cluster_col is None:
        return model.fit(cov_type="HC3")
    if df[cluster_col].isna().any():
        raise ValueError(f"'{cluster_col}' has missing values; drop NAs before fitting")
    return model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col].to_numpy()})


def se_comparison(df: pd.DataFrame, formula: str, cluster_col: str,
                  terms: list[str]) -> pd.DataFrame:
    """Side-by-side HC3 vs. clustered SEs and p-values for `terms`."""
    hc3 = fit_ols(df, formula)
    cl = fit_ols(df, formula, cluster_col)
    return pd.DataFrame({
        "b": hc3.params[terms],
        "SE_HC3": hc3.bse[terms],
        "SE_cluster": cl.bse[terms],
        "SE_ratio": cl.bse[terms] / hc3.bse[terms],
        "p_HC3": hc3.pvalues[terms],
        "p_cluster": cl.pvalues[terms],
    })


def delta_r2(df: pd.DataFrame, outcome: str, terms: list[str],
             controls: list[str] | tuple[str, ...] = ()) -> pd.Series:
    """Drop in R² when each term is removed from the full model (squared semipartial r)."""
    def r2(rhs: list[str]) -> float:
        return smf.ols(f"{outcome} ~ {' + '.join(rhs)}", data=df).fit().rsquared

    full_rhs = list(terms) + list(controls)
    full = r2(full_rhs)
    drops = {t: full - r2([x for x in full_rhs if x != t]) for t in terms}
    return pd.Series(drops, name="delta_R2")


def grouped_cv_r2(df: pd.DataFrame, outcome: str, x_cols: list[str], group_col: str,
                  n_splits: int = 5) -> dict:
    """Out-of-sample R² with folds split by player, so no hitter is in both train and test."""
    X = df[x_cols].to_numpy()
    y = df[outcome].to_numpy()
    groups = df[group_col].to_numpy()
    cv = GroupKFold(n_splits=n_splits)
    fold_r2 = cross_val_score(LinearRegression(), X, y, groups=groups, cv=cv, scoring="r2")
    oof = cross_val_predict(LinearRegression(), X, y, groups=groups, cv=cv)
    return {
        "fold_r2": fold_r2,
        "mean_r2": float(fold_r2.mean()),
        "sd_r2": float(fold_r2.std(ddof=1)),
        "pooled_oof_r2": float(r2_score(y, oof)),
    }


def fit_mixed(df: pd.DataFrame, formula: str, group_col: str) -> MixedLMResultsWrapper:
    """Linear mixed model with a random intercept per player (REML)."""
    return smf.mixedlm(formula, data=df, groups=df[group_col]).fit(reml=True)


def intraclass_correlation(mixed_result: MixedLMResultsWrapper) -> float:
    """Share of residual variance (after fixed effects) that sits between players."""
    between = float(mixed_result.cov_re.iloc[0, 0])
    return between / (between + float(mixed_result.scale))


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values for one family of tests, same index as input."""
    _, adjusted, _, _ = multipletests(pvalues.to_numpy(), method="holm")
    return pd.Series(adjusted, index=pvalues.index, name="p_holm")


def low_influence_rows(df: pd.DataFrame, formula: str,
                       threshold: float | None = None) -> pd.Series:
    """Boolean mask of rows with Cook's D at or below `threshold` (default 4/n) in plain OLS."""
    cooks = smf.ols(formula, data=df).fit().get_influence().cooks_distance[0]
    cutoff = 4 / len(df) if threshold is None else threshold
    return pd.Series(cooks <= cutoff, index=df.index)


def to_markdown(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Minimal GitHub-markdown table, so the report needs no `tabulate` dependency."""
    def fmt(v: object) -> str:
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:.2e}" if 0 < abs(v) < 1e-4 else format(v, floatfmt)
        return str(v)

    cols = [frame.index.name or ""] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in frame.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)
