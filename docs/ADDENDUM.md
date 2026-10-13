# Addendum to the write-up: robustness checks

`writeup.pdf` is the capstone as submitted in 2025. This addendum covers the checks added afterward. The full tables are in [`output/robustness_report.md`](../output/robustness_report.md), which `swing_kinematics_model.py` regenerates. That report records the data pull date and file hash. The numbers below come from the October 2026 pull.

Savant has revised its data since 2025. In-sample R² barely moved (.586 and .661, compared with .587 and .663 in the write-up), but some smaller exit-velocity coefficients did. For example, ideal attack angle rate went from β = .027 to .105. To keep data revisions separate from method changes, every comparison below is made on the same current data.

## Methods

The 408 player-seasons come from 219 hitters: 89 appear once, 71 twice and 59 three times. Seasons from the same hitter share that hitter's talent, so their residuals are correlated. HC3 standard errors correct for unequal variance but assume independent observations, so they are too small here. The primary models now use standard errors clustered by player, which allow for both problems. Both models keep season fixed effects (2024 and 2025 dummies, with 2023 as the reference). Rows with a missing model variable are dropped before fitting so the cluster IDs stay aligned with the design matrix; in the current pull no rows are missing.

Out-of-sample fit uses 5-fold cross-validation grouped by player (`GroupKFold`), so the same hitter never appears in both training and test folds. As a second answer to the repeated-player problem, each model is also fit as a linear mixed model with a random intercept per player (REML). Predictor importance is reported two ways: standardized coefficients, and ΔR², which is the drop in R² when that predictor alone is removed.

Several further checks were added. The models are refit without the partially tracked 2023 season and without high-influence observations (Cook's D > 4/n). A squared bat-speed term tests for curvature. Holm-adjusted p-values cover all 12 kinematic coefficients across the two models. A third model regresses xwOBAcon on the kinematics while also controlling for exit velocity.

## What changed

**Clustering widens standard errors by about 5 to 32 percent** (by a factor of 1.05 to 1.32 across the six kinematics). Bat speed is unaffected (p < 10⁻⁶³ in both models). In the exit-velocity model, ideal attack angle rate (HC3 p = .024, clustered p = .060) and swing path tilt (HC3 p = .022, clustered p = .069) are no longer significant at .05. The xwOBAcon model keeps the same significant set: bat speed, swing length, attack angle, attack direction and swing path tilt.

This supports the write-up's central claim. Swing path tilt adds 0.6 points of R² to exit velocity and is no longer significant there. For xwOBAcon it adds 4.8 points and remains at p < .001. Attack angle is null for exit velocity (ΔR² ≈ 0) but contributes 1.6 points to xwOBAcon.

**"Bat speed is the dominant predictor" is now a measured result.** Dropping bat speed alone costs 45 points of R² in both models (0.455 for exit velocity, 0.447 for xwOBAcon). No other predictor costs more than 5 points.

**Swing plane affects run value beyond how hard the ball is hit.** xwOBAcon is calculated partly from exit velocity, so Model 3 adds exit velocity as a control. Swing path tilt (β = .24) and attack angle (β = .20) stay significant at p < 10⁻⁸. That is the most direct evidence for the claim that swing plane governs where the ball goes. Attack direction drops out (p = .69), so its link to xwOBAcon runs through exit velocity. Bat speed's coefficient shrinks by 61%, meaning most of its xwOBAcon association comes from hitting the ball harder.

**Multicollinearity is low.** All VIFs are between 1.2 and 2.5, and the condition number is 5.0. Attack angle's null result for exit velocity is therefore not caused by collinearity with the other kinematics.

**Out-of-sample R² falls modestly.** With player-grouped CV, pooled out-of-fold R² is .535 for exit velocity (in-sample .586) and .624 for xwOBAcon (in-sample .661). Across folds, exit velocity ranges from .32 to .69, so its fit varies more depending on which hitters are held out.

**The mixed model gives the same significance call (at .05) as clustered OLS for every predictor in both models.** Between-player differences account for a large share of residual variance (ICC .63 for exit velocity, .55 for xwOBAcon). The one estimate that moves is attack angle on exit velocity, which goes from −0.02 to +0.20 (SE 0.12, p = .098). That estimate is unstable and imprecise, so the data do not support any claim about attack angle's effect on raw exit velocity. The xwOBAcon mixed model reports a boundary warning from statsmodels. It converges to the same estimates with either optimizer, and its between-player variance is far from zero.

**The results do not depend on 2023 or on influential points.** Dropping 2023 (N = 274) leaves every significance call in both models unchanged, and exit-velocity R² rises to .627. Dropping the high-influence rows (N = 385 for exit velocity) also leaves the xwOBAcon conclusions unchanged. In the exit-velocity model it moves ideal angle rate back under .05 (p = .020), which is one more sign that ideal angle rate's effect on exit velocity is fragile.

**Bat speed's effect on xwOBAcon curves upward.** A squared bat-speed term is significant for xwOBAcon (p = .010, ΔR² = .012) but not for exit velocity (p = .12). Each extra mph is worth slightly more run value at the top of the bat-speed range.

**Multiple comparisons.** Across the 12 kinematic tests, Holm adjustment keeps every result the write-up relies on except attack direction in the xwOBAcon model (adjusted p = .053). That result is borderline, and Model 3 shows it runs through exit velocity anyway.

## Limitations

- **2023 is a partial season.** Bat tracking was rolled out during 2023. In this sample, 2023 player-seasons rest on a median of 432 tracked competitive swings (minimum 168), compared with 950 in 2024 and 952 in 2025 (`get_data.py` prints this check). The outcomes, exit velocity and xwOBAcon, cover the full season. Dropping 2023 does not change the conclusions.
- **Observational design.** This is cross-sectional association across hitters. It does not show that adding bat speed to a given hitter raises their exit velocity, and the model does not support a causal claim about bat speed training.
- **Qualified hitters only.** This selects for hitters good enough to play every day, which restricts the range of every variable and likely weakens the correlations.
- **Seasonal averages.** Kinematics and outcomes are season means, so swing intent, pitch type and count are not modeled.
- **Data revisions.** Savant revises its data, so a fresh pull can shift results slightly. The report's data line identifies the exact file used.

## Short description

Player-clustered OLS with season fixed effects, validated with player-grouped cross-validation, a random-intercept mixed model, and sensitivity checks.
