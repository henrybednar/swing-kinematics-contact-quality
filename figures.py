"""Figures for the swing-kinematics analysis. Every number drawn comes from the data passed in."""

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from scipy import stats

BLUE = "#3A7DC9"
ORANGE = "#E07B39"
GOLD = "#C8A951"
DARK = "#0F1923"
LGRAY = "#8A9BAD"
WHITE = "#FFFFFF"
NONSIG = "#3A4A5A"
YEAR_COLORS = {2023: BLUE, 2024: GOLD, 2025: ORANGE}

SIG_NOTE = "Hatched: not significant (p ≥ .05, player-clustered SEs)"


def style_ax(ax: Axes) -> None:
    ax.set_facecolor(DARK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(LGRAY)
    ax.tick_params(colors=WHITE, labelsize=11)
    ax.yaxis.label.set_color(WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)


def nonsig_handle() -> Patch:
    return Patch(facecolor=NONSIG, edgecolor=LGRAY, hatch="////", label=SIG_NOTE)


def p_text(p: float) -> str:
    return "p < .001" if p < .001 else f"p = {p:.3f}"


def plot_scatter(df: pd.DataFrame, x: str, y: str, path: Path) -> None:
    """Figure 1: scatter by season with the simple OLS fit that the title's r describes."""
    r, p = stats.pearsonr(df[x], df[y])
    slope, intercept = np.polyfit(df[x], df[y], 1)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor(DARK)
    for yr, grp in df.groupby("year"):
        ax.scatter(grp[x], grp[y], s=22, alpha=0.55, color=YEAR_COLORS[yr],
                   edgecolor="none", label=str(yr))
    xs = np.linspace(df[x].min(), df[x].max(), 200)
    ax.plot(xs, intercept + slope * xs, color=WHITE, lw=2,
            label=f"Simple OLS fit ({slope:.2f} mph per mph)")

    ax.set_xlabel("Average Bat Speed (mph)", fontsize=13)
    ax.set_ylabel("Average Exit Velocity (mph)", fontsize=13)
    ax.set_title(f"Figure 1. Bat Speed vs. Exit Velocity  |  r = {r:.3f}, {p_text(p)}",
                 fontsize=13, fontweight="bold", color=WHITE)
    ax.legend(fontsize=10, frameon=False, labelcolor=WHITE)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)


def draw_coef_bars(ax: Axes, betas: Sequence[float], sigs: Sequence[bool],
                   color: str, labels: Sequence[str], fontsize: float) -> None:
    """Standardized-coefficient bars; non-significant bars are grey and hatched."""
    x = np.arange(len(labels))
    bars = ax.bar(x, betas, color=[color if s else NONSIG for s in sigs],
                  edgecolor="none", width=0.58)
    for bar, sig in zip(bars, sigs):
        if not sig:
            bar.set_hatch("////")
            bar.set_edgecolor(LGRAY)
    ax.axhline(0, color=LGRAY, lw=1)
    pad = 0.03 * (max(betas) - min(min(betas), 0))
    for bar, val in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + pad if val >= 0 else val - pad,
                f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                color=WHITE, fontsize=fontsize, fontweight="bold")
    lo, hi = min(min(betas), 0), max(max(betas), 0)
    span = hi - lo
    ax.set_ylim(lo - 0.15 * span, hi + 0.12 * span)
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace(" ", "\n", 1) for lab in labels],
                       fontsize=fontsize, color=WHITE)
    ax.set_ylabel("Standardized Coefficient (β)", fontsize=fontsize + 1.5)
    style_ax(ax)


def plot_coefficients(betas: Sequence[float], sigs: Sequence[bool], labels: Sequence[str],
                      color: str, title: str, path: Path) -> None:
    """Figures 2 and 3: one model's standardized coefficients."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    fig.patch.set_facecolor(DARK)
    draw_coef_bars(ax, betas, sigs, color, labels, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color=WHITE)
    ax.legend(handles=[nonsig_handle()], fontsize=10, frameon=False,
              labelcolor=WHITE, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)


def plot_comparison(panels: Sequence[dict], labels: Sequence[str], title: str,
                    path: Path) -> None:
    """Figure 4: models side by side on a shared y-axis.

    Each panel is a dict with keys betas, sigs, color, title.
    """
    all_betas = [b for panel in panels for b in panel["betas"]]
    ylim = (min(min(all_betas), 0) - 0.12, max(all_betas) + 0.16)

    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5.2), dpi=150)
    fig.patch.set_facecolor(DARK)
    for ax, panel in zip(axes, panels):
        draw_coef_bars(ax, panel["betas"], panel["sigs"], panel["color"], labels, fontsize=9.5)
        ax.set_title(panel["title"], fontsize=11, fontweight="bold", color=WHITE)
        ax.set_ylim(*ylim)
    fig.legend(handles=[nonsig_handle()], fontsize=9, loc="upper center", frameon=False,
               labelcolor=WHITE, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=10.5, fontweight="bold", color=WHITE, y=1.02)
    fig.tight_layout(pad=2.5)
    fig.savefig(path, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
