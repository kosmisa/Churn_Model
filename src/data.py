"""
Data loading and cleaning.

Responsibilities (kept separate from feature engineering on purpose):
  * load the CSV with correct dtypes,
  * repair values that are recoverable from their own definition,
  * turn missing values into (flag + imputed value) pairs.

No target leakage happens here. I only touch raw observation window columns.
"""
from __future__ import annotations
from collections.abc import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as cfg


def load_raw(path=cfg.DATA_PATH) -> pd.DataFrame:
    """Load the CSV, parsing date columns as datetimes."""
    df = pd.read_csv(path, parse_dates=cfg.DATE_COLS)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw frame.

    Two distinct kinds of missing data and  are handled differently:

    1. <avg_bet_amount> is *deterministically recoverable*. The data_dictionary.md 
        defines it as <total_turnover / live_bets_count>. I alsoverified that 
        this feature holds exactly on the non missing rows, so i can reconstruct the
        690 missing values instead of mean imputing them (that way there should be no information loss).

    2. Deposit columns are *genuinely unknown* when missing (NaN and 0 coexist in the data). 
        I will add an explicit <*_was_missing> flag so the model can
        learn whether "we have no deposit record" is itself predictive, then fill
        the value with 0 as a neutral placeholder.
    """
    df = df.copy()

    # 1. Deterministic reconstruction of avg_bet_amount from its definition.
    recon = df["total_turnover"] / df["live_bets_count"].replace(0, np.nan)
    df["avg_bet_amount"] = df["avg_bet_amount"].fillna(recon)

    # 2. Deposit columns: preserve the "unknown" signal, then fill.
    df["deposit_info_missing"] = df["deposit_count"].isna().astype(int)
    for col in cfg.DEPOSIT_COLS:
        df[col] = df[col].fillna(0.0)

    # A handful of singleton NaNs (a player with no bet in the window, etc.).
    # ggr_margin is undefined when turnover is 0 -> 0 is the neutral value.
    df["ggr_margin"] = df["ggr_margin"].fillna(0.0)

    return df


def load_clean(path=cfg.DATA_PATH) -> pd.DataFrame:
    cleaned = clean(load_raw(path))
    return cleaned


def iqr_outlier_summary(df: pd.DataFrame, cols: Iterable[str], k: float = 1.5,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
    - summary table per feature (IQR bounds, outlier counts, percentages)
    - boolean mask DataFrame (same index as df, one column per feature)
    """
    rows = []
    masks = {}

    for col in cols:
        s = df[col]
        valid = s.dropna()

        # Handle empty/constant columns safely
        if valid.empty:
            q1 = q3 = iqr = lower = upper = np.nan
            mask = pd.Series(False, index=df.index)
        else:
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - k * iqr
            upper = q3 + k * iqr

            mask = (s < lower) | (s > upper)

        masks[col] = mask

        n_out = int(mask.sum())
        n_valid = int(s.notna().sum())
        pct_out = (100.0 * n_out / n_valid) if n_valid else np.nan

        rows.append(
            {
                "feature": col,
                "n_valid": n_valid,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "n_outliers": n_out,
                "outlier_%": pct_out,
            }
        )

    summary = (pd.DataFrame(rows).sort_values(["outlier_%", "n_outliers"], ascending=False).reset_index(drop=True))
    mask_df = pd.DataFrame(masks, index=df.index)

    return summary, mask_df


def plot_distr(
    df: pd.DataFrame,
    cols: Iterable[str],
    target: str | None = None,
    ncols: int = 3,
    bins: int = 50,
    kde: bool = True,
) -> None:
    """
    Plot a distribution (histogram + optional KDE) for each numeric column.
    """

    cols = list(cols)
    n = len(cols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    i = -1
    for i, col in enumerate(cols):
        ax = axes[i]
        if target is not None:
            for cls, color in zip(sorted(df[target].unique()), ["#4C72B0", "#C44E52"]):
                sns.histplot(
                    df.loc[df[target] == cls, col].dropna(),
                    bins=bins, kde=kde, ax=ax, color=color,
                    stat="density", element="step", alpha=0.4,
                    label=f"{target}={cls}",
                )
            ax.legend(fontsize=8)
        else:
            sns.histplot(df[col].dropna(), bins=bins, kde=kde, ax=ax, color="#4C72B0")

        ax.set_title(col)
        unit = cfg.UNITS.get(col, "value")
        ax.set_xlabel(f"{col} [{unit}]")

    # hide empty panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()