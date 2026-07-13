"""
Data loading and cleaning.

Responsibilities (kept separate from feature engineering on purpose):
  * load the CSV with correct dtypes,
  * repair values that are recoverable from their own definition,
  * turn missing values into (flag + imputed value) pairs.

No target leakage happens here. I only touch raw observation window columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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
