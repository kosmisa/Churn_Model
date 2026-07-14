from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from typing import Literal

"""
Feature engineering, organised around the RFM framework.

RFM (Recency, Frequency, Monetary) is the framework that i choose for this churn modeling task.:
  * Recency  - how long since the player was last active? (I take this as thestrongest signal)
  * Frequency- how often / how intensely do they bet?
  * Monetary - how much do they stake, deposit and lose?

I also add engagement intensity and value ratios that normalise raw counts by
the players own tenure / active days, so a casual new player and a heavy veteran are compared fairly.

All features are computed strictly from observation window columns. Raw dates and leakage columns are dropped at the end.
"""

# Observation window length in days (Oct 3 -> Dec 31).
_WINDOW_DAYS = (pd.Timestamp(cfg.SNAPSHOT_DATE) - pd.Timestamp(cfg.OBSERVATION_START)).days

# Function that protects against division by zero
def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element base division that returns 0 where the denominator is 0."""

    result = (a / b.replace(0, np.nan)).fillna(0.0)

    return result

# Function to create new features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a cleaned frame."""

    df = df.copy()
    snap = pd.Timestamp(cfg.SNAPSHOT_DATE)

    # RFM framework implementation

    # RECENCY GROUP-----------------------------------------------------------------------------

    # 1.
    # Idle time relative to how long the player has existed. Idea is that veteran/champions users taht idle for
    # 10 days is less alarming than a 15 day old account that idle for 10 days.
    df["recency_ratio"] = _safe_div(df["days_since_last_bet"], df["tenure_days"])

    # 2.
    # How long ago they placed their FIRST bet in the window (usually onboarding duration).
    df["days_since_first_bet"] = (snap - df["first_live_bet_date"]).dt.days

    # 3.
    # Span of betting activity inside the window (engaged for a long stretch vs a single burst then silence).
    df["active_span_days"] = (df["last_live_bet_date"] - df["first_live_bet_date"]).dt.days.clip(lower=0)

    # 4.
    # Fraction of the window that had already elapsed at the last bet. Low value means player went quiet early in the window.
    df["last_bet_window_position"] = 1.0 - _safe_div(df["days_since_last_bet"], pd.Series(_WINDOW_DAYS, index=df.index))
    # Podsetnik: Probaj da nadjes jos formula za recency

    # FREQUENCY / ENGAGEMENT  GROUP --------------------------------=

    # 5. 
    # Bets per day actually played -> intensity when engaged.
    df["bets_per_active_day"] = _safe_div(df["live_bets_count"], df["days_active_in_observation"])

    # 6.
    # Consistency: active days spread across their span (regular vs bursty).
    df["activity_consistency"] = _safe_div(df["days_active_in_observation"], df["active_span_days"] + 1)


    # MONETARY --------------------------------

    # 7.
    df["turnover_per_active_day"] = _safe_div(df["total_turnover"], df["days_active_in_observation"])

    # 8.
    df["turnover_per_bet"] = _safe_div(df["total_turnover"], df["live_bets_count"])

    # 9.
    df["deposit_per_bet"] = _safe_div(df["total_deposit_amount"], df["live_bets_count"])

    # 10.
    # Net cash flow from the operator's view (deposits minus payouts).
    df["net_cash_flow"] = df["total_deposit_amount"] - df["total_payout"]

    # 11.
    df["has_no_deposit"] = (df["total_deposit_amount"] == 0).astype(int)
    

    # --- VALUE / TENURE ----------------------------------------------------

    # 12.
    df["turnover_per_tenure_day"] = _safe_div(df["total_turnover"], df["tenure_days"])

    # 13.
    df["is_new_player"] = (df["tenure_days"] <= 90).astype(int)

    return df


FeatureSet = Literal["raw", "engineered", "all"]

def build_feature_matrix(df: pd.DataFrame, feature_set: FeatureSet = "all"):
    """
    Return (X, y) ready for modelling.

    feature_set:
      - "raw" : cleaned raw columns only
      - "engineered" : engineered columns only
      - "all" : raw + engineered columns
    """

    if feature_set not in {"raw", "engineered", "all"}:
        raise ValueError(f"Unknown feature_set='{feature_set}'. Use 'raw', 'engineered', or 'all'.")

    base = df.copy()
    y = base[cfg.TARGET].astype(int)

    drop_cols = (
        [cfg.TARGET, cfg.ID_COL]
        + cfg.DROP_AFTER_FEATURES
        + list(cfg.LEAKAGE_COLS.keys())
    )

    if feature_set == "raw":
        X = base.drop(columns=[c for c in drop_cols if c in base.columns])

    else:
        feat = add_features(base)

        if feature_set == "all":
            X = feat.drop(columns=[c for c in drop_cols if c in feat.columns])

        else:
            # engineered only
            engineered_cols = [c for c in feat.columns if c not in base.columns]

            # Defensive filter in case any future engineered names collide with protected columns
            forbidden = set(drop_cols)
            engineered_cols = [c for c in engineered_cols if c not in forbidden]

            X = feat[engineered_cols]

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns leaked into X: {non_numeric}")

    return X, y
