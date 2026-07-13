from __future__ import annotations

from pathlib import Path

"""
Central configuration: paths, column groups and the leakage policy.

Keeping these in one place makes the leakage decision (the single most
important modelling choice in a churn task) explicit and auditable, rather
than scattered across notebook cells.
"""

# Paths ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "churn_dataset.csv"

TARGET = "churn"
ID_COL = "user_id"

# Reference date (from the dict).
SNAPSHOT_DATE = "2025-12-31"
OBSERVATION_START = "2025-10-03"
OUTCOME_WINDOW_DAYS = 30

# Column groups ---------------------------------
DATE_COLS = [
    "registration_date",
    "observation_start_date",
    "observation_end_date",
    "last_live_bet_date",
    "first_live_bet_date",
]


UNITS = {
    "total_turnover": "EUR",
    "total_payout": "EUR",
    "total_deposit_amount": "EUR",
    "ggr": "EUR",
    "avg_bet_amount": "EUR",
    "net_cash_flow": "EUR",
    "days_since_last_bet": "days",
    "tenure_days": "days",
    "days_active_in_observation": "days",
    "live_bets_count": "count",
    "deposit_count": "count",
    "bet_day_rate": "rate (0-1)",
    "ggr_margin": "ratio",
    "deposit_to_turnover_ratio": "ratio",
    }

# Columns that I WILL NOT use as model features.
LEAKAGE_COLS = {
    # Defines the target itself -> I will be using it as a look into the future.
    "live_bets_count_outcome_window": "target is derived from this column",
    # Constant for every row in this snapshot -> zero signal, only noise. I dont want to use it as a feature.
    "observation_start_date": "constant (single snapshot)",
    "observation_end_date": "constant (single snapshot)",
}

# Raw dates are not fed to the model directly. They are turned into numeric values.
# recency / tenure features in features.py. We drop the raw strings after that.
DROP_AFTER_FEATURES = DATE_COLS

# Deposit columns where a missing value means "unknown"
DEPOSIT_COLS = ["deposit_count", "total_deposit_amount", "deposit_to_turnover_ratio"]

RANDOM_STATE = 42
