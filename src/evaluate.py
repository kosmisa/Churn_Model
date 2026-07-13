from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

"""Evaluation utilities, chosen for a *retention targeting* use case.

The business does not consume a raw 0/1 label. It consumes a ranked list of players 
to contact within a limited budget. 
So beyond just ROC-AUC report i use:
  * PR-AUC        -  Performance under 17% class imbalance,
  * lift @ top-k  - "how many more churners do we catch vs random?",
  * precision @ k - "of the top-k we contact, how many really churn?",
  * calibration   - are the probabilities trustworthy as risk scores?
"""

def threshold_free_metrics(y_true, y_prob) -> dict:
    """
    Metrics that do not depend on a decision threshold.
    1. ROC-AUC: Area under the Receiver Operating Characteristic curve.
    2. PR-AUC: Area under the Precision-Recall curve.
    3. Brier score: Mean squared error of the predicted probabilities. For this the lower it is the better.
    """
    metrics = { 
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }

    return metrics


def threshold_metrics(y_true, y_prob, threshold=0.5) -> dict:
    """Metrics at a specific probability cut-off."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics

def topk_report(y_true, y_prob, ks=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Precision, recall and lift when contacting the top-k% riskiest players.

    ``lift`` = precision in segment / base churn rate. A lift of 3 means the that top-k% 
    list contains 3x more churners than a random sample of the same size.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(y_prob)[::-1]  # highest risk first.
    y_sorted = y_true[order]
    base_rate = y_true.mean()
    n = len(y_true)
    total_churn = y_true.sum()

    rows = []
    for k in ks:
        cutoff = max(1, int(round(k * n)))
        seg = y_sorted[:cutoff]
        precision = seg.mean()
        rows.append(
            {
                "top_k": f"{int(k * 100)}%",
                "n_contacted": cutoff,
                "precision": round(precision, 3),
                "recall": round(seg.sum() / total_churn, 3),
                "lift": round(precision / base_rate, 2),
            }
        )
    return pd.DataFrame(rows)


def cv_stability(scores: dict[str, list]) -> pd.DataFrame:
    """Summarise per fold scores as mean +/- std to demonstrate stability."""
    df = pd.DataFrame({
            metric: {"mean": np.mean(vals), 
                     "std": np.std(vals)}

            for metric, vals in scores.items()
        }).T.round(4)

    return df