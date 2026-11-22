"""
forecast_utils.py - helper functions shared by notebooks 03, 04 and 05.

Keeping them here (instead of copy-pasting in every notebook) guarantees that
ALL models are evaluated with exactly the same rules.

Vocabulary
----------
origin week  : the week "today". At the end of this week we know everything up to and including it.
target week  : the week we want to predict = origin week + horizon (h) weeks.
fold         : one rolling-window trial = one block of test target weeks + the data we may train on.
"""
import numpy as np
import pandas as pd

BLACK_FRIDAY_WEEKS = ["2017-11-20"]   # Monday of the week containing Black Friday 2017 (24 Nov 2017)


# ----------------------------------------------------------------------------------------------
# 1. Folds (rolling / expanding test windows)
# ----------------------------------------------------------------------------------------------
def make_folds(weeks, first_test=40, test_len=4, step=4):
    """
    Define the test windows in terms of TARGET weeks (positions in the list `weeks`).

    first_test : position of the first test target week (40 -> we have ~40 weeks of history)
    test_len   : number of target weeks in each test window (4 weeks ~ 1 month)
    step       : how far the window slides each time
    Incomplete windows at the end are dropped.
    """
    weeks = pd.DatetimeIndex(weeks)
    rows, k, start = [], 1, first_test
    while start + test_len <= len(weeks):
        rows.append({"fold": k,
                     "test_start": weeks[start], "test_end": weeks[start + test_len - 1],
                     "test_start_pos": start, "test_end_pos": start + test_len - 1})
        k, start = k + 1, start + step
    return pd.DataFrame(rows)


def split_fold(table, fold, horizon, target_col=None):
    """
    Return (train_idx, test_idx) as boolean masks over `table` (one row per category x origin week).

    Rules (no leakage):
      * test rows  : target week inside the test window.
      * train rows : target week at least `horizon` weeks BEFORE the first test target week,
                     i.e. the outcome was already known when the first test forecast is made.
      * rows without a target are ignored.
    """
    target_col = target_col or f"y_h{horizon}"
    target_week = table["week"] + pd.Timedelta(weeks=horizon)
    has_y = table[target_col].notna()
    test = has_y & (target_week >= fold["test_start"]) & (target_week <= fold["test_end"])
    train = has_y & (target_week <= fold["test_start"] - pd.Timedelta(weeks=horizon))
    return train, test


# ----------------------------------------------------------------------------------------------
# 2. Metrics
# ----------------------------------------------------------------------------------------------
def metrics(y_true, y_pred):
    """MAE, RMSE and WAPE (= total absolute error / total demand; a % error that can be compared across categories)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {"MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "WAPE": np.sum(np.abs(err)) / np.sum(y_true),
            "n": len(y_true)}


def score_folds(preds, exclude_target_weeks=None):
    """
    preds: DataFrame with columns category, fold, horizon, model, target_week, y_true, y_pred.
    Returns one row of metrics per (model, horizon, category, fold).
    `exclude_target_weeks` (e.g. the Black Friday week) are removed before scoring.
    """
    p = preds
    if exclude_target_weeks is not None:
        p = p[~p["target_week"].isin(pd.to_datetime(exclude_target_weeks))]
    rows = []
    for (model, h, cat, fold), g in p.groupby(["model", "horizon", "category", "fold"]):
        rows.append({"model": model, "horizon": h, "category": cat, "fold": fold,
                     **metrics(g["y_true"], g["y_pred"])})
    return pd.DataFrame(rows)


def summarise(fold_scores, by=("model", "horizon", "category")):
    """Mean and standard deviation of each metric ACROSS FOLDS (this is what Denes asked for)."""
    g = fold_scores.groupby(list(by))[["MAE", "RMSE", "WAPE"]]
    out = g.mean().join(g.std(), rsuffix="_sd")
    out["n_folds"] = fold_scores.groupby(list(by))["fold"].nunique()
    return out.reset_index()


# ----------------------------------------------------------------------------------------------
# 3. Baselines and calendar helpers
# ----------------------------------------------------------------------------------------------
def baseline_predictions(table, folds, horizon, column, name):
    """
    A baseline needs no training: its forecast is simply the value in `column`
    (for example `orders_l0` = last week's orders, or `orders_roll4` = 4-week average).
    Returns predictions in the standard format used by score_folds().
    """
    out = []
    for _, fold in folds.iterrows():
        _, test = split_fold(table, fold, horizon)
        t = table[test]
        out.append(pd.DataFrame({
            "model": name, "horizon": horizon, "category": t["category"].values, "fold": fold["fold"],
            "target_week": (t["week"] + pd.Timedelta(weeks=horizon)).values,
            "y_true": t[f"y_h{horizon}"].values, "y_pred": t[column].values}))
    return pd.concat(out, ignore_index=True)


def add_target_calendar(table, horizon):
    """Calendar information about the TARGET week (known in advance, so not leakage)."""
    t = table.copy()
    target_week = t["week"] + pd.Timedelta(weeks=horizon)
    t["target_month"] = target_week.dt.month
    t["target_is_black_friday"] = target_week.isin(pd.to_datetime(BLACK_FRIDAY_WEEKS)).astype(int)
    return t


def poisson_floor(mean_orders):
    """
    If orders were pure random counts (Poisson), even a PERFECT model would still miss by about
    0.8 / sqrt(mean) in relative terms. It tells us the best WAPE we can hope for.
    """
    return 0.8 / np.sqrt(mean_orders)


# ----------------------------------------------------------------------------------------------
# 4. Comparing models (notebooks 04 and 05)
# ----------------------------------------------------------------------------------------------
def overall_table(fold_scores):
    """WAPE averaged over categories inside each fold, then mean and std ACROSS folds."""
    per_fold = fold_scores.groupby(["model", "horizon", "fold"])[["MAE", "RMSE", "WAPE"]].mean().reset_index()
    g = per_fold.groupby(["model", "horizon"])[["MAE", "RMSE", "WAPE"]]
    out = g.mean().join(g.std(), rsuffix="_sd")
    return out.reset_index()


def paired_bootstrap(preds, model_a, model_b, horizon, n_boot=2000, seed=0, exclude_target_weeks=None):
    """
    Is model_b really better (or worse) than model_a, or is it luck?

    We look at the pooled WAPE difference (b - a): NEGATIVE means b is better.
    Whole target WEEKS are resampled with replacement (all categories of a week stay together,
    because they share the same weekly shocks). The 2.5% / 97.5% percentiles give a 95% interval.
    """
    p = preds[(preds["horizon"] == horizon) & preds["model"].isin([model_a, model_b])].copy()
    if exclude_target_weeks is not None:
        p = p[~p["target_week"].isin(pd.to_datetime(exclude_target_weeks))]
    p["abs_err"] = (p["y_pred"] - p["y_true"]).abs()
    wide = p.pivot_table(index=["category", "target_week"], columns="model", values="abs_err")
    y = p[p["model"] == model_a].set_index(["category", "target_week"])["y_true"]
    wk = pd.DataFrame({"a": wide[model_a], "b": wide[model_b], "y": y}).groupby("target_week").sum()
    a, b, yy = wk["a"].to_numpy(), wk["b"].to_numpy(), wk["y"].to_numpy()
    observed = (b.sum() - a.sum()) / yy.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wk), size=(n_boot, len(wk)))
    diffs = (b[idx].sum(1) - a[idx].sum(1)) / yy[idx].sum(1)
    return {"model_a": model_a, "model_b": model_b, "horizon": horizon, "n_weeks": len(wk),
            "diff_wape": observed, "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
            "share_b_better": float((diffs < 0).mean())}
