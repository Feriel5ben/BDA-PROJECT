"""
modelling.py - the XGBoost experiment runner and explanation tools (notebooks 04 and 05).

For every rolling-window fold and every feature set (model A, B, C, D) it:
  1. tunes the XGBoost settings using ONLY the training part of that fold (inner rolling validation),
  2. trains the final model on the whole training part,
  3. forecasts the test window.
Nothing from the test window is ever used for training or tuning.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import forecast_utils as fu

CALENDAR = ["target_is_black_friday"]          # known in advance, added to every model
ANCHOR = "orders_roll4"                        # the ratio target is  y / anchor  (can be "orders_l0" = last week)

DEFAULT_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.85,
                      colsample_bytree=0.85, min_child_weight=3, reg_lambda=5.0)

SEARCH_SPACE = dict(n_estimators=[100, 200, 300], max_depth=[2, 3, 4], learning_rate=[0.03, 0.05, 0.1],
                    subsample=[0.7, 0.85, 1.0], colsample_bytree=[0.7, 0.85, 1.0],
                    min_child_weight=[1, 3, 5, 10], reg_lambda=[1.0, 5.0, 10.0])


def sample_configs(n=12, seed=0):
    """Random search: n random combinations of settings (the SAME list is used for every model = fair)."""
    rng = np.random.default_rng(seed)
    return [{k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()} for _ in range(n)]


def _target(df, h, mode, anchor):
    y = df[f"y_h{h}"].to_numpy(float)
    return y / df[anchor].to_numpy(float) if mode == "ratio" else y


def _to_orders(pred, df, mode, anchor):
    out = pred * df[anchor].to_numpy(float) if mode == "ratio" else pred
    return np.clip(out, 0, None)


def fit_predict(train, test, features, h, params, mode="ratio", seeds=(0,), return_models=False, anchor=ANCHOR):
    """Train on `train`, forecast `test` (in ORDERS, whatever the target mode). Several seeds are averaged."""
    ytr = _target(train, h, mode, anchor)
    preds, models = [], []
    for s in seeds:
        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=s, n_jobs=2, **params)
        m.fit(train[features], ytr)
        preds.append(m.predict(test[features]))
        models.append(m)
    p = _to_orders(np.mean(preds, axis=0), test, mode, anchor)
    return (p, models) if return_models else p


def inner_scores(table, fold, h, features, configs, mode, n_inner=3, inner_len=4, exclude_weeks=None, anchor=ANCHOR):
    """
    Score every candidate setting with a small rolling validation INSIDE the training period:
    the last `n_inner` windows of `inner_len` weeks before the real test window.
    """
    last_target = fold["test_start"] - pd.Timedelta(weeks=h)      # latest target week allowed in training
    excl = pd.to_datetime(exclude_weeks) if exclude_weeks is not None else []
    scores, used = np.zeros(len(configs)), 0
    for j in range(n_inner):
        end = last_target - pd.Timedelta(weeks=inner_len * (n_inner - 1 - j))
        start = end - pd.Timedelta(weeks=inner_len - 1)
        tr_m, te_m = fu.split_fold(table, {"test_start": start, "test_end": end}, h)
        tr, te = table[tr_m], table[te_m]
        te = te[~(te["week"] + pd.Timedelta(weeks=h)).isin(excl)]      # the Black Friday week would dominate the score
        if len(tr) < 40 or len(te) == 0:
            continue
        used += 1
        y = te[f"y_h{h}"].to_numpy(float)
        for i, cfg in enumerate(configs):
            p = fit_predict(tr, te, features, h, cfg, mode, anchor=anchor)
            scores[i] += np.abs(p - y).sum() / y.sum()
    return scores / max(used, 1)


def run_experiment(table, folds, h, feature_sets, configs, target_mode="ratio", seeds=(0, 1, 2),
                   n_inner=3, inner_len=4, name_prefix="XGB-", return_models=False, verbose=True, anchor=ANCHOR):
    """Run all folds x all feature sets for one horizon. Returns (predictions, tuning_log, models)."""
    t = fu.add_target_calendar(table, h)
    preds, log, models = [], [], {}
    for _, fold in folds.iterrows():
        tr_m, te_m = fu.split_fold(t, fold, h)
        train, test = t[tr_m], t[te_m]
        for key, feats in feature_sets.items():
            feats = list(feats) + CALENDAR
            if len(configs) > 1:
                sc = inner_scores(t, fold, h, feats, configs, target_mode, n_inner, inner_len, fu.BLACK_FRIDAY_WEEKS, anchor)
                best_i = int(np.argmin(sc)); best_score = float(sc[best_i])
            else:
                best_i, best_score = 0, np.nan
            best = configs[best_i]
            out = fit_predict(train, test, feats, h, best, target_mode, seeds, return_models, anchor)
            p, ms = out if return_models else (out, None)
            if return_models:
                models[(int(fold["fold"]), key)] = (ms, feats, train, test)
            preds.append(pd.DataFrame({
                "model": f"{name_prefix}{key}", "horizon": h, "category": test["category"].values,
                "fold": int(fold["fold"]), "target_week": (test["week"] + pd.Timedelta(weeks=h)).values,
                "y_true": test[f"y_h{h}"].values, "y_pred": p}))
            log.append({"fold": int(fold["fold"]), "model": f"{name_prefix}{key}", "horizon": h,
                        "n_train_rows": len(train), "inner_wape": best_score, **best})
        if verbose:
            print(f"  fold {int(fold['fold'])}/{len(folds)} done (horizon {h}, {len(train)} training rows)", flush=True)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(log), models


# ----------------------------------------------------------------------------------------------
# Explanation tools (notebook 05)
# ----------------------------------------------------------------------------------------------
def predict_orders(models, df, features, h, mode="ratio", anchor=ANCHOR):
    """Average forecast (in orders) of the seed-models for the rows of df."""
    p = np.mean([m.predict(df[features]) for m in models], axis=0)
    return _to_orders(p, df, mode, anchor)


def gain_importance(models_dict, key):
    """XGBoost's built-in 'gain' importance, as a share of the total, averaged over folds and seeds."""
    rows = []
    for (fold, k), (ms, feats, _, _) in models_dict.items():
        if k != key:
            continue
        gains = pd.DataFrame([m.get_booster().get_score(importance_type="gain") for m in ms]).reindex(columns=feats).fillna(0).mean()
        rows.append(gains / gains.sum())
    return pd.concat(rows, axis=1).mean(axis=1).sort_values(ascending=False)


def permutation_importance(models_dict, key, h, groups=None, mode="ratio", anchor=ANCHOR, n_repeats=20, seed=0):
    """
    Permutation importance (PFI) on the TEST windows of every fold.
    We shuffle the values of a feature (or of a whole GROUP of features together) among the test rows and
    measure how much the WAPE gets worse. A big increase = the model really uses that information.
    The Black Friday week is left out of the scoring.
    Returns one row per fold and feature/group.
    """
    rng = np.random.default_rng(seed)
    bf = pd.to_datetime(fu.BLACK_FRIDAY_WEEKS)
    rows = []
    for (fold, k), (ms, feats, _, test) in models_dict.items():
        if k != key:
            continue
        keep = ~(test["week"] + pd.Timedelta(weeks=h)).isin(bf).to_numpy()
        te = test[keep]
        y = te[f"y_h{h}"].to_numpy(float)
        base_err = np.abs(predict_orders(ms, te, feats, h, mode, anchor) - y).sum()
        grp = groups if groups is not None else {f: [f] for f in feats}
        for g, cols in grp.items():
            cols = [c for c in cols if c in feats]
            if not cols:
                continue
            incs = []
            for _ in range(n_repeats):
                perm = te.copy()
                perm[cols] = perm[cols].to_numpy()[rng.permutation(len(perm))]
                err = np.abs(predict_orders(ms, perm, feats, h, mode, anchor) - y).sum()
                incs.append((err - base_err) / y.sum())
            rows.append({"fold": fold, "feature": g, "increase_wape": float(np.mean(incs))})
    return pd.DataFrame(rows)


def shap_values(models_dict, key, h):
    """
    SHAP values from XGBoost's own TreeSHAP (no extra package needed), on the test rows of all folds.
    Units: contribution to the predicted RATIO (ratio target). Returns (shap_df, feature_values_df).
    """
    sh, xs = [], []
    for (fold, k), (ms, feats, _, test) in models_dict.items():
        if k != key:
            continue
        keep = ~(test["week"] + pd.Timedelta(weeks=h)).isin(pd.to_datetime(fu.BLACK_FRIDAY_WEEKS)).to_numpy()
        te = test[keep]
        d = xgb.DMatrix(te[feats])
        c = np.mean([m.get_booster().predict(d, pred_contribs=True) for m in ms], axis=0)
        sh.append(pd.DataFrame(c[:, :-1], columns=feats, index=te.index))
        xs.append(te[feats])
    return pd.concat(sh), pd.concat(xs)
