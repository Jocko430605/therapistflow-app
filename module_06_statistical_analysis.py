"""
Module 6: Statistical Trend Analysis and Hypothesis Testing
=============================================================
Pipeline: COVID-19 Ego-Network Behavioral Activation Study
Version: 1.1
Authors: Ben (Kaiser Permanente) / Dr. Christian Lopez (Lafayette College)

Purpose:
    Formal statistical testing of the core hypothesis: the Dunbar layer at
    which COVID-19 risk concentrates within an ego's social network predicts
    the speed and stage of behavioral activation along the TTM.

    All analyses run per-phase first, then results are compared across phases.

    Part A: Mann-Kendall trend detection (per ego per phase)
    Part B: Theil-Sen activation rate + Kruskal-Wallis group comparison
    Part C: Regression (risk -> activation) controlling for ambient signal
    Part D: Confusion matrix analysis (TTM stage prediction)
    Part E: Cross-phase comparison synthesis table

Inputs:
    {INPUT_DIR}/ego_phase_activation_{INPUT_DATE}.csv
    {INPUT_DIR}/ego_latency_{INPUT_DATE}.csv
    {INPUT_DIR}/ambient_risk_timeseries_{INPUT_DATE}.csv

Outputs:
    {OUTPUT_DIR}/trend_analysis_{RUN_DATE}.csv
    {OUTPUT_DIR}/slope_comparison_{RUN_DATE}.csv
    {OUTPUT_DIR}/regression_results_{RUN_DATE}.csv
    {OUTPUT_DIR}/confusion_matrices_{RUN_DATE}.csv
    {OUTPUT_DIR}/cross_phase_summary_{RUN_DATE}.csv

Usage:
    1. Set INPUT_DIR, INPUT_DATE, and OUTPUT_DIR.
    2. Run: python module_06_statistical_analysis.py
"""

import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

from pipeline_utils import PHASES, DUNBAR_LAYERS

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = "./data/processed"
INPUT_DATE = "2026_03_28"
OUTPUT_DIR = "./data/processed"
RUN_DATE = datetime.now().strftime("%Y_%m_%d")

RISK_THRESHOLD = 0.1
MK_ALPHA = 0.05
MK_IMPLEMENTATION = "scipy"
MIN_WINDOWS_FOR_TREND = 3
KW_ALPHA = 0.05
DUNN_CORRECTION = "bonferroni"
REGRESSION_APPROACH = "ols"
CLASSIFIER_TYPE = "random_forest"
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_module5_outputs(input_dir, input_date):
    """Load all three Module 5 output files."""
    activation_path = os.path.join(
        input_dir, f"ego_phase_activation_{input_date}.csv"
    )
    latency_path = os.path.join(input_dir, f"ego_latency_{input_date}.csv")
    ambient_path = os.path.join(
        input_dir, f"ambient_risk_timeseries_{input_date}.csv"
    )

    print("Loading Module 5 outputs...")
    activation_df = pd.read_csv(activation_path)
    print(f"  Activation table: {len(activation_df):,} rows")

    latency_df = pd.read_csv(latency_path)
    print(f"  Latency table: {len(latency_df):,} rows")

    ambient_df = pd.read_csv(ambient_path)
    print(f"  Ambient timeseries: {len(ambient_df):,} rows")

    print(f"  Unique egos: {activation_df['ego_PersonID'].nunique()}")
    return activation_df, latency_df, ambient_df


# =============================================================================
# PART A: MANN-KENDALL TREND DETECTION
# =============================================================================

def _mk_test_scipy(values):
    """Mann-Kendall via scipy.stats.kendalltau."""
    from scipy.stats import kendalltau
    time_index = np.arange(1, len(values) + 1)
    tau, p_value = kendalltau(time_index, values)
    return tau, p_value


def _mk_test_pymannkendall(values):
    """Mann-Kendall via pymannkendall."""
    import pymannkendall as mk
    result = mk.original_test(values)
    return result.Tau, result.p


def compute_mann_kendall(activation_df):
    """Part A: Mann-Kendall test per ego per phase."""
    print("\n" + "=" * 70)
    print("PART A: Mann-Kendall Trend Detection")
    print("=" * 70)

    mk_func = (
        _mk_test_pymannkendall
        if MK_IMPLEMENTATION == "pymannkendall"
        else _mk_test_scipy
    )
    print(f"  Implementation: {MK_IMPLEMENTATION}")

    results = []

    for phase_key in PHASES:
        phase_data = activation_df[activation_df["phase"] == phase_key]
        for ego_id in phase_data["ego_PersonID"].unique():
            ego_phase = phase_data[
                phase_data["ego_PersonID"] == ego_id
            ].sort_values("window_number")
            scores = ego_phase["continuous_activation_score"].values
            valid_scores = scores[~np.isnan(scores)]

            if len(valid_scores) < MIN_WINDOWS_FOR_TREND:
                results.append({
                    "ego_PersonID": ego_id, "phase": phase_key,
                    "mk_trend": "insufficient_data",
                    "mk_p_value": np.nan, "mk_significant": False,
                    "mk_tau": np.nan,
                })
                continue

            if np.all(valid_scores == valid_scores[0]):
                results.append({
                    "ego_PersonID": ego_id, "phase": phase_key,
                    "mk_trend": "no trend", "mk_p_value": 1.0,
                    "mk_significant": False, "mk_tau": 0.0,
                })
                continue

            try:
                tau, p_value = mk_func(valid_scores)
            except Exception:
                results.append({
                    "ego_PersonID": ego_id, "phase": phase_key,
                    "mk_trend": "error", "mk_p_value": np.nan,
                    "mk_significant": False, "mk_tau": np.nan,
                })
                continue

            if p_value < MK_ALPHA:
                trend = "increasing" if tau > 0 else "decreasing"
            else:
                trend = "no trend"

            results.append({
                "ego_PersonID": ego_id, "phase": phase_key,
                "mk_trend": trend, "mk_p_value": p_value,
                "mk_significant": p_value < MK_ALPHA, "mk_tau": tau,
            })

    trend_df = pd.DataFrame(results)

    for phase_key in PHASES:
        phase_trends = trend_df[trend_df["phase"] == phase_key]
        n_total = len(phase_trends)
        if n_total == 0:
            continue
        n_sig = phase_trends["mk_significant"].sum()
        n_inc = len(phase_trends[
            phase_trends["mk_significant"] & (phase_trends["mk_trend"] == "increasing")
        ])
        n_dec = len(phase_trends[
            phase_trends["mk_significant"] & (phase_trends["mk_trend"] == "decreasing")
        ])
        print(f"\n  {PHASES[phase_key]['label']}:")
        print(f"    Total: {n_total}, Significant: {n_sig} ({n_sig/n_total*100:.1f}%)")
        print(f"    Increasing: {n_inc}, Decreasing: {n_dec}")

    return trend_df


# =============================================================================
# PART B: THEIL-SEN ACTIVATION RATE + GROUP COMPARISON
# =============================================================================

def compute_theil_sen(activation_df, trend_df):
    """Part B: Compute Theil-Sen slope per ego per phase."""
    print("\n" + "=" * 70)
    print("PART B: Theil-Sen Activation Rate")
    print("=" * 70)

    from scipy.stats import theilslopes

    slopes = []
    intercepts = []

    for _, row in trend_df.iterrows():
        ego_id = row["ego_PersonID"]
        phase_key = row["phase"]

        ego_phase = activation_df[
            (activation_df["ego_PersonID"] == ego_id)
            & (activation_df["phase"] == phase_key)
        ].sort_values("window_number")

        scores = ego_phase["continuous_activation_score"].values
        valid_mask = ~np.isnan(scores)
        valid_scores = scores[valid_mask]
        valid_windows = ego_phase["window_number"].values[valid_mask]

        if len(valid_scores) < MIN_WINDOWS_FOR_TREND:
            slopes.append(np.nan)
            intercepts.append(np.nan)
            continue

        if np.all(valid_scores == valid_scores[0]):
            slopes.append(0.0)
            intercepts.append(valid_scores[0])
            continue

        try:
            result = theilslopes(valid_scores, valid_windows)
            slopes.append(result.slope)
            intercepts.append(result.intercept)
        except Exception:
            slopes.append(np.nan)
            intercepts.append(np.nan)

    trend_df = trend_df.copy()
    trend_df["ts_slope"] = slopes
    trend_df["ts_intercept"] = intercepts
    trend_df["ts_slope_significant"] = trend_df["mk_significant"]

    for phase_key in PHASES:
        phase_data = trend_df[trend_df["phase"] == phase_key]
        valid = phase_data["ts_slope"].dropna()
        if len(valid) == 0:
            continue
        print(f"\n  {PHASES[phase_key]['label']}:")
        print(f"    Mean slope: {valid.mean():.4f}, Median: {valid.median():.4f}")

    return trend_df


def assign_activation_groups(trend_df, latency_df):
    """Assign ego-phase to activation group by first-affected Dunbar layer."""
    first_risk_map = {}
    for _, row in latency_df.iterrows():
        ego_id = row["ego_PersonID"]
        phase = row["phase"]
        layer_windows = {}
        for layer in DUNBAR_LAYERS:
            col = f"first_risk_window_{layer}"
            val = row.get(col, np.nan)
            if pd.notna(val) and val > 0:
                layer_windows[layer] = val
        if layer_windows:
            first_risk_map[(ego_id, phase)] = min(
                layer_windows, key=layer_windows.get
            )
        else:
            first_risk_map[(ego_id, phase)] = "none"

    trend_df = trend_df.copy()
    trend_df["first_risk_layer"] = trend_df.apply(
        lambda r: first_risk_map.get(
            (r["ego_PersonID"], r["phase"]), "none"
        ),
        axis=1,
    )

    group_map = {
        "inner_5": 1, "middle_15": 2, "outer_50": 3,
        "outer_150": 4, "none": 5,
    }
    trend_df["activation_group"] = trend_df["first_risk_layer"].map(group_map)
    return trend_df


def compute_slope_comparison(trend_df):
    """Compare Theil-Sen slopes across groups per phase."""
    print("\n  --- Slope comparison by activation group ---")
    from scipy.stats import kruskal

    results = []
    group_labels = {
        1: "inner_5", 2: "middle_15", 3: "outer_50",
        4: "outer_150", 5: "ambient_only",
    }

    for phase_key in PHASES:
        phase_data = trend_df[trend_df["phase"] == phase_key].copy()
        phase_data = phase_data[phase_data["ts_slope"].notna()]

        groups_with_data = []
        group_slopes = {}
        for grp in sorted(phase_data["activation_group"].unique()):
            grp_slopes = phase_data[
                phase_data["activation_group"] == grp
            ]["ts_slope"].values
            if len(grp_slopes) >= 2:
                groups_with_data.append(grp)
                group_slopes[grp] = grp_slopes

        kw_stat = np.nan
        kw_p = np.nan
        if len(groups_with_data) >= 2:
            try:
                kw_arrays = [group_slopes[g] for g in groups_with_data]
                kw_stat, kw_p = kruskal(*kw_arrays)
            except Exception:
                pass

        if pd.notna(kw_stat):
            sig = (
                "***" if kw_p < 0.001 else "**" if kw_p < 0.01
                else "*" if kw_p < 0.05 else "ns"
            )
            print(f"\n  {PHASES[phase_key]['label']}:")
            print(f"    Kruskal-Wallis H={kw_stat:.3f}, p={kw_p:.4f} {sig}")

        for grp in range(1, 6):
            grp_data = phase_data[phase_data["activation_group"] == grp]
            results.append({
                "phase": phase_key,
                "activation_group": grp,
                "activation_group_label": group_labels[grp],
                "n_egos": len(grp_data),
                "mean_slope": grp_data["ts_slope"].mean() if len(grp_data) > 0 else np.nan,
                "median_slope": grp_data["ts_slope"].median() if len(grp_data) > 0 else np.nan,
                "std_slope": grp_data["ts_slope"].std() if len(grp_data) > 0 else np.nan,
                "kruskal_wallis_statistic": kw_stat,
                "kruskal_wallis_p_value": kw_p,
            })

        if pd.notna(kw_p) and kw_p < KW_ALPHA and len(groups_with_data) >= 2:
            _run_dunn_posthoc(phase_data, groups_with_data, phase_key)

    return pd.DataFrame(results)


def _run_dunn_posthoc(phase_data, groups_with_data, phase_key):
    """Run Dunn's post-hoc test."""
    try:
        import scikit_posthocs as sp
        dunn_result = sp.posthoc_dunn(
            phase_data, val_col="ts_slope", group_col="activation_group",
            p_adjust=DUNN_CORRECTION,
        )
        print(f"    Dunn's post-hoc ({DUNN_CORRECTION} correction):")
        for i in range(len(dunn_result)):
            for j in range(i + 1, len(dunn_result)):
                grp_i = dunn_result.index[i]
                grp_j = dunn_result.columns[j]
                p = dunn_result.iloc[i, j]
                sig = (
                    "***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.05 else "ns"
                )
                print(f"      Group {grp_i} vs {grp_j}: p={p:.4f} {sig}")
    except ImportError:
        _run_dunn_manual(phase_data, groups_with_data, phase_key)
    except Exception as e:
        print(f"    Dunn's post-hoc failed: {e}")


def _run_dunn_manual(phase_data, groups_with_data, phase_key):
    """Manual Dunn's test when scikit_posthocs unavailable."""
    from scipy.stats import norm, rankdata

    all_slopes = phase_data["ts_slope"].values
    all_groups = phase_data["activation_group"].values
    n_total = len(all_slopes)
    ranks = rankdata(all_slopes)

    group_mean_ranks = {}
    group_ns = {}
    for g in groups_with_data:
        mask = all_groups == g
        group_mean_ranks[g] = ranks[mask].mean()
        group_ns[g] = mask.sum()

    n_comparisons = len(groups_with_data) * (len(groups_with_data) - 1) // 2

    unique_ranks, counts = np.unique(ranks, return_counts=True)
    tie_correction = np.sum(counts ** 3 - counts) / (12 * (n_total - 1))
    # Guard against negative variance from heavy ties
    base_var = n_total * (n_total + 1) / 12 - tie_correction
    sigma = np.sqrt(max(0.0, base_var))

    if sigma == 0:
        print(f"    Dunn's post-hoc: zero variance (all identical slopes)")
        return

    print(f"    Dunn's post-hoc (manual, {DUNN_CORRECTION} correction):")
    for i, g_i in enumerate(groups_with_data):
        for g_j in groups_with_data[i + 1:]:
            diff = abs(group_mean_ranks[g_i] - group_mean_ranks[g_j])
            se = sigma * np.sqrt(1.0 / group_ns[g_i] + 1.0 / group_ns[g_j])
            if se == 0:
                continue
            z = diff / se
            p = 2 * (1 - norm.cdf(abs(z)))
            if DUNN_CORRECTION == "bonferroni" and n_comparisons > 0:
                p = min(p * n_comparisons, 1.0)
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "ns"
            )
            print(f"      Group {g_i} vs {g_j}: z={z:.3f}, p={p:.4f} {sig}")


# =============================================================================
# PART C: REGRESSION (RISK -> ACTIVATION)
# =============================================================================

def compute_regression(activation_df, ambient_df):
    """
    Part C: OLS per phase with ego fixed effects (demeaned), plus pooled.

    NOTE on the pooled model: demeaning removes within-ego means across all
    phases, so the phase dummies capture whether a phase's deviation from
    the ego-mean is systematically different -- not raw phase-level
    differences. Interpret accordingly.
    """
    print("\n" + "=" * 70)
    print("PART C: Regression Analysis (Risk -> Activation)")
    print("=" * 70)

    merged = activation_df.merge(
        ambient_df[["phase", "window_start", "ambient_emotion_risk"]].rename(
            columns={"ambient_emotion_risk": "ambient_risk_global"}
        ),
        on=["phase", "window_start"],
        how="left",
    )
    if "ambient_risk" not in merged.columns:
        merged["ambient_risk"] = merged["ambient_risk_global"].fillna(0)
    ambient_col = "ambient_risk"

    results = []
    for phase_key in PHASES:
        phase_data = merged[merged["phase"] == phase_key].copy()
        results.append(_fit_ols_phase(phase_data, phase_key, ambient_col))

    results.append(_fit_ols_pooled(merged, ambient_col))

    return pd.DataFrame(results)


def _fit_ols_phase(phase_data, phase_key, ambient_col):
    """Fit OLS for a single phase with ego fixed effects."""
    y_col = "continuous_activation_score"
    feature_cols = [
        "inner_5_risk", "middle_15_risk", "outer_50_risk",
        "outer_150_risk", ambient_col,
    ]
    output_names = ["inner_5", "middle_15", "outer_50", "outer_150", "ambient"]

    result = {"phase": phase_key, "n_observations": len(phase_data)}
    for name in output_names:
        result[f"coef_{name}"] = np.nan
        result[f"pvalue_{name}"] = np.nan
    result["r_squared"] = np.nan

    cols_needed = [y_col] + feature_cols + ["ego_PersonID"]
    clean = phase_data[cols_needed].dropna()
    if len(clean) < len(feature_cols) + 2:
        print(f"\n  {PHASES[phase_key]['label']}: insufficient data ({len(clean)} obs)")
        return result

    result["n_observations"] = len(clean)

    try:
        import statsmodels.api as sm

        ego_means = clean.groupby("ego_PersonID")[
            [y_col] + feature_cols
        ].transform("mean")
        demeaned = clean[[y_col] + feature_cols] - ego_means

        if demeaned[y_col].std() == 0:
            print(f"\n  {PHASES[phase_key]['label']}: no within-ego variation")
            return result

        X = sm.add_constant(demeaned[feature_cols])
        y = demeaned[y_col]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X).fit()

        result["r_squared"] = model.rsquared

        print(f"\n  {PHASES[phase_key]['label']} (n={len(clean):,}, "
              f"R2={model.rsquared:.4f}):")
        for i, name in enumerate(output_names):
            coef_key = feature_cols[i]
            if coef_key in model.params:
                coef = model.params[coef_key]
                pval = model.pvalues[coef_key]
                result[f"coef_{name}"] = coef
                result[f"pvalue_{name}"] = pval
                sig = (
                    "***" if pval < 0.001 else "**" if pval < 0.01
                    else "*" if pval < 0.05 else ""
                )
                print(f"    {name:15s}: B={coef:+.4f}  p={pval:.4f} {sig}")

    except ImportError:
        print(f"\n  {PHASES[phase_key]['label']}: statsmodels not available")
    except Exception as e:
        print(f"\n  {PHASES[phase_key]['label']}: regression failed -- {e}")

    return result


def _fit_ols_pooled(activation_data, ambient_col):
    """
    Fit pooled OLS across all phases with phase indicators.
    NOTE: Phase dummies after demeaning capture phase-specific deviations
    from the ego mean, not raw phase differences.
    """
    y_col = "continuous_activation_score"
    feature_cols = [
        "inner_5_risk", "middle_15_risk", "outer_50_risk",
        "outer_150_risk", ambient_col,
    ]
    output_names = ["inner_5", "middle_15", "outer_50", "outer_150", "ambient"]

    result = {"phase": "pooled_all_phases", "n_observations": len(activation_data)}
    for name in output_names:
        result[f"coef_{name}"] = np.nan
        result[f"pvalue_{name}"] = np.nan
    result["r_squared"] = np.nan

    cols_needed = [y_col] + feature_cols + ["ego_PersonID", "phase"]
    clean = activation_data[cols_needed].dropna()
    if len(clean) < len(feature_cols) + 2:
        print(f"\n  Pooled model: insufficient data ({len(clean)} obs)")
        return result

    result["n_observations"] = len(clean)

    try:
        import statsmodels.api as sm

        phase_dummies = pd.get_dummies(
            clean["phase"], prefix="phase", drop_first=True
        )

        ego_means = clean.groupby("ego_PersonID")[
            [y_col] + feature_cols
        ].transform("mean")
        demeaned = clean[[y_col] + feature_cols] - ego_means

        if demeaned[y_col].std() == 0:
            print(f"\n  Pooled model: no within-ego variation")
            return result

        X = pd.concat([demeaned[feature_cols], phase_dummies], axis=1)
        X = sm.add_constant(X)
        y = demeaned[y_col]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X).fit()

        result["r_squared"] = model.rsquared

        print(f"\n  Pooled model (n={len(clean):,}, R2={model.rsquared:.4f}):")
        for i, name in enumerate(output_names):
            coef_key = feature_cols[i]
            if coef_key in model.params:
                coef = model.params[coef_key]
                pval = model.pvalues[coef_key]
                result[f"coef_{name}"] = coef
                result[f"pvalue_{name}"] = pval
                sig = (
                    "***" if pval < 0.001 else "**" if pval < 0.01
                    else "*" if pval < 0.05 else ""
                )
                print(f"    {name:15s}: B={coef:+.4f}  p={pval:.4f} {sig}")

    except ImportError:
        print(f"\n  Pooled model: statsmodels not available")
    except Exception as e:
        print(f"\n  Pooled model: regression failed -- {e}")

    return result


# =============================================================================
# PART D: CONFUSION MATRIX ANALYSIS
# =============================================================================

def compute_confusion_matrices(activation_df, ambient_df):
    """
    Part D: Random Forest classifier predicting TTM stage.
    Leave-one-phase-out CV. Compares full model vs ambient-only.

    NOTE: The autoregressive prior_ttm feature may dominate risk features
    if TTM stages are sticky. Consider reporting feature importances and
    running without prior_ttm as an additional comparison.
    """
    print("\n" + "=" * 70)
    print("PART D: Confusion Matrix Analysis")
    print("=" * 70)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support, f1_score,
        )
    except ImportError:
        print("  scikit-learn not available, skipping")
        return _empty_confusion_df()

    merged = activation_df.merge(
        ambient_df[["phase", "window_start", "ambient_emotion_risk"]].rename(
            columns={"ambient_emotion_risk": "ambient_risk_global"}
        ),
        on=["phase", "window_start"],
        how="left",
    )
    if "ambient_risk" not in merged.columns:
        merged["ambient_risk"] = merged["ambient_risk_global"].fillna(0)

    risk_features = [
        "inner_5_risk", "middle_15_risk", "outer_50_risk",
        "outer_150_risk", "ambient_risk",
    ]
    ambient_only_features = ["ambient_risk"]

    valid_stages = [
        "precontemplation", "contemplation", "preparation", "action"
    ]
    merged = merged[merged["assigned_ttm_stage"].isin(valid_stages)].copy()

    if len(merged) == 0:
        print("  No valid TTM stage assignments found")
        return _empty_confusion_df()

    merged = merged.sort_values(["ego_PersonID", "phase", "window_number"])
    stage_to_num = {
        "precontemplation": 0, "contemplation": 1,
        "preparation": 2, "action": 3,
    }
    merged["ttm_numeric"] = merged["assigned_ttm_stage"].map(stage_to_num)
    merged["prior_ttm"] = merged.groupby(["ego_PersonID", "phase"])[
        "ttm_numeric"
    ].shift(1).fillna(0)

    phase_to_num = {k: i for i, k in enumerate(PHASES.keys())}
    merged["phase_numeric"] = merged["phase"].map(phase_to_num)

    full_features = risk_features + ["prior_ttm", "phase_numeric"]
    model_b_features = ambient_only_features + ["prior_ttm", "phase_numeric"]

    results = []
    for holdout_phase in PHASES:
        train_data = merged[merged["phase"] != holdout_phase]
        test_data = merged[merged["phase"] == holdout_phase]

        if len(train_data) < 10 or len(test_data) < 5:
            results.append(_phase_confusion_result(holdout_phase, len(test_data)))
            continue

        X_train = train_data[full_features].fillna(0)
        y_train = train_data["assigned_ttm_stage"]
        X_test = test_data[full_features].fillna(0)
        y_test = test_data["assigned_ttm_stage"]

        if len(y_train.unique()) < 2 or len(y_test.unique()) < 1:
            results.append(_phase_confusion_result(holdout_phase, len(X_test)))
            continue

        # Model A: full model
        clf_a = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            random_state=RF_RANDOM_STATE, class_weight="balanced",
        )
        try:
            clf_a.fit(X_train, y_train)
            y_pred_a = clf_a.predict(X_test)
            acc_a = accuracy_score(y_test, y_pred_a)
            macro_f1_a = f1_score(
                y_test, y_pred_a, average="macro", zero_division=0
            )
            weighted_f1_a = f1_score(
                y_test, y_pred_a, average="weighted", zero_division=0
            )
            precision_a, recall_a, f1_a, _ = precision_recall_fscore_support(
                y_test, y_pred_a, labels=valid_stages, zero_division=0,
            )
        except Exception as e:
            print(f"  {holdout_phase}: Model A failed -- {e}")
            results.append(_phase_confusion_result(holdout_phase, len(X_test)))
            continue

        # Model B: ambient-only
        X_train_b = train_data[model_b_features].fillna(0)
        X_test_b = test_data[model_b_features].fillna(0)
        clf_b = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            random_state=RF_RANDOM_STATE, class_weight="balanced",
        )
        try:
            clf_b.fit(X_train_b, y_train)
            y_pred_b = clf_b.predict(X_test_b)
            acc_b = accuracy_score(y_test, y_pred_b)
        except Exception:
            acc_b = np.nan

        row = {
            "phase": holdout_phase,
            "n_test_observations": len(X_test),
            "accuracy": acc_a,
            "macro_f1": macro_f1_a,
            "weighted_f1": weighted_f1_a,
        }
        for i, stage in enumerate(valid_stages):
            row[f"precision_{stage}"] = precision_a[i]
            row[f"recall_{stage}"] = recall_a[i]
            row[f"f1_{stage}"] = f1_a[i]

        row["ambient_only_accuracy"] = acc_b
        row["proximity_value_added"] = (
            acc_a - acc_b if pd.notna(acc_b) else np.nan
        )
        results.append(row)

        print(f"\n  {PHASES[holdout_phase]['label']} (held out, n={len(X_test)}):")
        print(f"    Model A accuracy: {acc_a:.3f}  macro-F1: {macro_f1_a:.3f}")
        if pd.notna(acc_b):
            print(f"    Model B accuracy: {acc_b:.3f}  (ambient-only)")
            print(f"    Proximity value:  {acc_a - acc_b:+.3f}")

    return pd.DataFrame(results)


def _phase_confusion_result(phase_key, n_obs):
    """Empty result row."""
    valid_stages = [
        "precontemplation", "contemplation", "preparation", "action"
    ]
    row = {
        "phase": phase_key, "n_test_observations": n_obs,
        "accuracy": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan,
    }
    for stage in valid_stages:
        row[f"precision_{stage}"] = np.nan
        row[f"recall_{stage}"] = np.nan
        row[f"f1_{stage}"] = np.nan
    row["ambient_only_accuracy"] = np.nan
    row["proximity_value_added"] = np.nan
    return row


def _empty_confusion_df():
    """Empty confusion matrix DataFrame."""
    valid_stages = [
        "precontemplation", "contemplation", "preparation", "action"
    ]
    cols = [
        "phase", "n_test_observations", "accuracy", "macro_f1", "weighted_f1"
    ]
    for stage in valid_stages:
        cols.extend([
            f"precision_{stage}", f"recall_{stage}", f"f1_{stage}"
        ])
    cols.extend(["ambient_only_accuracy", "proximity_value_added"])
    return pd.DataFrame(columns=cols)


# =============================================================================
# PART E: CROSS-PHASE COMPARISON SUMMARY
# =============================================================================

def compute_cross_phase_summary(trend_df, slope_comp_df, regression_df,
                                 confusion_df):
    """Part E: Synthesis table comparing metrics across all five phases."""
    print("\n" + "=" * 70)
    print("PART E: Cross-Phase Comparison Summary")
    print("=" * 70)

    results = []
    for phase_key in PHASES:
        row = {"phase": phase_key, "phase_label": PHASES[phase_key]["label"]}

        phase_trends = trend_df[trend_df["phase"] == phase_key]
        n_total = len(phase_trends)
        if n_total > 0:
            n_sig_inc = len(phase_trends[
                phase_trends["mk_significant"]
                & (phase_trends["mk_trend"] == "increasing")
            ])
            n_sig_dec = len(phase_trends[
                phase_trends["mk_significant"]
                & (phase_trends["mk_trend"] == "decreasing")
            ])
            row["pct_significant_increasing"] = n_sig_inc / n_total * 100
            row["pct_significant_decreasing"] = n_sig_dec / n_total * 100
        else:
            row["pct_significant_increasing"] = np.nan
            row["pct_significant_decreasing"] = np.nan

        valid_slopes = phase_trends["ts_slope"].dropna()
        row["mean_ts_slope"] = (
            valid_slopes.mean() if len(valid_slopes) > 0 else np.nan
        )
        row["median_ts_slope"] = (
            valid_slopes.median() if len(valid_slopes) > 0 else np.nan
        )

        for grp, label in [
            (1, "inner_5"), (2, "middle_15"), (3, "outer_50"),
            (4, "outer_150"), (5, "ambient"),
        ]:
            grp_slopes = phase_trends[
                phase_trends["activation_group"] == grp
            ]["ts_slope"].dropna()
            row[f"mean_slope_{label}"] = (
                grp_slopes.mean() if len(grp_slopes) > 0 else np.nan
            )

        if len(regression_df) > 0:
            phase_reg = regression_df[regression_df["phase"] == phase_key]
            if len(phase_reg) > 0:
                reg_row = phase_reg.iloc[0]
                for layer in [
                    "inner_5", "middle_15", "outer_50", "outer_150", "ambient"
                ]:
                    row[f"reg_coef_{layer}"] = reg_row.get(
                        f"coef_{layer}", np.nan
                    )
                row["reg_r_squared"] = reg_row.get("r_squared", np.nan)

        if len(confusion_df) > 0:
            phase_cm = confusion_df[confusion_df["phase"] == phase_key]
            if len(phase_cm) > 0:
                cm_row = phase_cm.iloc[0]
                row["classifier_accuracy"] = cm_row.get("accuracy", np.nan)
                row["ambient_only_accuracy"] = cm_row.get(
                    "ambient_only_accuracy", np.nan
                )
                row["proximity_value_added"] = cm_row.get(
                    "proximity_value_added", np.nan
                )

        results.append(row)

    summary_df = pd.DataFrame(results)

    print(f"\n  {'Phase':<25s} {'%Inc':>6s} {'%Dec':>6s} {'Slope':>8s} "
          f"{'R2':>6s} {'Acc':>6s} {'Prox+':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
    for _, row in summary_df.iterrows():
        def _fmt(val, fmt_str):
            return fmt_str.format(val) if pd.notna(val) else "---"

        label = row["phase_label"][:25]
        print(f"  {label:<25s} "
              f"{_fmt(row.get('pct_significant_increasing'), '{:.1f}'):>6s} "
              f"{_fmt(row.get('pct_significant_decreasing'), '{:.1f}'):>6s} "
              f"{_fmt(row.get('mean_ts_slope'), '{:.4f}'):>8s} "
              f"{_fmt(row.get('reg_r_squared'), '{:.3f}'):>6s} "
              f"{_fmt(row.get('classifier_accuracy'), '{:.3f}'):>6s} "
              f"{_fmt(row.get('proximity_value_added'), '{:+.3f}'):>6s}")

    return summary_df


# =============================================================================
# CONSOLE REPORT
# =============================================================================

def print_summary_report(trend_df, slope_comp_df, regression_df,
                          confusion_df, summary_df):
    """Print narrative summary."""
    print("\n" + "=" * 70)
    print("MODULE 6: STATISTICAL ANALYSIS -- SUMMARY REPORT")
    print("=" * 70)

    n_egos = trend_df["ego_PersonID"].nunique()
    print(f"\n  Egos analyzed: {n_egos}")
    print(f"  Phases: {trend_df['phase'].nunique()}")

    print(f"\n--- Core Hypothesis Test ---")
    print(f"  H: Inner-layer risk produces steeper activation slopes "
          f"than outer-layer risk")

    hypothesis_supported = 0
    hypothesis_tested = 0

    for phase_key in PHASES:
        phase_trends = trend_df[trend_df["phase"] == phase_key]
        grp1 = phase_trends[
            phase_trends["activation_group"] == 1
        ]["ts_slope"].dropna()
        grp4 = phase_trends[
            phase_trends["activation_group"] == 4
        ]["ts_slope"].dropna()

        if len(grp1) >= 2 and len(grp4) >= 2:
            hypothesis_tested += 1
            if grp1.mean() > grp4.mean():
                hypothesis_supported += 1
                print(f"  {PHASES[phase_key]['label']}: SUPPORTED "
                      f"(inner={grp1.mean():.4f} > outer={grp4.mean():.4f})")
            else:
                print(f"  {PHASES[phase_key]['label']}: NOT SUPPORTED "
                      f"(inner={grp1.mean():.4f} <= outer={grp4.mean():.4f})")
        else:
            print(f"  {PHASES[phase_key]['label']}: CANNOT TEST "
                  f"(insufficient data)")

    if hypothesis_tested > 0:
        print(f"\n  Supported in {hypothesis_supported}/{hypothesis_tested} "
              f"testable phases")
    else:
        print(f"\n  No phases had sufficient data to test")

    if len(summary_df) > 0 and "proximity_value_added" in summary_df.columns:
        valid_prox = summary_df[summary_df["proximity_value_added"].notna()]
        if len(valid_prox) > 0:
            best = valid_prox.loc[valid_prox["proximity_value_added"].idxmax()]
            print(f"\n  Strongest proximity effect: {best['phase_label']} "
                  f"({best['proximity_value_added']:+.3f})")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    activation_df, latency_df, ambient_df = load_module5_outputs(
        INPUT_DIR, INPUT_DATE
    )

    trend_df = compute_mann_kendall(activation_df)
    trend_df = compute_theil_sen(activation_df, trend_df)
    trend_df = assign_activation_groups(trend_df, latency_df)
    slope_comp_df = compute_slope_comparison(trend_df)
    regression_df = compute_regression(activation_df, ambient_df)
    confusion_df = compute_confusion_matrices(activation_df, ambient_df)
    summary_df = compute_cross_phase_summary(
        trend_df, slope_comp_df, regression_df, confusion_df
    )

    print_summary_report(
        trend_df, slope_comp_df, regression_df, confusion_df, summary_df
    )

    outputs = {
        f"trend_analysis_{RUN_DATE}.csv": trend_df,
        f"slope_comparison_{RUN_DATE}.csv": slope_comp_df,
        f"regression_results_{RUN_DATE}.csv": regression_df,
        f"confusion_matrices_{RUN_DATE}.csv": confusion_df,
        f"cross_phase_summary_{RUN_DATE}.csv": summary_df,
    }

    print(f"\nOutputs saved:")
    for filename, df in outputs.items():
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"  {path} ({len(df):,} rows)")

    print(f"\nPipeline complete.")
    return trend_df, slope_comp_df, regression_df, confusion_df, summary_df


if __name__ == "__main__":
    main()
