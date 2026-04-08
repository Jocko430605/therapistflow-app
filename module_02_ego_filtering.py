"""
Module 2: Ego Filtering
========================
Pipeline: COVID-19 Ego-Network Behavioral Activation Study
Version: 1.1
Authors: Ben (Kaiser Permanente) / Dr. Christian Lopez (Lafayette College)

Purpose:
    Reads the ego_candidates CSV from Module 1 and applies a series of filters
    to identify viable egos for network construction. Filters target temporal
    persistence, interaction volume, bot/corporate accounts, and low utilizers.

Inputs:
    {INPUT_DIR}/ego_candidates_{INPUT_DATE}.csv
        Output from Module 1 (one row per unique PersonID with summary stats).

Outputs:
    {OUTPUT_DIR}/ego_list_{RUN_DATE}.csv
        Filtered list of viable egos for Module 3 (network construction).

    {OUTPUT_DIR}/filtering_report_{RUN_DATE}.csv
        Step-by-step filtering report showing how many candidates were
        removed at each stage.

Usage:
    1. Set INPUT_DIR and INPUT_DATE to locate the Module 1 output.
    2. Set OUTPUT_DIR to where output files should be written.
    3. Adjust filtering thresholds as needed.
    4. Run: python module_02_ego_filtering.py
    5. Or open in Jupyter and run cell by cell.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# CONFIGURATION — Update these paths for your environment
# =============================================================================

# Directory containing Module 1 output
INPUT_DIR = "./data/processed"

# Date stamp of the Module 1 output file to read
# Update this to match the ego_candidates file you want to filter
INPUT_DATE = "2026_03_28"

# Directory for output files
OUTPUT_DIR = "./data/processed"

# Run date stamp for output file naming
RUN_DATE = datetime.now().strftime("%Y_%m_%d")

# =============================================================================
# FILTERING THRESHOLDS — Adjust these without modifying logic below
# =============================================================================

# Temporal persistence: minimum number of phases (out of 5) a person must
# be active in to be considered a viable ego
MIN_PHASES_ACTIVE = 4

# Interaction volume: minimum number of unique reply targets across the
# full study period. Ensures the ego has enough alters to construct a
# meaningful network with Dunbar layer differentiation.
MIN_UNIQUE_REPLY_TARGETS = 15

# Bot/corporate detection: accounts with follower/friend ratio above this
# threshold are flagged as likely corporate, celebrity, or broadcast accounts
# (not organic social network participants)
MAX_FOLLOWER_FRIEND_RATIO = 50.0

# Bot detection: accounts with retweet ratio above this threshold are flagged
# as likely automated/bot accounts
MAX_RETWEET_RATIO = 0.95

# Low utilizer: minimum total tweets across the study period
MIN_TOTAL_TWEETS = 20


# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

def load_ego_candidates(input_dir, input_date):
    """
    Load the ego_candidates CSV from Module 1.
    Returns DataFrame with one row per PersonID.
    """
    filepath = os.path.join(input_dir, f"ego_candidates_{input_date}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Ego candidates file not found: {filepath}\n"
            f"Check INPUT_DIR and INPUT_DATE settings. "
            f"Run Module 1 first if needed."
        )

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} ego candidates from {filepath}")
    return df


def apply_filters(df):
    """
    Apply sequential filters to ego candidates. Each filter is applied to the
    full candidate pool independently, then combined. This lets us report how
    many candidates each filter would remove on its own before applying the
    intersection.

    Returns:
        filtered_df: DataFrame containing only viable egos
        report: list of dicts recording each filtering step
    """
    report = []
    n_start = len(df)

    report.append({
        "step": 0,
        "filter": "Starting candidates (Module 1 output)",
        "threshold": "—",
        "candidates_remaining": n_start,
        "removed_this_step": 0,
        "pct_of_original": 100.0,
    })

    # --- Step 1: Minimum total tweets (low utilizer filter) ---
    mask_tweets = df["total_tweets"] >= MIN_TOTAL_TWEETS
    df_filtered = df[mask_tweets].copy()
    removed = n_start - len(df_filtered)
    report.append({
        "step": 1,
        "filter": "Minimum total tweets",
        "threshold": f">= {MIN_TOTAL_TWEETS}",
        "candidates_remaining": len(df_filtered),
        "removed_this_step": removed,
        "pct_of_original": len(df_filtered) / n_start * 100,
    })
    print(f"  Step 1 — Min tweets (>= {MIN_TOTAL_TWEETS}): "
          f"{len(df_filtered):,} remain ({removed:,} removed)")

    # --- Step 2: Temporal persistence (phase activity) ---
    n_before = len(df_filtered)
    mask_phases = df_filtered["phases_active_count"] >= MIN_PHASES_ACTIVE
    df_filtered = df_filtered[mask_phases].copy()
    removed = n_before - len(df_filtered)
    report.append({
        "step": 2,
        "filter": "Temporal persistence (phases active)",
        "threshold": f">= {MIN_PHASES_ACTIVE} of 5 phases",
        "candidates_remaining": len(df_filtered),
        "removed_this_step": removed,
        "pct_of_original": len(df_filtered) / n_start * 100,
    })
    print(f"  Step 2 — Phases active (>= {MIN_PHASES_ACTIVE}): "
          f"{len(df_filtered):,} remain ({removed:,} removed)")

    # --- Step 3: Interaction volume (unique reply targets) ---
    n_before = len(df_filtered)
    mask_interactions = df_filtered["unique_reply_targets"] >= MIN_UNIQUE_REPLY_TARGETS
    df_filtered = df_filtered[mask_interactions].copy()
    removed = n_before - len(df_filtered)
    report.append({
        "step": 3,
        "filter": "Interaction volume (unique reply targets)",
        "threshold": f">= {MIN_UNIQUE_REPLY_TARGETS}",
        "candidates_remaining": len(df_filtered),
        "removed_this_step": removed,
        "pct_of_original": len(df_filtered) / n_start * 100,
    })
    print(f"  Step 3 — Reply targets (>= {MIN_UNIQUE_REPLY_TARGETS}): "
          f"{len(df_filtered):,} remain ({removed:,} removed)")

    # --- Step 4: Bot/corporate exclusion (follower-friend ratio) ---
    n_before = len(df_filtered)
    # NaN ratio means 0 friends — exclude these too (likely inactive or anomalous)
    mask_ffr = (
        df_filtered["follower_friend_ratio"].notna()
        & (df_filtered["follower_friend_ratio"] <= MAX_FOLLOWER_FRIEND_RATIO)
    )
    df_filtered = df_filtered[mask_ffr].copy()
    removed = n_before - len(df_filtered)
    report.append({
        "step": 4,
        "filter": "Bot/corporate filter (follower-friend ratio)",
        "threshold": f"<= {MAX_FOLLOWER_FRIEND_RATIO}",
        "candidates_remaining": len(df_filtered),
        "removed_this_step": removed,
        "pct_of_original": len(df_filtered) / n_start * 100,
    })
    print(f"  Step 4 — Follower/friend ratio (<= {MAX_FOLLOWER_FRIEND_RATIO}): "
          f"{len(df_filtered):,} remain ({removed:,} removed)")

    # --- Step 5: Bot exclusion (retweet ratio) ---
    n_before = len(df_filtered)
    mask_rt = (
        df_filtered["retweet_ratio"].notna()
        & (df_filtered["retweet_ratio"] <= MAX_RETWEET_RATIO)
    )
    df_filtered = df_filtered[mask_rt].copy()
    removed = n_before - len(df_filtered)
    report.append({
        "step": 5,
        "filter": "Bot filter (retweet ratio)",
        "threshold": f"<= {MAX_RETWEET_RATIO}",
        "candidates_remaining": len(df_filtered),
        "removed_this_step": removed,
        "pct_of_original": len(df_filtered) / n_start * 100,
    })
    print(f"  Step 5 — Retweet ratio (<= {MAX_RETWEET_RATIO}): "
          f"{len(df_filtered):,} remain ({removed:,} removed)")

    # --- Final summary row ---
    report.append({
        "step": 99,
        "filter": "FINAL — Viable egos",
        "threshold": "All filters applied",
        "candidates_remaining": len(df_filtered),
        "removed_this_step": n_start - len(df_filtered),
        "pct_of_original": len(df_filtered) / n_start * 100,
    })

    return df_filtered, report


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary_report(df_candidates, df_filtered, report):
    """Print a human-readable summary of the filtering process."""
    n_start = len(df_candidates)
    n_final = len(df_filtered)

    print("\n" + "=" * 70)
    print("MODULE 2: EGO FILTERING — SUMMARY REPORT")
    print("=" * 70)

    print(f"\nInput: {n_start:,} ego candidates from Module 1")
    print(f"Output: {n_final:,} viable egos ({n_final / n_start * 100:.1f}% pass rate)"
          if n_start > 0 else "Output: 0 viable egos")

    print(f"\n--- Filter cascade ---")
    for row in report:
        if row["step"] == 99:
            print(f"  {'─' * 55}")
        print(f"  Step {row['step']:>2}: {row['filter']:45s} → "
              f"{row['candidates_remaining']:>6,} "
              f"({row['pct_of_original']:5.1f}%)")

    if n_final > 0:
        print(f"\n--- Viable ego characteristics ---")
        print(f"  Total tweets: median={df_filtered['total_tweets'].median():.0f}, "
              f"mean={df_filtered['total_tweets'].mean():.0f}, "
              f"range=[{df_filtered['total_tweets'].min()}, {df_filtered['total_tweets'].max():,}]")
        print(f"  Reply targets: median={df_filtered['unique_reply_targets'].median():.0f}, "
              f"mean={df_filtered['unique_reply_targets'].mean():.0f}, "
              f"range=[{df_filtered['unique_reply_targets'].min()}, {df_filtered['unique_reply_targets'].max():,}]")
        print(f"  Phases active: median={df_filtered['phases_active_count'].median():.0f}, "
              f"mean={df_filtered['phases_active_count'].mean():.1f}")
        print(f"  Followers: median={df_filtered['followers_latest'].median():.0f}, "
              f"mean={df_filtered['followers_latest'].mean():.0f}")
        print(f"  Follower/friend ratio: median={df_filtered['follower_friend_ratio'].median():.2f}, "
              f"mean={df_filtered['follower_friend_ratio'].mean():.2f}")
        print(f"  Retweet ratio: median={df_filtered['retweet_ratio'].median():.2f}, "
              f"mean={df_filtered['retweet_ratio'].mean():.2f}")
    else:
        print(f"\n  No viable egos found. This is expected if running against")
        print(f"  pre-phase sample data (Jan 22, 2020 is before Phase 1 starts Feb 19).")
        print(f"  The full dataset on Lafayette's cluster will produce viable egos.")

    print(f"\n--- Threshold settings used ---")
    print(f"  MIN_TOTAL_TWEETS:           {MIN_TOTAL_TWEETS}")
    print(f"  MIN_PHASES_ACTIVE:          {MIN_PHASES_ACTIVE} of 5")
    print(f"  MIN_UNIQUE_REPLY_TARGETS:   {MIN_UNIQUE_REPLY_TARGETS}")
    print(f"  MAX_FOLLOWER_FRIEND_RATIO:  {MAX_FOLLOWER_FRIEND_RATIO}")
    print(f"  MAX_RETWEET_RATIO:          {MAX_RETWEET_RATIO}")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Module 1 output
    df_candidates = load_ego_candidates(INPUT_DIR, INPUT_DATE)

    # Apply filters
    print(f"\nApplying filters...")
    df_filtered, report = apply_filters(df_candidates)

    # Print summary report
    print_summary_report(df_candidates, df_filtered, report)

    # Save outputs
    ego_output = os.path.join(OUTPUT_DIR, f"ego_list_{RUN_DATE}.csv")
    report_output = os.path.join(OUTPUT_DIR, f"filtering_report_{RUN_DATE}.csv")

    df_filtered.to_csv(ego_output, index=False)
    pd.DataFrame(report).to_csv(report_output, index=False)

    print(f"\nOutputs saved:")
    print(f"  Viable egos: {ego_output} ({len(df_filtered):,} rows)")
    print(f"  Filter report: {report_output}")
    print(f"\nNext step: Run Module 3 (Interaction Extraction) on {ego_output}")

    return df_filtered, report


if __name__ == "__main__":
    df_filtered, report = main()
