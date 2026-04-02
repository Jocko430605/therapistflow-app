"""
Module 1: Ego Candidate Identification
=======================================
Pipeline: COVID-19 Ego-Network Behavioral Activation Study
Version: 1.1
Authors: Ben (Kaiser Permanente) / Dr. Christian Lopez (Lafayette College)

Purpose:
    Reads raw Summary_Details CSV files, identifies all unique users (PersonIDs),
    and computes per-person summary statistics needed for ego filtering in Module 2.

Inputs:
    Raw Summary_Details CSV files following naming convention:
    YYYY_MM_DD_HH_Summary_Details.csv (or YYYY_MM_DD_HH_CT_Summary_Details.csv)

Outputs:
    {OUTPUT_DIR}/ego_candidates_{RUN_DATE}.csv
        One row per unique PersonID with summary stats across all input files.

    {OUTPUT_DIR}/ego_candidates_phase_activity_{RUN_DATE}.csv
        One row per PersonID per phase, with tweet counts and interaction counts.

Usage:
    1. Set DATA_DIR to the directory containing Summary_Details CSV files.
    2. Set OUTPUT_DIR to where output files should be written.
    3. Run: python module_01_ego_candidates.py
    4. Or open in Jupyter and run cell by cell.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

from pipeline_utils import (
    PHASES,
    find_summary_files,
    parse_twitter_dates,
    assign_phases,
)

# =============================================================================
# CONFIGURATION — Update these paths for your environment
# =============================================================================

# Directory containing raw Summary_Details CSV files
DATA_DIR = "./data/raw"

# Directory for output files
OUTPUT_DIR = "./data/processed"

# Run date stamp for output file naming
RUN_DATE = datetime.now().strftime("%Y_%m_%d")

# Minimum total tweets to be considered (very permissive first pass)
MIN_TOTAL_TWEETS = 1


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_and_parse_file(filepath):
    """
    Load a single Summary_Details CSV and parse key fields.
    Returns DataFrame with parsed datetime and phase assignment.
    """
    df = pd.read_csv(filepath)

    # Vectorised datetime parse (much faster than row-by-row .apply)
    df["datetime_utc"] = parse_twitter_dates(df["Date Created"])

    # Vectorised phase assignment
    df["phase"] = assign_phases(df["datetime_utc"])

    # Flag interaction types
    df["is_retweet"] = df["RT"] == "YES"
    df["is_reply"] = df["reply_to_user_ID"].notna()
    df["has_original_id"] = df["Orginal_ID"].notna()

    # Clean up PersonID — use nullable int to survive NaN rows
    df["PersonID"] = pd.array(df["PersonID"].values, dtype=pd.Int64Dtype())
    df = df.dropna(subset=["PersonID"])

    # Clean reply_to_user_ID (float due to NaN, convert to nullable int)
    df["reply_to_user_ID_clean"] = pd.array(
        df["reply_to_user_ID"].values, dtype=pd.Int64Dtype()
    )

    return df


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_all_files(data_dir):
    """
    Load all Summary_Details files, combine, and compute per-person stats.
    Returns:
        person_summary: DataFrame with one row per PersonID
        phase_activity: DataFrame with one row per PersonID per phase
        combined_df: Full combined DataFrame (for inspection/debugging)
    """
    files = find_summary_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No Summary_Details CSV files found in {data_dir}. "
            f"Check DATA_DIR path."
        )

    # Load and combine all files
    print("Loading and parsing files...")
    dfs = []
    for i, f in enumerate(files):
        df = load_and_parse_file(f)
        dfs.append(df)
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"  Processed {i + 1}/{len(files)} files")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal tweets loaded: {len(combined):,}")
    print(f"Unique PersonIDs: {combined['PersonID'].nunique():,}")
    print(f"Date range: {combined['datetime_utc'].min()} to {combined['datetime_utc'].max()}")

    # ----- Per-person summary stats -----
    print("\nComputing per-person summary statistics...")

    person_groups = combined.groupby("PersonID")

    person_summary = pd.DataFrame({
        "PersonID": person_groups["Tweet_ID"].first().index,
    })

    # Tweet counts
    person_summary["total_tweets"] = person_groups["Tweet_ID"].count().values
    person_summary["total_original_tweets"] = person_groups["is_retweet"].apply(
        lambda x: (~x).sum()
    ).values
    person_summary["total_retweets"] = person_groups["is_retweet"].sum().values
    person_summary["total_replies"] = person_groups["is_reply"].sum().values

    # Interaction metrics (key for ego-network viability)
    # Unique accounts this person replied to
    person_summary["unique_reply_targets"] = person_groups["reply_to_user_ID_clean"].apply(
        lambda x: x.dropna().nunique()
    ).values

    # Total interaction partners (reply targets for now; will expand with
    # mention_user_IDs and retweeted_user_ID when those fields are available)
    person_summary["unique_interaction_partners"] = person_summary["unique_reply_targets"]

    # Temporal range
    person_summary["first_tweet_date"] = person_groups["datetime_utc"].min().values
    person_summary["last_tweet_date"] = person_groups["datetime_utc"].max().values
    person_summary["active_days_span"] = (
        person_summary["last_tweet_date"] - person_summary["first_tweet_date"]
    ).dt.days

    # Account metrics (snapshot from most recent tweet)
    latest_idx = person_groups["datetime_utc"].idxmax()
    latest_tweets = combined.loc[latest_idx]
    person_summary["followers_latest"] = latest_tweets["followers"].values
    person_summary["friends_latest"] = latest_tweets["friends"].values
    person_summary["favorites_latest"] = latest_tweets["favorites"].values

    # Follower-to-friend ratio (useful for bot/corporate filtering)
    person_summary["follower_friend_ratio"] = np.where(
        person_summary["friends_latest"] > 0,
        person_summary["followers_latest"] / person_summary["friends_latest"],
        np.nan,
    )

    # Retweet ratio (high retweet ratio may indicate bot behavior)
    person_summary["retweet_ratio"] = np.where(
        person_summary["total_tweets"] > 0,
        person_summary["total_retweets"] / person_summary["total_tweets"],
        np.nan,
    )

    # ----- Phase activity -----
    print("Computing phase activity...")

    phase_activity = (
        combined.groupby(["PersonID", "phase"])
        .agg(
            tweet_count=("Tweet_ID", "count"),
            original_count=("is_retweet", lambda x: (~x).sum()),
            retweet_count=("is_retweet", "sum"),
            reply_count=("is_reply", "sum"),
            unique_reply_targets=("reply_to_user_ID_clean", lambda x: x.dropna().nunique()),
        )
        .reset_index()
    )

    # Pivot phase presence into person_summary (binary: active in phase Y/N)
    phase_keys = list(PHASES.keys())
    for phase_key in phase_keys:
        phase_persons = phase_activity[
            phase_activity["phase"] == phase_key
        ]["PersonID"].unique()
        person_summary[f"active_in_{phase_key}"] = person_summary["PersonID"].isin(
            phase_persons
        )

    # Count how many phases each person was active in
    phase_cols = [f"active_in_{pk}" for pk in phase_keys]
    person_summary["phases_active_count"] = person_summary[phase_cols].sum(axis=1)

    # Sort by total tweets descending
    person_summary = person_summary.sort_values("total_tweets", ascending=False)

    return person_summary, phase_activity, combined


def print_summary_report(person_summary, phase_activity):
    """Print a human-readable summary of ego candidate statistics."""
    n = len(person_summary)
    print("\n" + "=" * 70)
    print("MODULE 1: EGO CANDIDATE IDENTIFICATION — SUMMARY REPORT")
    print("=" * 70)

    print(f"\nTotal unique PersonIDs: {n:,}")
    print(f"Total tweets processed: {person_summary['total_tweets'].sum():,}")

    print(f"\n--- Tweet volume distribution ---")
    for threshold in [1, 5, 10, 20, 50, 100, 500]:
        count = (person_summary["total_tweets"] >= threshold).sum()
        print(f"  {threshold:>4d}+ tweets: {count:>6,} persons ({count/n*100:.1f}%)")

    print(f"\n--- Interaction partners (reply targets) ---")
    for threshold in [1, 5, 10, 15, 20, 50]:
        count = (person_summary["unique_reply_targets"] >= threshold).sum()
        print(f"  {threshold:>4d}+ reply targets: {count:>6,} persons ({count/n*100:.1f}%)")

    print(f"\n--- Phase activity ---")
    for phase_key, phase_def in PHASES.items():
        col = f"active_in_{phase_key}"
        count = person_summary[col].sum()
        print(f"  {phase_def['label']:45s}: {count:>6,} ({count/n*100:.1f}%)")

    print(f"\n--- Phases active count distribution ---")
    for num_phases in range(6):
        count = (person_summary["phases_active_count"] == num_phases).sum()
        print(f"  Active in {num_phases} phases: {count:>6,} ({count/n*100:.1f}%)")

    # Key viability metric
    viable_4plus = (person_summary["phases_active_count"] >= 4).sum()
    viable_4plus_interacting = (
        (person_summary["phases_active_count"] >= 4)
        & (person_summary["unique_reply_targets"] >= 5)
    ).sum()
    print(f"\n--- Ego viability (preliminary) ---")
    print(f"  Active in 4+ phases: {viable_4plus:,}")
    print(f"  Active in 4+ phases AND 5+ reply targets: {viable_4plus_interacting:,}")

    print(f"\n--- Account characteristics ---")
    print(f"  Followers: median={person_summary['followers_latest'].median():.0f}, "
          f"mean={person_summary['followers_latest'].mean():.0f}, "
          f"max={person_summary['followers_latest'].max():,}")
    print(f"  Follower/friend ratio: median={person_summary['follower_friend_ratio'].median():.2f}, "
          f"mean={person_summary['follower_friend_ratio'].mean():.2f}")
    print(f"  Retweet ratio: median={person_summary['retweet_ratio'].median():.2f}, "
          f"mean={person_summary['retweet_ratio'].mean():.2f}")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process all files
    person_summary, phase_activity, combined = process_all_files(DATA_DIR)

    # Print summary report
    print_summary_report(person_summary, phase_activity)

    # Save outputs
    person_output = os.path.join(
        OUTPUT_DIR, f"ego_candidates_{RUN_DATE}.csv"
    )
    phase_output = os.path.join(
        OUTPUT_DIR, f"ego_candidates_phase_activity_{RUN_DATE}.csv"
    )

    person_summary.to_csv(person_output, index=False)
    phase_activity.to_csv(phase_output, index=False)

    print(f"\nOutputs saved:")
    print(f"  Person summary: {person_output}")
    print(f"  Phase activity: {phase_output}")
    print(f"\nNext step: Run Module 2 (Ego Filtering) on {person_output}")

    return person_summary, phase_activity


if __name__ == "__main__":
    person_summary, phase_activity = main()
