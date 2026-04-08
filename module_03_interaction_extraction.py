"""
Module 3: Interaction Extraction
==================================
Pipeline: COVID-19 Ego-Network Behavioral Activation Study
Version: 1.1
Authors: Ben (Kaiser Permanente) / Dr. Christian Lopez (Lafayette College)

Purpose:
    Scans raw Summary_Details CSVs and extracts all person-to-person
    interactions involving viable egos from Module 2's ego_list. Produces
    directed edges (ego <-> alter) that Module 4 uses for Dunbar layer
    assignment.

    Handles three interaction types, gracefully skipping any whose fields
    are not present in the current data extract:
      - Replies (reply_to_user_ID) -- strongest signal
      - Mentions (mention_user_IDs) -- moderate signal, may not be available
      - Retweets (retweeted_user_ID) -- weakest signal, may not be available

Inputs:
    {INPUT_DIR}/ego_list_{INPUT_DATE}.csv
        Module 2 output: one row per viable ego PersonID.

    {DATA_DIR}/*_Summary_Details.csv
        Raw tweet files (local sample or full Lafayette dataset).

Outputs:
    {OUTPUT_DIR}/interaction_edges_{RUN_DATE}.csv
        One row per interaction event (directed edge).

    {OUTPUT_DIR}/interaction_summary_{RUN_DATE}.csv
        One row per ego-alter pair with aggregated counts.

Usage:
    1. Set DATA_DIR to the directory containing raw Summary_Details CSVs.
    2. Set INPUT_DIR and INPUT_DATE to locate the Module 2 ego_list.
    3. Set OUTPUT_DIR to where output files should be written.
    4. Run: python module_03_interaction_extraction.py
    5. Or open in Jupyter and run cell by cell.
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

# Directory containing Module 2 output (ego_list)
INPUT_DIR = "./data/processed"

# Date stamp of the Module 2 ego_list file to read
INPUT_DATE = "2026_03_28"

# Directory for output files
OUTPUT_DIR = "./data/processed"

# Run date stamp for output file naming
RUN_DATE = datetime.now().strftime("%Y_%m_%d")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_ego_list(input_dir, input_date):
    """
    Load the ego_list CSV from Module 2.
    Returns set of viable ego PersonIDs for fast lookup.
    """
    filepath = os.path.join(input_dir, f"ego_list_{input_date}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Ego list file not found: {filepath}\n"
            f"Check INPUT_DIR and INPUT_DATE settings. "
            f"Run Module 2 first if needed."
        )

    df = pd.read_csv(filepath)
    ego_ids = set(df["PersonID"].values)
    print(f"Loaded {len(ego_ids):,} viable ego PersonIDs from {filepath}")
    return ego_ids


def detect_available_fields(sample_df):
    """
    Check which interaction-relevant fields are available in the data.
    Returns a dict of field availability and logs status.
    """
    available = {}

    # reply_to_user_ID — always expected
    available["reply_to_user_ID"] = "reply_to_user_ID" in sample_df.columns
    if available["reply_to_user_ID"]:
        print("  [AVAILABLE] reply_to_user_ID — reply interactions will be extracted")
    else:
        print("  [MISSING]   reply_to_user_ID — WARNING: primary interaction field not found")

    # mention_user_IDs — may not be in current extract
    available["mention_user_IDs"] = "mention_user_IDs" in sample_df.columns
    if available["mention_user_IDs"]:
        print("  [AVAILABLE] mention_user_IDs — mention interactions will be extracted")
    else:
        print("  [SKIPPED]   mention_user_IDs — not in current extract, mention interactions skipped")

    # retweeted_user_ID — may not be in current extract
    available["retweeted_user_ID"] = "retweeted_user_ID" in sample_df.columns
    if available["retweeted_user_ID"]:
        print("  [AVAILABLE] retweeted_user_ID — retweet interactions will be extracted")
    else:
        has_original_id = "Orginal_ID" in sample_df.columns
        if has_original_id:
            print("  [SKIPPED]   retweeted_user_ID — not in current extract. "
                  "Orginal_ID is present (can identify retweet events) but "
                  "retweeted author PersonID is unknown, so retweet edges cannot be built.")
        else:
            print("  [SKIPPED]   retweeted_user_ID — not in current extract, "
                  "retweet interactions skipped")

    return available


# =============================================================================
# VECTORISED INTERACTION EXTRACTION
# =============================================================================

def _extract_reply_edges(df, ego_ids):
    """
    Vectorised extraction of reply-based edges (outbound and inbound).
    Returns a list of DataFrames (may be empty).
    """
    frames = []
    replies = df[df["reply_to_user_ID"].notna()].copy()
    if len(replies) == 0:
        return frames

    replies["reply_to_user_ID_int"] = replies["reply_to_user_ID"].astype(np.int64)

    # Outbound: ego replied to someone
    ego_mask = replies["PersonID"].isin(ego_ids)
    if ego_mask.any():
        out = replies.loc[ego_mask, ["PersonID", "reply_to_user_ID_int",
                                      "Tweet_ID", "datetime_utc", "phase"]].copy()
        out.columns = ["ego_PersonID", "alter_PersonID", "tweet_id", "datetime_utc", "phase"]
        out["interaction_type"] = "reply"
        out["direction"] = "outbound"
        frames.append(out)

    # Inbound: non-ego replied to an ego
    inbound_mask = (~ego_mask) & replies["reply_to_user_ID_int"].isin(ego_ids)
    if inbound_mask.any():
        inb = replies.loc[inbound_mask, ["reply_to_user_ID_int", "PersonID",
                                          "Tweet_ID", "datetime_utc", "phase"]].copy()
        inb.columns = ["ego_PersonID", "alter_PersonID", "tweet_id", "datetime_utc", "phase"]
        inb["interaction_type"] = "reply"
        inb["direction"] = "inbound"
        frames.append(inb)

    return frames


def _extract_mention_edges(df, ego_ids):
    """
    Vectorised extraction of mention-based edges.
    Explodes the comma-delimited mention_user_IDs field into individual rows,
    then identifies ego-involved edges.
    """
    frames = []
    mentions = df[df["mention_user_IDs"].notna()].copy()
    if len(mentions) == 0:
        return frames

    # Explode comma-delimited mention IDs into one row per mention
    mentions["mention_list"] = mentions["mention_user_IDs"].astype(str).str.split(",")
    exploded = mentions.explode("mention_list")
    exploded["mention_id"] = pd.to_numeric(
        exploded["mention_list"].str.strip(), errors="coerce"
    ).astype("Int64")
    exploded = exploded.dropna(subset=["mention_id"])
    exploded["mention_id"] = exploded["mention_id"].astype(np.int64)

    # Drop self-mentions
    exploded = exploded[exploded["PersonID"] != exploded["mention_id"]]
    if len(exploded) == 0:
        return frames

    author_is_ego = exploded["PersonID"].isin(ego_ids)
    target_is_ego = exploded["mention_id"].isin(ego_ids)

    # Outbound: ego mentioned someone
    out_mask = author_is_ego
    if out_mask.any():
        out = exploded.loc[out_mask, ["PersonID", "mention_id",
                                       "Tweet_ID", "datetime_utc", "phase"]].copy()
        out.columns = ["ego_PersonID", "alter_PersonID", "tweet_id", "datetime_utc", "phase"]
        out["interaction_type"] = "mention"
        out["direction"] = "outbound"
        frames.append(out)

    # Inbound: non-ego mentioned an ego
    inb_mask = (~author_is_ego) & target_is_ego
    if inb_mask.any():
        inb = exploded.loc[inb_mask, ["mention_id", "PersonID",
                                       "Tweet_ID", "datetime_utc", "phase"]].copy()
        inb.columns = ["ego_PersonID", "alter_PersonID", "tweet_id", "datetime_utc", "phase"]
        inb["interaction_type"] = "mention"
        inb["direction"] = "inbound"
        frames.append(inb)

    return frames


def _extract_retweet_edges(df, ego_ids):
    """
    Vectorised extraction of retweet-based edges (outbound and inbound).
    """
    frames = []
    retweets = df[df["retweeted_user_ID"].notna()].copy()
    if len(retweets) == 0:
        return frames

    retweets["retweeted_user_ID_int"] = retweets["retweeted_user_ID"].astype(np.int64)

    ego_mask = retweets["PersonID"].isin(ego_ids)

    # Outbound: ego retweeted someone
    if ego_mask.any():
        out = retweets.loc[ego_mask, ["PersonID", "retweeted_user_ID_int",
                                       "Tweet_ID", "datetime_utc", "phase"]].copy()
        out.columns = ["ego_PersonID", "alter_PersonID", "tweet_id", "datetime_utc", "phase"]
        out["interaction_type"] = "retweet"
        out["direction"] = "outbound"
        frames.append(out)

    # Inbound: non-ego retweeted an ego
    inb_mask = (~ego_mask) & retweets["retweeted_user_ID_int"].isin(ego_ids)
    if inb_mask.any():
        inb = retweets.loc[inb_mask, ["retweeted_user_ID_int", "PersonID",
                                       "Tweet_ID", "datetime_utc", "phase"]].copy()
        inb.columns = ["ego_PersonID", "alter_PersonID", "tweet_id", "datetime_utc", "phase"]
        inb["interaction_type"] = "retweet"
        inb["direction"] = "inbound"
        frames.append(inb)

    return frames


def extract_interactions_from_file(df, ego_ids, available_fields):
    """
    Vectorised extraction of all ego-involved interactions from a single
    parsed DataFrame.  Returns a DataFrame of edge records (may be empty).
    """
    # Parse datetime and phase once for the whole file
    df["datetime_utc"] = parse_twitter_dates(df["Date Created"])
    df["phase"] = assign_phases(df["datetime_utc"])

    edge_frames = []

    if available_fields.get("reply_to_user_ID"):
        edge_frames.extend(_extract_reply_edges(df, ego_ids))

    if available_fields.get("mention_user_IDs"):
        edge_frames.extend(_extract_mention_edges(df, ego_ids))

    if available_fields.get("retweeted_user_ID"):
        edge_frames.extend(_extract_retweet_edges(df, ego_ids))

    if edge_frames:
        return pd.concat(edge_frames, ignore_index=True)
    return pd.DataFrame(columns=[
        "ego_PersonID", "alter_PersonID", "interaction_type",
        "direction", "tweet_id", "datetime_utc", "phase",
    ])


def process_all_files(data_dir, ego_ids):
    """
    Scan all Summary_Details CSVs and extract ego-involved interactions.

    Performance: loads the ego_id set into memory first, then filters each
    CSV as it's read — never loads the full dataset into memory at once.

    Returns:
        edges_df: DataFrame of all interaction edges
        available_fields: dict of which interaction types were available
    """
    files = find_summary_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No Summary_Details CSV files found in {data_dir}. "
            f"Check DATA_DIR path."
        )

    # Detect available fields from first file
    print("\nDetecting available interaction fields...")
    sample_df = pd.read_csv(files[0], nrows=5)
    available_fields = detect_available_fields(sample_df)

    # Process each file
    print(f"\nExtracting interactions from {len(files)} files...")
    all_edge_frames = []
    total_edges = 0
    for i, filepath in enumerate(files):
        df = pd.read_csv(filepath)

        # Quick filter: only process file if it contains any ego PersonIDs
        # or any reply targets that are egos. This is the performance
        # optimization for the full Lafayette dataset.
        person_ids_in_file = set(df["PersonID"].values)
        has_egos = bool(person_ids_in_file & ego_ids)

        has_inbound = False
        if available_fields.get("reply_to_user_ID") and "reply_to_user_ID" in df.columns:
            reply_targets = set(df["reply_to_user_ID"].dropna().astype(np.int64).values)
            has_inbound = bool(reply_targets & ego_ids)

        # NOTE: when mention_user_IDs or retweeted_user_ID become available,
        # extend this check to also look for ego IDs in those columns.

        if has_egos or has_inbound:
            file_edges = extract_interactions_from_file(df, ego_ids, available_fields)
            if len(file_edges) > 0:
                all_edge_frames.append(file_edges)
                total_edges += len(file_edges)

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  Processed {i + 1}/{len(files)} files "
                  f"({total_edges:,} edges so far)")

    if not all_edge_frames:
        print("\nNo interactions found. This may indicate:")
        print("  - The ego_list is empty")
        print("  - The raw data doesn't overlap with the ego PersonIDs")
        print("  - All data is outside phase windows")
        edges_df = pd.DataFrame(columns=[
            "ego_PersonID", "alter_PersonID", "interaction_type",
            "direction", "tweet_id", "datetime_utc", "phase",
        ])
    else:
        edges_df = pd.concat(all_edge_frames, ignore_index=True)
        edges_df = edges_df.sort_values(["ego_PersonID", "datetime_utc"]).reset_index(drop=True)

    return edges_df, available_fields


# =============================================================================
# INTERACTION SUMMARY (EGO-ALTER PAIR AGGREGATION)
# =============================================================================

def build_interaction_summary(edges_df):
    """
    Aggregate interaction edges into one row per ego-alter pair.
    Computes counts by type and direction, reciprocity, and phase span.
    """
    if len(edges_df) == 0:
        return pd.DataFrame(columns=[
            "ego_PersonID", "alter_PersonID", "total_interactions",
            "reply_count_outbound", "reply_count_inbound",
            "mention_count_outbound", "mention_count_inbound",
            "retweet_count_outbound", "retweet_count_inbound",
            "reciprocal", "phases_interacted",
        ])

    pairs = edges_df.groupby(["ego_PersonID", "alter_PersonID"])

    def agg_pair(group):
        """Aggregate a single ego-alter pair's interactions."""
        result = {
            "total_interactions": len(group),
            "reply_count_outbound": ((group["interaction_type"] == "reply") & (group["direction"] == "outbound")).sum(),
            "reply_count_inbound": ((group["interaction_type"] == "reply") & (group["direction"] == "inbound")).sum(),
            "mention_count_outbound": ((group["interaction_type"] == "mention") & (group["direction"] == "outbound")).sum(),
            "mention_count_inbound": ((group["interaction_type"] == "mention") & (group["direction"] == "inbound")).sum(),
            "retweet_count_outbound": ((group["interaction_type"] == "retweet") & (group["direction"] == "outbound")).sum(),
            "retweet_count_inbound": ((group["interaction_type"] == "retweet") & (group["direction"] == "inbound")).sum(),
            "phases_interacted": group["phase"].nunique(),
        }
        # Reciprocal = both outbound and inbound interactions exist
        has_outbound = (group["direction"] == "outbound").any()
        has_inbound = (group["direction"] == "inbound").any()
        result["reciprocal"] = has_outbound and has_inbound
        return pd.Series(result)

    summary = pairs.apply(agg_pair, include_groups=False).reset_index()

    # Ensure integer types for count columns
    count_cols = [
        "total_interactions", "reply_count_outbound", "reply_count_inbound",
        "mention_count_outbound", "mention_count_inbound",
        "retweet_count_outbound", "retweet_count_inbound",
        "phases_interacted",
    ]
    for col in count_cols:
        summary[col] = summary[col].astype(int)

    # Sort by ego then total interactions descending
    summary = summary.sort_values(
        ["ego_PersonID", "total_interactions"], ascending=[True, False]
    ).reset_index(drop=True)

    return summary


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary_report(edges_df, summary_df, ego_ids, available_fields):
    """Print a human-readable summary of the extraction results."""
    n_edges = len(edges_df)
    n_egos = len(ego_ids)

    print("\n" + "=" * 70)
    print("MODULE 3: INTERACTION EXTRACTION — SUMMARY REPORT")
    print("=" * 70)

    # --- Field availability ---
    print(f"\n--- Interaction field availability ---")
    for field, avail in available_fields.items():
        status = "AVAILABLE" if avail else "NOT IN DATA"
        print(f"  {field:25s}: {status}")

    if n_edges == 0:
        print(f"\nNo interactions extracted from {n_egos} viable egos.")
        print("This is expected if the raw data doesn't contain the ego PersonIDs")
        print("(e.g., running against the Jan 22 sample with Lafayette-derived egos).")
        print("\n" + "=" * 70)
        return

    # --- Overall counts ---
    egos_with_interactions = edges_df["ego_PersonID"].nunique()
    print(f"\n--- Overall ---")
    print(f"  Viable egos: {n_egos}")
    print(f"  Egos with interactions: {egos_with_interactions} "
          f"({egos_with_interactions / n_egos * 100:.1f}%)")
    print(f"  Total interaction edges: {n_edges:,}")
    print(f"  Unique ego-alter pairs: {len(summary_df):,}")

    # --- Breakdown by type ---
    print(f"\n--- By interaction type ---")
    for itype in ["reply", "mention", "retweet"]:
        type_edges = edges_df[edges_df["interaction_type"] == itype]
        if len(type_edges) > 0:
            outbound = (type_edges["direction"] == "outbound").sum()
            inbound = (type_edges["direction"] == "inbound").sum()
            print(f"  {itype:10s}: {len(type_edges):>6,} edges "
                  f"(outbound: {outbound:,}, inbound: {inbound:,})")
        else:
            print(f"  {itype:10s}:      0 edges (field not available or no interactions)")

    # --- By phase ---
    print(f"\n--- By phase ---")
    for phase_key, phase_def in PHASES.items():
        phase_edges = edges_df[edges_df["phase"] == phase_key]
        print(f"  {phase_def['label']:45s}: {len(phase_edges):>6,} edges")
    outside = edges_df[edges_df["phase"] == "outside_phases"]
    if len(outside) > 0:
        print(f"  {'Outside phase windows':45s}: {len(outside):>6,} edges")

    # --- Alters per ego ---
    alters_per_ego = summary_df.groupby("ego_PersonID")["alter_PersonID"].count()
    print(f"\n--- Unique alters per ego ---")
    print(f"  Mean:   {alters_per_ego.mean():.1f}")
    print(f"  Median: {alters_per_ego.median():.1f}")
    print(f"  Min:    {alters_per_ego.min()}")
    print(f"  Max:    {alters_per_ego.max()}")

    # --- Reciprocity ---
    n_reciprocal = summary_df["reciprocal"].sum()
    print(f"\n--- Reciprocity ---")
    print(f"  Reciprocal ego-alter pairs: {n_reciprocal:,} "
          f"({n_reciprocal / len(summary_df) * 100:.1f}% of pairs)")

    # --- Interaction intensity distribution ---
    print(f"\n--- Interactions per ego-alter pair ---")
    print(f"  Mean:   {summary_df['total_interactions'].mean():.1f}")
    print(f"  Median: {summary_df['total_interactions'].median():.1f}")
    print(f"  Max:    {summary_df['total_interactions'].max()}")
    for threshold in [1, 2, 5, 10]:
        count = (summary_df["total_interactions"] >= threshold).sum()
        print(f"  {threshold}+ interactions: {count:,} pairs "
              f"({count / len(summary_df) * 100:.1f}%)")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load ego list from Module 2
    ego_ids = load_ego_list(INPUT_DIR, INPUT_DATE)

    if not ego_ids:
        print("No viable egos in ego_list. Nothing to extract.")
        print("Run Module 2 first, or check INPUT_DIR/INPUT_DATE settings.")
        return None, None

    # Extract interactions
    edges_df, available_fields = process_all_files(DATA_DIR, ego_ids)

    # Build ego-alter summary
    print("\nBuilding ego-alter interaction summary...")
    summary_df = build_interaction_summary(edges_df)

    # Print report
    print_summary_report(edges_df, summary_df, ego_ids, available_fields)

    # Save outputs
    edges_output = os.path.join(OUTPUT_DIR, f"interaction_edges_{RUN_DATE}.csv")
    summary_output = os.path.join(OUTPUT_DIR, f"interaction_summary_{RUN_DATE}.csv")

    edges_df.to_csv(edges_output, index=False)
    summary_df.to_csv(summary_output, index=False)

    print(f"\nOutputs saved:")
    print(f"  Interaction edges: {edges_output} ({len(edges_df):,} rows)")
    print(f"  Interaction summary: {summary_output} ({len(summary_df):,} rows)")
    print(f"\nNext step: Run Module 4 (Dunbar Assignment) on {summary_output}")

    return edges_df, summary_df


if __name__ == "__main__":
    edges_df, summary_df = main()
