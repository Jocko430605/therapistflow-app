"""
Module 4: Dunbar Layer Assignment
====================================
Pipeline: COVID-19 Ego-Network Behavioral Activation Study
Version: 1.1
Authors: Ben (Kaiser Permanente) / Dr. Christian Lopez (Lafayette College)

Purpose:
    Takes ego-alter interaction data from Module 3 and assigns each alter to
    a Dunbar layer within the ego's social network. Uses four relationship
    signals to compute a composite tie-strength score:

      1. Interaction intensity -- weighted combination of reply, mention,
         and retweet counts (replies strongest, retweets weakest)
      2. Linguistic intimacy -- GoEmotions-based emotion classification of
         reply text (requires full_text field; zero contribution if absent)
      3. Initiation balance -- how equally ego and alter initiate contact
      4. Temporal consistency -- how many phases the pair interacted across

    Alters are ranked by composite score and assigned to Dunbar layers:
      inner_5 (1-5), middle_15 (6-15), outer_50 (16-50),
      outer_150 (51-150), beyond_150 (151+)

Inputs:
    {INPUT_DIR}/interaction_summary_{INPUT_DATE}.csv
    {INPUT_DIR}/interaction_edges_{INPUT_DATE}.csv
    {DATA_DIR}/*_Summary_Details.csv (for linguistic intimacy if full_text available)

Outputs:
    {OUTPUT_DIR}/ego_network_{RUN_DATE}.csv
    {OUTPUT_DIR}/network_summary_{RUN_DATE}.csv

Usage:
    1. Set DATA_DIR, INPUT_DIR, INPUT_DATE, and OUTPUT_DIR.
    2. Adjust scoring weights and thresholds as needed.
    3. Run: python module_04_dunbar_assignment.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

from pipeline_utils import (
    PHASES,
    DUNBAR_LAYERS,
    TOTAL_PHASES,
    find_summary_files,
    load_goemotions_classifier,
    load_emotion_cache,
    save_emotion_cache,
    classify_tweets_batch,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "./data/raw"
INPUT_DIR = "./data/processed"
INPUT_DATE = "2026_03_28"
OUTPUT_DIR = "./data/processed"
RUN_DATE = datetime.now().strftime("%Y_%m_%d")

# =============================================================================
# INTERACTION INTENSITY WEIGHTS
# =============================================================================

W_REPLY = 3.0
W_MENTION = 2.0
W_RETWEET = 1.0
RECIPROCITY_BONUS = 1.5

# =============================================================================
# COMPOSITE SCORE WEIGHTS
# =============================================================================

W_INTIMACY = 0.5
W_BALANCE = 0.5
W_CONSISTENCY = 0.5

# =============================================================================
# GOEMOTIONS CONFIGURATION
# =============================================================================

ENABLE_GOEMOTIONS = True
GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"
GOEMOTIONS_BATCH_SIZE = 64
GOEMOTIONS_TOP_K = 5
GOEMOTIONS_DEVICE = "cpu"
EMOTION_CACHE_DIR = "./data/processed"

EMOTION_INTIMACY_WEIGHTS = {
    # High intimacy (1.0)
    "love": 1.0, "caring": 1.0, "grief": 1.0, "gratitude": 1.0,
    "relief": 1.0, "desire": 1.0, "nervousness": 1.0,
    # Moderate intimacy (0.5)
    "joy": 0.5, "sadness": 0.5, "fear": 0.5, "anger": 0.5,
    "surprise": 0.5, "excitement": 0.5, "pride": 0.5, "admiration": 0.5,
    "amusement": 0.5, "embarrassment": 0.5, "remorse": 0.5,
    "optimism": 0.5, "disappointment": 0.5,
    # Low intimacy / informational (0.0)
    "neutral": 0.0, "curiosity": 0.0, "approval": 0.0,
    "disapproval": 0.0, "realization": 0.0, "confusion": 0.0,
    "annoyance": 0.0,
    # Counter-intimacy (-0.3)
    "disgust": -0.3,
}

# =============================================================================
# DUNBAR LAYER BOUNDARIES
# =============================================================================

LAYER_BOUNDARIES = [
    ("inner_5", 1, 5),
    ("middle_15", 6, 15),
    ("outer_50", 16, 50),
    ("outer_150", 51, 150),
    ("beyond_150", 151, None),
]

# =============================================================================
# MINIMUM VIABLE NETWORK THRESHOLDS (flag, not remove)
# =============================================================================

MIN_INNER = 2
MIN_MIDDLE = 5
MIN_OUTER_50 = 10
MIN_OUTER_150 = 15
MIN_TOTAL_ALTERS = 30


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_interaction_data(input_dir, input_date):
    """Load Module 3 outputs."""
    summary_path = os.path.join(input_dir, f"interaction_summary_{input_date}.csv")
    edges_path = os.path.join(input_dir, f"interaction_edges_{input_date}.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Interaction summary not found: {summary_path}\n"
            f"Run Module 3 first if needed."
        )
    if not os.path.exists(edges_path):
        raise FileNotFoundError(
            f"Interaction edges not found: {edges_path}\n"
            f"Run Module 3 first if needed."
        )

    summary_df = pd.read_csv(summary_path)
    edges_df = pd.read_csv(edges_path)
    print(f"Loaded {len(summary_df):,} ego-alter pairs from {summary_path}")
    print(f"Loaded {len(edges_df):,} interaction edges from {edges_path}")
    return summary_df, edges_df


def compute_intimacy_from_emotions(emotions_list):
    """
    Compute a single intimacy score from a list of top-k emotion dicts.
    Returns a float in [0, 1] range.
    """
    if not emotions_list:
        return 0.0
    weighted_sum = 0.0
    total_prob = 0.0
    for em in emotions_list:
        weight = EMOTION_INTIMACY_WEIGHTS.get(em["label"], 0.0)
        weighted_sum += weight * em["score"]
        total_prob += em["score"]
    if total_prob > 0:
        raw = weighted_sum / total_prob  # Range: ~ -0.3 to 1.0
        return max(0.0, min(1.0, (raw + 0.3) / 1.3))
    return 0.0


# =============================================================================
# SIGNAL 1: INTERACTION INTENSITY
# =============================================================================

def compute_interaction_intensity(summary_df):
    """
    Compute weighted interaction intensity score per ego-alter pair.
    Replies weighted highest, retweets lowest. Reciprocity amplifies.
    """
    reply_total = summary_df["reply_count_outbound"] + summary_df["reply_count_inbound"]
    mention_total = summary_df["mention_count_outbound"] + summary_df["mention_count_inbound"]
    retweet_total = summary_df["retweet_count_outbound"] + summary_df["retweet_count_inbound"]

    raw_score = (
        W_REPLY * reply_total
        + W_MENTION * mention_total
        + W_RETWEET * retweet_total
    )

    raw_score = pd.Series(
        np.where(summary_df["reciprocal"], raw_score * RECIPROCITY_BONUS, raw_score),
        index=summary_df.index,
    )

    return raw_score, reply_total, mention_total, retweet_total


# =============================================================================
# SIGNAL 2: GOEMOTIONS-BASED LINGUISTIC INTIMACY
# =============================================================================

def extract_needed_tweet_texts(data_dir, needed_tweet_ids):
    """Scan raw CSV files once and extract full_text for needed_tweet_ids."""
    files = find_summary_files(data_dir)
    if not files:
        print("  [SKIPPED] No raw CSV files found")
        return {}, False

    sample = pd.read_csv(files[0], nrows=5)
    if "full_text" not in sample.columns:
        print("  [SKIPPED] full_text column not in data")
        return {}, False

    print(f"  Scanning {len(files)} files to load full_text for "
          f"{len(needed_tweet_ids):,} tweets...")

    tweet_texts = {}
    for filepath in files:
        df = pd.read_csv(filepath, usecols=["Tweet_ID", "full_text"])
        matches = df[df["Tweet_ID"].isin(needed_tweet_ids)].copy()
        if len(matches) > 0:
            batch = dict(zip(
                matches["Tweet_ID"].astype(int),
                matches["full_text"].astype(str),
            ))
            tweet_texts.update(batch)
        if len(tweet_texts) >= len(needed_tweet_ids):
            break

    print(f"  Found text for {len(tweet_texts):,}/{len(needed_tweet_ids):,} tweets")
    return tweet_texts, True


def compute_goemotions_intimacy(summary_df, reply_edges, tweet_texts, has_full_text):
    """
    Compute linguistic intimacy using GoEmotions emotion classification.
    Returns (scores_series, available_bool, emotion_cache).
    """
    classifier, model_available = load_goemotions_classifier(
        GOEMOTIONS_MODEL, GOEMOTIONS_TOP_K, GOEMOTIONS_DEVICE, ENABLE_GOEMOTIONS
    )
    if not model_available or not has_full_text or len(reply_edges) == 0 or not tweet_texts:
        return pd.Series(0.0, index=summary_df.index), model_available and has_full_text, {}

    # Load existing cache, classify uncached tweets, save updated cache
    emotion_cache = load_emotion_cache(EMOTION_CACHE_DIR, GOEMOTIONS_TOP_K)
    tweet_ids_list = list(tweet_texts.keys())
    texts_list = [tweet_texts[tid] for tid in tweet_ids_list]
    emotion_cache = classify_tweets_batch(
        classifier, texts_list, tweet_ids_list, emotion_cache, GOEMOTIONS_BATCH_SIZE
    )
    save_emotion_cache(emotion_cache, EMOTION_CACHE_DIR, GOEMOTIONS_TOP_K)

    # Pre-build tweet_id -> intimacy_score lookup (vectorised instead of .apply)
    intimacy_lookup = {
        tid: compute_intimacy_from_emotions(ems)
        for tid, ems in emotion_cache.items()
    }

    # Map to edges and filter to those with text
    reply_edges = reply_edges[reply_edges["tweet_id"].isin(tweet_texts.keys())].copy()
    if len(reply_edges) == 0:
        return pd.Series(0.0, index=summary_df.index), True, emotion_cache

    reply_edges["goemotions_intimacy"] = (
        reply_edges["tweet_id"].astype(int).map(intimacy_lookup).fillna(0.0)
    )

    # Aggregate per ego-alter pair
    pair_scores = (
        reply_edges.groupby(["ego_PersonID", "alter_PersonID"])
        ["goemotions_intimacy"]
        .mean()
        .reset_index()
        .rename(columns={"goemotions_intimacy": "ge_intimacy_score"})
    )

    # Merge back to summary_df via index-aligned lookup
    merge_key = pd.MultiIndex.from_arrays(
        [summary_df["ego_PersonID"], summary_df["alter_PersonID"]]
    )
    score_key = pd.MultiIndex.from_arrays(
        [pair_scores["ego_PersonID"], pair_scores["alter_PersonID"]]
    )
    score_map = pd.Series(
        pair_scores["ge_intimacy_score"].values, index=score_key
    )
    scores = merge_key.map(score_map).fillna(0.0)
    scores = pd.Series(scores.values, index=summary_df.index, dtype=float)

    n_nonzero = int((scores > 0).sum())
    mean_nonzero = scores[scores > 0].mean() if n_nonzero > 0 else 0.0
    print(f"  GoEmotions intimacy: mean={scores.mean():.3f}, "
          f"nonzero mean={mean_nonzero:.3f}, "
          f"pairs with scores={n_nonzero}")

    return scores, True, emotion_cache


# =============================================================================
# SIGNAL 3: INITIATION BALANCE
# =============================================================================

def compute_initiation_balance(summary_df):
    """
    Compute initiation balance. Perfectly balanced (0.5) -> 1.0,
    one-directional (0.0 or 1.0) -> 0.0.
    """
    outbound_total = (
        summary_df["reply_count_outbound"]
        + summary_df["mention_count_outbound"]
        + summary_df["retweet_count_outbound"]
    )
    total = summary_df["total_interactions"]
    outbound_ratio = np.where(total > 0, outbound_total / total, 0.5)
    balance_score = 1.0 - 2.0 * np.abs(outbound_ratio - 0.5)
    return pd.Series(balance_score, index=summary_df.index)


# =============================================================================
# SIGNAL 4: TEMPORAL CONSISTENCY
# =============================================================================

def compute_temporal_consistency(summary_df):
    """phases_interacted / TOTAL_PHASES -> score from 0.2 to 1.0."""
    return summary_df["phases_interacted"] / TOTAL_PHASES


# =============================================================================
# COMPOSITE SCORING AND LAYER ASSIGNMENT
# =============================================================================

def compute_composite_and_assign_layers(summary_df, edges_df, data_dir):
    """
    Compute all four signals, combine into composite score, rank alters
    per ego, and assign Dunbar layers.
    """
    signals_available = {}

    # --- Signal 1: Interaction intensity ---
    print("\nComputing Signal 1: Interaction intensity...")
    raw_intensity, reply_total, mention_total, retweet_total = (
        compute_interaction_intensity(summary_df)
    )

    # Normalize intensity per ego (min-max within each ego's alters)
    def normalize_per_ego(group):
        vals = group.values
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            return pd.Series(1.0, index=group.index)
        return (group - vmin) / (vmax - vmin)

    intensity_grouped = summary_df[["ego_PersonID"]].copy()
    intensity_grouped["_raw"] = raw_intensity
    intensity_score = (
        intensity_grouped.groupby("ego_PersonID")["_raw"]
        .transform(lambda g: normalize_per_ego(g))
    )
    signals_available["interaction_intensity"] = True
    print(f"  Intensity scores: mean={intensity_score.mean():.3f}, "
          f"median={intensity_score.median():.3f}")

    # --- Extract tweet text once for all signals ---
    reply_edges = edges_df[edges_df["interaction_type"] == "reply"].copy()
    if len(reply_edges) > 0:
        needed_tweet_ids = set(reply_edges["tweet_id"].values)
        print("\nExtracting required tweet text from raw files once for all signals...")
        tweet_texts, has_full_text = extract_needed_tweet_texts(data_dir, needed_tweet_ids)
    else:
        tweet_texts, has_full_text = {}, False

    # --- Signal 2: GoEmotions-based linguistic intimacy ---
    intimacy_score = pd.Series(0.0, index=summary_df.index)
    goemotions_available = False
    if ENABLE_GOEMOTIONS:
        print("\nComputing Signal 2: GoEmotions-based linguistic intimacy...")
        intimacy_score, goemotions_available, emotion_cache = (
            compute_goemotions_intimacy(summary_df, reply_edges, tweet_texts, has_full_text)
        )
    else:
        print("\nSignal 2: GoEmotions DISABLED (ENABLE_GOEMOTIONS=False)")
    signals_available["goemotions_intimacy"] = goemotions_available
    if not goemotions_available:
        print("  -> Intimacy signal = 0 (no effect on composite)")

    # --- Signal 3: Initiation balance ---
    print("\nComputing Signal 3: Initiation balance...")
    balance_score = compute_initiation_balance(summary_df)
    signals_available["initiation_balance"] = True
    print(f"  Balance scores: mean={balance_score.mean():.3f}, "
          f"median={balance_score.median():.3f}")

    # --- Signal 4: Temporal consistency ---
    print("\nComputing Signal 4: Temporal consistency...")
    consistency_score = compute_temporal_consistency(summary_df)
    signals_available["temporal_consistency"] = True
    print(f"  Consistency scores: mean={consistency_score.mean():.3f}, "
          f"median={consistency_score.median():.3f}")

    # --- Composite score ---
    print("\nComputing composite scores...")
    composite = (
        raw_intensity
        * (1 + W_INTIMACY * intimacy_score)
        * (1 + W_BALANCE * balance_score)
        * (1 + W_CONSISTENCY * consistency_score)
    )

    # --- Build network DataFrame ---
    network_df = summary_df[["ego_PersonID", "alter_PersonID"]].copy()
    network_df["composite_score"] = composite.values
    network_df["interaction_intensity_score"] = intensity_score.values
    network_df["intimacy_score"] = intimacy_score.values
    network_df["balance_score"] = balance_score.values
    network_df["consistency_score"] = consistency_score.values
    network_df["reply_count_total"] = reply_total.values
    network_df["mention_count_total"] = mention_total.values
    network_df["retweet_count_total"] = retweet_total.values
    network_df["reciprocal"] = summary_df["reciprocal"].values
    network_df["phases_interacted"] = summary_df["phases_interacted"].values

    # --- Rank and assign layers per ego ---
    print("Assigning Dunbar layers...")

    def rank_to_layer(rank):
        for layer_name, low, high in LAYER_BOUNDARIES:
            if high is None:
                if rank >= low:
                    return layer_name
            elif low <= rank <= high:
                return layer_name
        return "beyond_150"

    network_df["layer_rank"] = (
        network_df.groupby("ego_PersonID")["composite_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    network_df["dunbar_layer"] = network_df["layer_rank"].apply(rank_to_layer)

    col_order = [
        "ego_PersonID", "alter_PersonID",
        "dunbar_layer", "layer_rank",
        "composite_score",
        "interaction_intensity_score",
        "intimacy_score",
        "balance_score", "consistency_score",
        "reply_count_total", "mention_count_total", "retweet_count_total",
        "reciprocal", "phases_interacted",
    ]
    network_df = network_df[col_order]

    return network_df, signals_available


# =============================================================================
# NETWORK SUMMARY
# =============================================================================

def build_network_summary(network_df):
    """Build one-row-per-ego summary with layer counts and quality flags."""
    layer_names = ["inner_5", "middle_15", "outer_50", "outer_150", "beyond_150"]

    # Vectorized layer counts
    layer_dummies = pd.get_dummies(network_df["dunbar_layer"])
    layer_dummies["ego_PersonID"] = network_df["ego_PersonID"].values
    layer_counts_df = layer_dummies.groupby("ego_PersonID").sum()
    for layer in layer_names:
        if layer not in layer_counts_df.columns:
            layer_counts_df[layer] = 0
    layer_counts_df = layer_counts_df.rename(
        columns={l: f"n_{l}" for l in layer_names}
    )
    layer_counts_df["total_alters"] = layer_counts_df[
        [f"n_{l}" for l in layer_names]
    ].sum(axis=1)

    # Per-layer stats
    inner_stats = (
        network_df[network_df["dunbar_layer"] == "inner_5"]
        .groupby("ego_PersonID")
        .agg(
            mean_composite_inner_5=("composite_score", "mean"),
            pct_reciprocal_inner_5=("reciprocal", "mean"),
        )
    )
    inner_stats["pct_reciprocal_inner_5"] *= 100

    outer_stats = (
        network_df[network_df["dunbar_layer"] == "outer_150"]
        .groupby("ego_PersonID")
        .agg(
            mean_composite_outer_150=("composite_score", "mean"),
            pct_reciprocal_outer_150=("reciprocal", "mean"),
        )
    )
    outer_stats["pct_reciprocal_outer_150"] *= 100

    summary_df = layer_counts_df.join(inner_stats, how="left").join(
        outer_stats, how="left"
    )
    summary_df = summary_df.reset_index()

    # Minimum viable network flags (vectorised)
    checks = {
        "n_inner_5": MIN_INNER,
        "n_middle_15": MIN_MIDDLE,
        "n_outer_50": MIN_OUTER_50,
        "n_outer_150": MIN_OUTER_150,
        "total_alters": MIN_TOTAL_ALTERS,
    }
    note_parts = []
    for col, threshold in checks.items():
        if col in summary_df.columns:
            mask = summary_df[col] < threshold
            note = np.where(
                mask,
                col + "=" + summary_df[col].astype(str) + f" < {threshold}",
                "",
            )
            note_parts.append(pd.Series(note, index=summary_df.index))

    if note_parts:
        combined_notes = note_parts[0]
        for part in note_parts[1:]:
            combined_notes = combined_notes.where(
                combined_notes == "", combined_notes + "; "
            ) + part
        summary_df["network_quality_notes"] = combined_notes.str.strip("; ")
    else:
        summary_df["network_quality_notes"] = ""

    summary_df["meets_minimum_network"] = summary_df["network_quality_notes"] == ""

    col_order = [
        "ego_PersonID", "total_alters",
    ] + [f"n_{l}" for l in layer_names] + [
        "mean_composite_inner_5", "mean_composite_outer_150",
        "pct_reciprocal_inner_5", "pct_reciprocal_outer_150",
        "meets_minimum_network", "network_quality_notes",
    ]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    return summary_df


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary_report(network_df, summary_df, signals_available):
    """Print a human-readable summary of Dunbar layer assignment results."""
    n_egos = len(summary_df)
    n_pairs = len(network_df)

    print("\n" + "=" * 70)
    print("MODULE 4: DUNBAR LAYER ASSIGNMENT -- SUMMARY REPORT")
    print("=" * 70)

    print(f"\n--- Scoring signals ---")
    for signal, avail in signals_available.items():
        status = "ACTIVE" if avail else "INACTIVE (data not available)"
        print(f"  {signal:25s}: {status}")

    print(f"\n--- Weight settings ---")
    print(f"  W_REPLY={W_REPLY}, W_MENTION={W_MENTION}, W_RETWEET={W_RETWEET}")
    print(f"  RECIPROCITY_BONUS={RECIPROCITY_BONUS}")
    print(f"  W_INTIMACY={W_INTIMACY}, W_BALANCE={W_BALANCE}, "
          f"W_CONSISTENCY={W_CONSISTENCY}")

    if n_pairs == 0:
        print(f"\nNo ego-alter pairs to assign. Check Module 3 output.")
        print("=" * 70)
        return

    print(f"\n--- Network sizes ---")
    print(f"  Total egos: {n_egos}")
    print(f"  Total ego-alter pairs: {n_pairs:,}")
    print(f"  Alters per ego: mean={summary_df['total_alters'].mean():.1f}, "
          f"median={summary_df['total_alters'].median():.1f}, "
          f"min={summary_df['total_alters'].min()}, "
          f"max={summary_df['total_alters'].max()}")

    layer_names = ["inner_5", "middle_15", "outer_50", "outer_150", "beyond_150"]
    print(f"\n--- Mean alters per layer ---")
    for layer in layer_names:
        col = f"n_{layer}"
        mean_n = summary_df[col].mean()
        max_possible = {"inner_5": 5, "middle_15": 10, "outer_50": 35,
                        "outer_150": 100, "beyond_150": "inf"}[layer]
        print(f"  {layer:15s}: mean={mean_n:.1f} (max possible: {max_possible})")

    print(f"\n--- Composite score gradient across layers ---")
    for layer in layer_names:
        layer_data = network_df[network_df["dunbar_layer"] == layer]
        if len(layer_data) > 0:
            print(f"  {layer:15s}: mean={layer_data['composite_score'].mean():.2f}, "
                  f"median={layer_data['composite_score'].median():.2f}, "
                  f"n={len(layer_data)}")
        else:
            print(f"  {layer:15s}: (no alters assigned)")

    print(f"\n--- Reciprocity gradient ---")
    for layer in layer_names:
        layer_data = network_df[network_df["dunbar_layer"] == layer]
        if len(layer_data) > 0:
            pct = layer_data["reciprocal"].mean() * 100
            print(f"  {layer:15s}: {pct:.1f}% reciprocal (n={len(layer_data)})")

    meets = summary_df["meets_minimum_network"].sum()
    fails = n_egos - meets
    print(f"\n--- Minimum viable network thresholds ---")
    print(f"  Meets minimum: {meets}/{n_egos} egos ({meets / n_egos * 100:.1f}%)")
    print(f"  Below minimum: {fails}/{n_egos} egos ({fails / n_egos * 100:.1f}%)")

    if fails > 0:
        print(f"\n  Egos below minimum thresholds:")
        below = summary_df[~summary_df["meets_minimum_network"]]
        for _, row in below.iterrows():
            print(f"    {int(row['ego_PersonID'])}: {row['network_quality_notes']}")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_df, edges_df = load_interaction_data(INPUT_DIR, INPUT_DATE)

    if len(summary_df) == 0:
        print("No ego-alter pairs found. Run Module 3 first.")
        return None, None

    network_df, signals_available = compute_composite_and_assign_layers(
        summary_df, edges_df, DATA_DIR
    )

    print("\nBuilding network summary...")
    network_summary_df = build_network_summary(network_df)

    print_summary_report(network_df, network_summary_df, signals_available)

    network_output = os.path.join(OUTPUT_DIR, f"ego_network_{RUN_DATE}.csv")
    summary_output = os.path.join(OUTPUT_DIR, f"network_summary_{RUN_DATE}.csv")

    network_df.to_csv(network_output, index=False)
    network_summary_df.to_csv(summary_output, index=False)

    print(f"\nOutputs saved:")
    print(f"  Ego network: {network_output} ({len(network_df):,} rows)")
    print(f"  Network summary: {summary_output} ({len(network_summary_df):,} rows)")
    print(f"\nNext step: Run Module 5 (Risk Activation) on {network_output}")

    return network_df, network_summary_df


if __name__ == "__main__":
    network_df, network_summary_df = main()
