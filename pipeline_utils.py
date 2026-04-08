"""
Pipeline Utilities: Shared constants and helpers
==================================================
Pipeline: COVID-19 Ego-Network Behavioral Activation Study
Version: 1.1
Authors: Ben (Kaiser Permanente) / Dr. Christian Lopez (Lafayette College)

Purpose:
    Shared phase definitions, file discovery, date/phase parsing, and
    GoEmotions helpers used across all pipeline modules.  Centralised here
    to prevent drift between modules (e.g. a phase date changed in one but
    not another).
"""

import os
import re
import glob
import pandas as pd
import numpy as np

# =============================================================================
# PHASE DEFINITIONS
# Anchored to WA DOH genomic sequencing variant doubling rates
#
# NOTE: Summer 2020 (Jun–Sep) is intentionally excluded — no dominant variant
# signature between initial wave subsidence and winter surge onset.
# =============================================================================

PHASES = {
    "phase_1_novel": {
        "label": "Novel Wave (Kirkland/Initial)",
        "start": "2020-02-19",
        "end": "2020-05-31",
    },
    "phase_2_winter_surge": {
        "label": "Winter Surge + Vaccine Arrival",
        "start": "2020-10-01",
        "end": "2021-02-28",
    },
    "phase_3_lull": {
        "label": "Lull / Control (Post-Vaccine Normalization)",
        "start": "2021-03-01",
        "end": "2021-06-14",
    },
    "phase_4_delta": {
        "label": "Delta",
        "start": "2021-06-15",
        "end": "2021-10-31",
    },
    "phase_5_omicron": {
        "label": "Omicron Onset",
        "start": "2021-11-01",
        "end": "2021-12-31",
    },
}

# Pre-compute datetime boundaries for vectorised phase assignment
_PHASE_BOUNDS = []
for phase_key, phase_def in PHASES.items():
    _PHASE_BOUNDS.append((
        phase_key,
        pd.to_datetime(phase_def["start"]),
        pd.to_datetime(phase_def["end"]),
    ))

# Also attach datetime objects to the PHASES dict for any per-row usage
for phase_key in PHASES:
    PHASES[phase_key]["start_dt"] = pd.to_datetime(PHASES[phase_key]["start"])
    PHASES[phase_key]["end_dt"] = pd.to_datetime(PHASES[phase_key]["end"])

# Dunbar layers used across modules
DUNBAR_LAYERS = ["inner_5", "middle_15", "outer_50", "outer_150"]

# Total phases in the study
TOTAL_PHASES = 5


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_summary_files(data_dir):
    """
    Find all Summary_Details CSV files in the data directory.
    Handles both naming conventions:
        YYYY_MM_DD_HH_Summary_Details.csv
        YYYY_MM_DD_HH_CT_Summary_Details.csv
    """
    patterns = [
        os.path.join(data_dir, "*_Summary_Details.csv"),
        os.path.join(data_dir, "*_CT_Summary_Details.csv"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    print(f"Found {len(files)} Summary_Details files in {data_dir}")
    return files


# =============================================================================
# VECTORISED DATE PARSING
# =============================================================================

def parse_twitter_dates(series):
    """
    Vectorised parse of Twitter API date strings.

    Input : pd.Series of strings like 'Wed Jan 22 18:00:00 +0000 2020'
    Output: pd.Series of tz-aware (UTC) Timestamps; unparseable -> NaT
    """
    return pd.to_datetime(
        series,
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )


# =============================================================================
# VECTORISED PHASE ASSIGNMENT
# =============================================================================

def assign_phases(dt_series):
    """
    Vectorised phase assignment for an entire datetime Series.

    Input : pd.Series of tz-aware (UTC) or tz-naive Timestamps
    Output: pd.Series of phase key strings ('phase_1_novel', ...,
            'outside_phases', 'unknown')
    """
    # Work with tz-naive for comparison against phase boundaries
    if hasattr(dt_series.dt, "tz") and dt_series.dt.tz is not None:
        dt_naive = dt_series.dt.tz_localize(None)
    else:
        dt_naive = dt_series

    result = pd.Series("outside_phases", index=dt_series.index)
    result[dt_series.isna()] = "unknown"

    for phase_key, start_dt, end_dt in _PHASE_BOUNDS:
        mask = (dt_naive >= start_dt) & (dt_naive <= end_dt)
        result[mask] = phase_key

    return result


# =============================================================================
# VECTORISED TEXT SEARCH
# =============================================================================

def build_keyword_pattern(keywords):
    """Build a compiled regex pattern from a keyword list for vectorised matching."""
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile("|".join(escaped), re.IGNORECASE)


def series_contains_any(text_series, keywords):
    """Vectorised check: returns boolean Series for whether each text contains any keyword."""
    pattern = build_keyword_pattern(keywords)
    return text_series.str.contains(pattern, na=False)


# =============================================================================
# GOEMOTIONS HELPERS — shared between Modules 4 and 5
# =============================================================================

_goemotions_classifier = None  # Module-level singleton


def load_goemotions_classifier(model_name, top_k, device, enable=True):
    """
    Attempt to load GoEmotions classifier pipeline.
    Returns (classifier, success_bool). Caches the classifier as a singleton
    so it's only loaded once per session.
    """
    global _goemotions_classifier
    if _goemotions_classifier is not None:
        return _goemotions_classifier, True
    if not enable:
        return None, False
    try:
        from transformers import pipeline as hf_pipeline
        _goemotions_classifier = hf_pipeline(
            "text-classification",
            model=model_name,
            top_k=top_k,
            device=device,
        )
        print(f"  GoEmotions model loaded: {model_name} (device={device})")
        return _goemotions_classifier, True
    except Exception as e:
        print(f"  [FALLBACK] GoEmotions not available "
              f"({type(e).__name__}: {e})")
        return None, False


def load_emotion_cache(cache_dir, top_k=5):
    """Load cached GoEmotions classifications from CSV."""
    cache_path = os.path.join(cache_dir, "goemotions_cache.csv")
    if not os.path.exists(cache_path):
        return {}
    df = pd.read_csv(cache_path)
    cache = {}
    for row in df.itertuples(index=False):
        tweet_id = int(row.tweet_id)
        emotions = []
        for i in range(1, top_k + 1):
            em = getattr(row, f"emotion_{i}", None)
            sc = getattr(row, f"score_{i}", None)
            if pd.notna(em) and pd.notna(sc):
                emotions.append({"label": str(em), "score": float(sc)})
        cache[tweet_id] = emotions
    print(f"  Loaded {len(cache):,} cached emotion classifications")
    return cache


def save_emotion_cache(cache, cache_dir, top_k=5):
    """Save GoEmotions classifications to CSV for cross-module reuse."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "goemotions_cache.csv")
    rows = []
    for tweet_id, emotions in cache.items():
        row = {"tweet_id": tweet_id}
        for i, em in enumerate(emotions[:top_k], 1):
            row[f"emotion_{i}"] = em["label"]
            row[f"score_{i}"] = em["score"]
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(cache_path, index=False)
        print(f"  Saved {len(rows):,} emotion classifications to {cache_path}")


def classify_tweets_batch(classifier, texts, tweet_ids, cache, batch_size=64):
    """
    Classify tweets in batches using GoEmotions. Uses and updates the cache
    to avoid re-classifying tweets.
    Returns the updated cache dict.
    """
    uncached = [(tid, txt) for tid, txt in zip(tweet_ids, texts)
                if tid not in cache]
    if not uncached:
        print(f"  All {len(tweet_ids):,} tweets found in cache")
        return cache

    uncached_ids, uncached_texts = zip(*uncached)
    # Truncate to RoBERTa's 512 token window
    uncached_texts = [t[:512] for t in uncached_texts]
    n = len(uncached_texts)
    print(f"  Classifying {n:,} uncached tweets in batches of {batch_size}...")

    for i in range(0, n, batch_size):
        batch_texts = list(uncached_texts[i:i + batch_size])
        batch_ids = uncached_ids[i:i + batch_size]
        results = classifier(batch_texts)
        for tid, result in zip(batch_ids, results):
            cache[tid] = result
        done = min(i + batch_size, n)
        if done % (batch_size * 10) == 0 or done == n:
            print(f"    Classified {done:,}/{n:,}")

    return cache


def get_dominant_emotion(emotion_cache, tweet_id):
    """Get the highest-probability emotion label for a tweet."""
    emotions = emotion_cache.get(int(tweet_id), [])
    if emotions:
        return max(emotions, key=lambda e: e["score"])["label"]
    return None
