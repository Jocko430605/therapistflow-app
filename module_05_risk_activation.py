"""
Module 5: Risk Signal and Behavioral Activation Measurement
=============================================================
Pipeline: COVID-19 Ego-Network Behavioral Activation Study
Version: 1.1
Authors: Ben (Kaiser Permanente) / Dr. Christian Lopez (Lafayette College)

Purpose:
    Core analytical module. Measures the relationship between COVID-19 risk
    exposure within each Dunbar layer of an ego's social network and the
    ego's behavioral activation along the Transtheoretical Model (TTM).

    For each ego at each biweekly window within each pandemic phase:
      A. Computes risk density per Dunbar layer (independent variable)
      B. Scores behavioral activation / TTM stage (dependent variable)
      C. Measures activation latency from risk signal to behavioral change

Inputs:
    {INPUT_DIR}/ego_network_{INPUT_DATE}.csv
    {DATA_DIR}/*_Summary_Details.csv

Outputs:
    {OUTPUT_DIR}/ego_phase_activation_{RUN_DATE}.csv
    {OUTPUT_DIR}/ego_latency_{RUN_DATE}.csv
    {OUTPUT_DIR}/ambient_risk_timeseries_{RUN_DATE}.csv

Usage:
    1. Set DATA_DIR, INPUT_DIR, INPUT_DATE, and OUTPUT_DIR.
    2. Run: python module_05_risk_activation.py

NOTE ON TTM SEED WORDS: The seed words below are grounded in Sacco et al.
(2023) validated TTM scale items for COVID-19 vaccination. The Preparation
and some Contemplation seeds skew toward vaccination-specific language.
If the study scope includes broader COVID behaviors (masking, distancing,
testing), the team should review and expand the seed dictionaries to cover
those behavioral domains. Word2Vec expansion will amplify whatever bias
exists in the seeds.
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pipeline_utils import (
    PHASES,
    DUNBAR_LAYERS,
    find_summary_files,
    parse_twitter_dates,
    assign_phases,
    series_contains_any,
    load_goemotions_classifier,
    load_emotion_cache,
    save_emotion_cache,
    classify_tweets_batch,
    get_dominant_emotion,
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
# BIWEEKLY WINDOW SIZE
# =============================================================================

WINDOW_DAYS = 14

# =============================================================================
# RISK DENSITY WEIGHTS
# =============================================================================

W_KEYWORD = 0.6
W_SENTIMENT = 0.2
W_EMOTION = 0.2

# =============================================================================
# RISK DENSITY THRESHOLD (for latency computation)
# =============================================================================

RISK_THRESHOLD = 0.1

# =============================================================================
# MINIMUM TWEETS PER WINDOW FOR TTM SCORING
# =============================================================================

MIN_TWEETS_FOR_TTM = 2

# =============================================================================
# COVID-RELATED KEYWORD DICTIONARY
# =============================================================================

COVID_KEYWORDS = [
    "covid", "coronavirus", "sars-cov", "pandemic",
    "quarantine", "lockdown", "social distancing",
    "mask", "n95", "ppe",
    "vaccine", "vaccination", "vaccinated", "pfizer", "moderna",
    "johnson & johnson", "j&j", "booster",
    "positive test", "tested positive", "pcr test", "rapid test",
    "ventilator", "icu", "hospitalized",
    "variant", "delta variant", "omicron", "omicron variant",
    "case count", "case rate", "death toll",
    "flatten the curve", "stay home", "work from home",
]

# =============================================================================
# THREAT KEYWORD DICTIONARY (Part A: risk density)
# =============================================================================

THREAT_KEYWORDS = [
    "tested positive", "have covid", "got covid", "has covid",
    "diagnosed with", "in the hospital", "hospitalized",
    "icu", "on a ventilator", "ventilator",
    "quarantining", "in quarantine",
    "my family member is sick", "family is sick",
    "lost someone to covid", "lost someone to", "passed away from covid",
    "covid symptoms", "feeling sick", "have symptoms",
    "waiting for test results", "waiting for results",
    "fever and cough", "can't breathe", "trouble breathing",
    "so sick", "really sick", "very ill",
]

# =============================================================================
# NEGATIVE SENTIMENT KEYWORDS (Part A: sentiment-based risk signal)
# =============================================================================

NEGATIVE_SENTIMENT_WORDS = [
    "scared", "afraid", "terrified", "anxious", "worried",
    "dying", "death", "dead", "died", "killed",
    "horrible", "terrible", "awful", "devastating", "heartbreaking",
    "overwhelmed", "exhausted", "burned out", "struggling",
    "angry", "furious", "outraged", "frustrated",
    "crying", "sobbing", "grief", "mourning", "loss",
    "hopeless", "helpless", "desperate", "nightmare",
    "emergency", "crisis", "catastrophe", "disaster",
    "suffering", "pain", "agony",
]

# =============================================================================
# GOEMOTIONS CONFIGURATION
# =============================================================================

ENABLE_GOEMOTIONS = True
GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"
GOEMOTIONS_BATCH_SIZE = 64
GOEMOTIONS_TOP_K = 5
GOEMOTIONS_DEVICE = "cpu"
EMOTION_CACHE_DIR = "./data/processed"

RISK_EMOTIONS = {"fear", "nervousness", "sadness", "grief", "remorse"}
NON_RISK_EMOTIONS = {"joy", "optimism", "excitement", "pride", "relief"}

# =============================================================================
# TTM STAGE KEYWORD DICTIONARIES (simple approach -- kept for comparison)
# =============================================================================

TTM_PRECONTEMPLATION = [
    "overblown", "hoax", "just the flu", "not worried",
    "media hype", "scam", "fake news", "plandemic",
    "not that bad", "blown out of proportion", "exaggerated",
    "more people die from", "survival rate",
    "i'm not afraid", "living in fear", "sheep",
]

TTM_CONTEMPLATION = [
    "should i", "is it safe", "wondering if", "anyone know",
    "how bad is", "worried about", "concerned about",
    "what do you think", "any advice", "has anyone",
    "what should we do", "is it worth", "trying to decide",
    "reading about", "looking at the data", "the numbers show",
    "i'm not sure", "on the fence",
]

TTM_PREPARATION = [
    "thinking about getting", "looking into", "need to find",
    "where can i get", "planning to", "going to try to",
    "making an appointment", "ordered masks",
    "signed up for", "scheduling", "booked my",
    "stocking up on", "preparing to", "getting ready to",
    "found a testing site", "found a vaccine site",
]

TTM_ACTION = [
    "got tested", "got vaccinated", "got my shot", "got my vaccine",
    "got the booster", "fully vaccinated",
    "staying home", "wearing a mask", "wearing my mask",
    "quarantining", "working from home", "cancelled plans",
    "social distancing", "keeping distance",
    "got my results", "tested negative",
    "just got my second dose", "just got boosted",
    "please get vaccinated", "go get tested", "wear your mask",
    "protect your family", "do your part",
]

# =============================================================================
# TTM SEED WORDS -- Derived from Sacco et al. (2023) scale items
# NOTE: Preparation and some Contemplation seeds are vaccination-specific.
# Review and expand for broader COVID behavioral domains if needed.
# =============================================================================

TTM_SEEDS_PRECONTEMPLATION = [
    "overblown", "hoax", "just the flu", "not worried", "don't need",
    "my immune system", "not that serious", "media hype", "scam",
    "plandemic", "no worse than", "refuse",
]

TTM_SEEDS_CONTEMPLATION = [
    "should i", "is it safe", "wondering", "not sure", "considering",
    "thinking about", "worried about", "concerned", "looking into",
    "pros and cons", "anyone know", "what do you think", "hesitant",
]

TTM_SEEDS_PREPARATION = [
    "going to get", "planning to", "making appointment", "signed up",
    "where can i", "scheduled", "looking for", "trying to find",
    "need to get", "eligible", "my turn",
]

TTM_SEEDS_ACTION = [
    "got vaccinated", "got my shot", "fully vaccinated", "got tested",
    "staying home", "wearing mask", "quarantining", "social distancing",
    "cancelled plans", "working from home", "booster", "second dose",
    "negative test",
]

# =============================================================================
# SEED-WORD EXPANSION CONFIGURATION
# =============================================================================

ENABLE_SEED_EXPANSION = True
EXPANSION_SIMILARITY_THRESHOLD = 0.6
EXPANSION_TOP_N = 20
EXPANDED_DICT_DIR = "./data/processed"

W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 5

EMOTION_TTM_REINFORCEMENT = {
    "fear": {"contemplation": 0.3},
    "nervousness": {"contemplation": 0.2},
    "sadness": {"contemplation": 0.2},
    "grief": {"contemplation": 0.1},
    "optimism": {"preparation": 0.2, "action": 0.2},
    "caring": {"preparation": 0.1, "action": 0.2},
    "approval": {"action": 0.1},
    "pride": {"action": 0.15},
    "joy": {"action": 0.1},
    "disgust": {"precontemplation": 0.2},
    "annoyance": {"precontemplation": 0.15},
    "anger": {"precontemplation": 0.1, "contemplation": 0.1},
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_biweekly_windows(phase_key, phase_def):
    """Generate biweekly (14-day) windows for a given phase."""
    start = pd.to_datetime(phase_def["start"])
    end = pd.to_datetime(phase_def["end"])
    windows = []
    window_num = 1
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=WINDOW_DAYS - 1), end)
        windows.append({
            "phase": phase_key,
            "window_number": window_num,
            "window_start": current,
            "window_end": window_end,
        })
        current = window_end + timedelta(days=1)
        window_num += 1
    return windows


def generate_all_windows():
    """Generate biweekly windows across all phases."""
    all_windows = []
    for phase_key, phase_def in PHASES.items():
        all_windows.extend(generate_biweekly_windows(phase_key, phase_def))
    return all_windows


def compute_emotion_risk_ratio(tweet_ids, emotion_cache):
    """
    Compute emotion-based risk ratio for a set of tweets.
    Returns: (n_risk - n_nonrisk) / n_total, clipped to [0, 1].
    """
    if not tweet_ids or not emotion_cache:
        return 0.0
    n_risk = 0
    n_nonrisk = 0
    n_total = 0
    for tid in tweet_ids:
        dominant = get_dominant_emotion(emotion_cache, tid)
        if dominant is None:
            continue
        n_total += 1
        if dominant in RISK_EMOTIONS:
            n_risk += 1
        elif dominant in NON_RISK_EMOTIONS:
            n_nonrisk += 1
    if n_total == 0:
        return 0.0
    raw = (n_risk - n_nonrisk) / n_total
    return max(0.0, min(1.0, raw))


def classify_all_tweets_goemotions(tweets_df, has_full_text, cache_dir):
    """Pre-classify all tweets with full_text using GoEmotions."""
    if not ENABLE_GOEMOTIONS or not has_full_text:
        return {}, False

    classifier, available = load_goemotions_classifier(
        GOEMOTIONS_MODEL, GOEMOTIONS_TOP_K, GOEMOTIONS_DEVICE, ENABLE_GOEMOTIONS
    )
    if not available:
        return {}, False

    cache = load_emotion_cache(cache_dir, GOEMOTIONS_TOP_K)

    valid_tweets = tweets_df[tweets_df.get("full_text", pd.Series()).notna()].copy()
    if len(valid_tweets) == 0:
        return cache, True

    tweet_ids = valid_tweets["Tweet_ID"].astype(int).tolist()
    texts = valid_tweets["full_text"].astype(str).tolist()

    uncached = [(tid, txt) for tid, txt in zip(tweet_ids, texts)
                if tid not in cache]
    if uncached:
        uc_ids, uc_texts = zip(*uncached)
        cache = classify_tweets_batch(
            classifier, uc_texts, uc_ids, cache, GOEMOTIONS_BATCH_SIZE
        )
        save_emotion_cache(cache, cache_dir, GOEMOTIONS_TOP_K)
    else:
        print(f"  All {len(tweet_ids):,} tweets found in cache")

    return cache, True


# =============================================================================
# SEED-WORD EXPANSION
# =============================================================================

def load_or_build_expanded_dictionaries(tweets_df, has_full_text):
    """
    Load expanded TTM dictionaries from JSON, or build them using Word2Vec.
    """
    seed_dicts = {
        "precontemplation": TTM_SEEDS_PRECONTEMPLATION,
        "contemplation": TTM_SEEDS_CONTEMPLATION,
        "preparation": TTM_SEEDS_PREPARATION,
        "action": TTM_SEEDS_ACTION,
    }
    expanded = {stage: list(seeds) for stage, seeds in seed_dicts.items()}

    if not ENABLE_SEED_EXPANSION:
        print("  Seed expansion DISABLED -- using seed words only")
        return expanded

    json_path = os.path.join(EXPANDED_DICT_DIR, "ttm_expanded_dictionaries.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            saved = json.load(f)
        for stage in expanded:
            if stage in saved and "full" in saved[stage]:
                expanded[stage] = saved[stage]["full"]
        total_expanded = sum(len(v) for v in expanded.values())
        total_seeds = sum(len(v) for v in seed_dicts.values())
        print(f"  Loaded expanded dictionaries from {json_path}")
        print(f"  Total terms: {total_expanded} "
              f"(seeds: {total_seeds}, "
              f"expansions: {total_expanded - total_seeds})")
        return expanded

    if not has_full_text:
        print("  [SKIPPED] No full_text -- cannot train Word2Vec for expansion")
        return expanded

    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("  [FALLBACK] gensim not installed -- using seed words only")
        return expanded

    texts = tweets_df["full_text"].dropna().astype(str)
    if len(texts) < 100:
        print(f"  [SKIPPED] Only {len(texts)} tweets with text -- too few for Word2Vec")
        return expanded

    print(f"  Training Word2Vec on {len(texts):,} tweets...")
    sentences = [t.lower().split() for t in texts]
    model = Word2Vec(
        sentences,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=4,
        seed=42,
    )
    vocab = set(model.wv.key_to_index.keys())
    print(f"  Word2Vec vocabulary: {len(vocab):,} terms")

    save_data = {}
    for stage, seeds in seed_dicts.items():
        expansion_set = set()
        for seed_phrase in seeds:
            words = seed_phrase.lower().split()
            for word in words:
                if word in vocab:
                    try:
                        similar = model.wv.most_similar(word, topn=EXPANSION_TOP_N)
                        for sim_word, sim_score in similar:
                            if sim_score >= EXPANSION_SIMILARITY_THRESHOLD:
                                expansion_set.add(sim_word)
                    except KeyError:
                        continue
        full_list = list(seeds) + sorted(expansion_set - set(
            w for s in seeds for w in s.lower().split()
        ))
        expanded[stage] = full_list
        save_data[stage] = {
            "seeds": seeds,
            "expanded": sorted(expansion_set),
            "full": full_list,
        }
        print(f"  {stage}: {len(seeds)} seeds + "
              f"{len(expansion_set)} expansions = {len(full_list)} total")

    os.makedirs(EXPANDED_DICT_DIR, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved expanded dictionaries to {json_path}")

    return expanded


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ego_network(input_dir, input_date):
    """Load Module 4 ego_network output."""
    filepath = os.path.join(input_dir, f"ego_network_{input_date}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Ego network file not found: {filepath}\n"
            f"Run Module 4 first if needed."
        )
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} ego-alter pairs from {filepath}")
    return df


def load_all_tweets(data_dir, all_network_ids):
    """
    Load raw Summary_Details CSVs iteratively, dropping any tweet not
    from our network (egos or alters). Prevents OOM on billion-tweet datasets.
    """
    files = find_summary_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No Summary_Details CSV files found in {data_dir}"
        )

    sample = pd.read_csv(files[0], nrows=2)
    has_full_text = "full_text" in sample.columns

    cols_needed = ["Tweet_ID", "PersonID", "Date Created", "RT"]
    if has_full_text:
        cols_needed.append("full_text")

    print(f"  full_text column: {'AVAILABLE' if has_full_text else 'NOT IN DATA'}")

    dfs = []
    for i, filepath in enumerate(files):
        try:
            available_cols = pd.read_csv(filepath, nrows=0).columns.tolist()
            read_cols = [c for c in cols_needed if c in available_cols]
            df = pd.read_csv(filepath, usecols=read_cols)
            df = df[df["PersonID"].isin(all_network_ids)]
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  Loaded {i + 1}/{len(files)} files")

    if not dfs:
        print("  No relevant tweets found for any network IDs.")
        return pd.DataFrame(columns=cols_needed), has_full_text

    combined = pd.concat(dfs, ignore_index=True)
    combined["datetime_utc"] = parse_twitter_dates(combined["Date Created"])
    combined["date_naive"] = combined["datetime_utc"].dt.tz_localize(None)

    print(f"  Total filtered network tweets: {len(combined):,}")
    return combined, has_full_text


# =============================================================================
# PART A: RISK DENSITY PER DUNBAR LAYER
# =============================================================================

def compute_layer_risk_density(tweets_df, ego_network_df, all_windows,
                               has_full_text, emotion_cache=None):
    """
    For each ego, at each biweekly window, measure risk density per Dunbar
    layer using three signals:
      1. Keyword-based threat detection (W_KEYWORD)
      2. Sentiment shift from lull baseline (W_SENTIMENT)
      3. GoEmotions emotion risk ratio (W_EMOTION)

    NOTE: Phase 3 (lull) baseline -- alters with no Phase 3 tweets default
    to baseline=0.0, meaning any negative sentiment registers as a shift.
    This is conservative but may create upward bias for alters who joined
    the dataset after Phase 3.
    """
    goemotions_active = bool(emotion_cache)

    # Build lookup: ego_id -> {layer -> set of alter_ids}
    ego_layers = {}
    for ego_id, group in ego_network_df.groupby("ego_PersonID"):
        ego_layers[ego_id] = {}
        for layer in DUNBAR_LAYERS:
            layer_alters = set(
                group[group["dunbar_layer"] == layer]["alter_PersonID"].values
            )
            ego_layers[ego_id][layer] = layer_alters

    all_ego_ids = set(ego_network_df["ego_PersonID"].unique())
    all_alter_ids = set(ego_network_df["alter_PersonID"].unique())

    alter_tweets = tweets_df[tweets_df["PersonID"].isin(all_alter_ids)].copy()
    print(f"  Alter tweets in dataset: {len(alter_tweets):,}")

    # Compute baseline negative sentiment rate from Phase 3 (lull)
    phase3_start = pd.to_datetime(PHASES["phase_3_lull"]["start"])
    phase3_end = pd.to_datetime(PHASES["phase_3_lull"]["end"])

    alter_baselines = {}
    if has_full_text:
        phase3_tweets = alter_tweets[
            (alter_tweets["date_naive"] >= phase3_start)
            & (alter_tweets["date_naive"] <= phase3_end)
        ].copy()
        if len(phase3_tweets) > 0:
            phase3_texts = phase3_tweets["full_text"].fillna("")
            phase3_tweets["_has_neg"] = series_contains_any(
                phase3_texts.astype(str), NEGATIVE_SENTIMENT_WORDS
            )
            baseline_agg = phase3_tweets.groupby("PersonID")["_has_neg"].mean()
            alter_baselines = baseline_agg.to_dict()

    # Process each window
    ego_window_risks = {ego_id: [] for ego_id in all_ego_ids}

    for window in all_windows:
        w_start = window["window_start"]
        w_end = window["window_end"]

        window_alter_tweets = alter_tweets[
            (alter_tweets["date_naive"] >= w_start)
            & (alter_tweets["date_naive"] <= w_end)
        ]

        for ego_id in all_ego_ids:
            risk_row = {
                "phase": window["phase"],
                "window_number": window["window_number"],
                "window_start": w_start.strftime("%Y-%m-%d"),
                "window_end": w_end.strftime("%Y-%m-%d"),
            }

            for layer in DUNBAR_LAYERS:
                alter_set = ego_layers[ego_id].get(layer, set())
                n_alters = len(alter_set)

                keyword_density = 0.0
                sentiment_shift = 0.0
                emotion_risk = 0.0

                if n_alters > 0 and has_full_text:
                    layer_tweets = window_alter_tweets[
                        window_alter_tweets["PersonID"].isin(alter_set)
                    ]
                    texts = layer_tweets["full_text"].dropna().astype(str)

                    if len(texts) > 0:
                        threat_count = series_contains_any(
                            texts, THREAT_KEYWORDS
                        ).sum()
                        keyword_density = threat_count / n_alters

                        neg_count = series_contains_any(
                            texts, NEGATIVE_SENTIMENT_WORDS
                        ).sum()
                        window_neg_rate = neg_count / len(texts)

                        baseline_rates = [
                            alter_baselines.get(a, 0.0) for a in alter_set
                        ]
                        mean_baseline = (
                            np.mean(baseline_rates) if baseline_rates else 0.0
                        )
                        sentiment_shift = max(0.0, window_neg_rate - mean_baseline)

                    if goemotions_active and len(texts) > 0:
                        layer_tweet_ids = (
                            layer_tweets["Tweet_ID"].dropna().astype(int).tolist()
                        )
                        if layer_tweet_ids:
                            emotion_risk = compute_emotion_risk_ratio(
                                layer_tweet_ids, emotion_cache
                            )

                layer_risk = (
                    W_KEYWORD * keyword_density
                    + W_SENTIMENT * sentiment_shift
                    + W_EMOTION * emotion_risk
                )

                risk_row[f"keyword_risk_{layer}"] = round(keyword_density, 4)
                risk_row[f"sentiment_risk_{layer}"] = round(sentiment_shift, 4)
                risk_row[f"emotion_risk_{layer}"] = round(emotion_risk, 4)
                risk_row[f"{layer}_risk"] = round(layer_risk, 4)

            ego_window_risks[ego_id].append(risk_row)

    return ego_window_risks


def compute_ambient_risk(tweets_df, all_windows, has_full_text,
                         emotion_cache=None):
    """
    Compute ambient COVID risk signal across ALL tweets in each biweekly window.
    """
    goemotions_active = bool(emotion_cache)
    rows = []
    for window in all_windows:
        w_start = window["window_start"]
        w_end = window["window_end"]

        window_tweets = tweets_df[
            (tweets_df["date_naive"] >= w_start)
            & (tweets_df["date_naive"] <= w_end)
        ]

        total_tweets = len(window_tweets)
        covid_tweets = 0
        pct_covid = 0.0
        mean_negative = 0.0
        keyword_threat_volume = 0.0
        ambient_emotion_risk = 0.0

        if total_tweets > 0 and has_full_text:
            texts = window_tweets["full_text"].dropna().astype(str)
            if len(texts) > 0:
                covid_count = series_contains_any(texts, COVID_KEYWORDS).sum()
                covid_tweets = int(covid_count)
                pct_covid = covid_count / len(texts)

                neg_count = series_contains_any(
                    texts, NEGATIVE_SENTIMENT_WORDS
                ).sum()
                mean_negative = neg_count / len(texts)

                threat_count = series_contains_any(texts, THREAT_KEYWORDS).sum()
                keyword_threat_volume = int(threat_count)

        if goemotions_active and total_tweets > 0 and has_full_text:
            ambient_tweet_ids = (
                window_tweets["Tweet_ID"].dropna().astype(int).tolist()
            )
            if ambient_tweet_ids:
                ambient_emotion_risk = compute_emotion_risk_ratio(
                    ambient_tweet_ids, emotion_cache
                )

        rows.append({
            "phase": window["phase"],
            "window_start": w_start.strftime("%Y-%m-%d"),
            "window_end": w_end.strftime("%Y-%m-%d"),
            "total_tweets": total_tweets,
            "covid_tweets": covid_tweets,
            "pct_covid": round(pct_covid, 4),
            "mean_negative_sentiment": round(mean_negative, 4),
            "keyword_threat_volume": keyword_threat_volume,
            "ambient_emotion_risk": round(ambient_emotion_risk, 4),
        })

    return pd.DataFrame(rows)


# =============================================================================
# PART B: EGO BEHAVIORAL ACTIVATION SCORING
# =============================================================================

def compute_ego_activation(tweets_df, ego_ids, all_windows, has_full_text,
                           emotion_cache=None, expanded_dicts=None):
    """
    For each ego at each biweekly window, score TTM stage.

    Two scoring approaches run in parallel:
      1. Simple keyword-based (original) -> kw_* columns
      2. Seed-expansion + GoEmotions reinforcement -> primary columns
    """
    goemotions_active = bool(emotion_cache)

    if expanded_dicts is None:
        expanded_dicts = {
            "precontemplation": TTM_SEEDS_PRECONTEMPLATION,
            "contemplation": TTM_SEEDS_CONTEMPLATION,
            "preparation": TTM_SEEDS_PREPARATION,
            "action": TTM_SEEDS_ACTION,
        }

    ego_tweets = tweets_df[tweets_df["PersonID"].isin(ego_ids)].copy()
    print(f"  Ego tweets in dataset: {len(ego_tweets):,}")

    ego_window_activation = {ego_id: [] for ego_id in ego_ids}

    for window in all_windows:
        w_start = window["window_start"]
        w_end = window["window_end"]

        window_ego_tweets = ego_tweets[
            (ego_tweets["date_naive"] >= w_start)
            & (ego_tweets["date_naive"] <= w_end)
        ]

        for ego_id in ego_ids:
            ego_window_tw = window_ego_tweets[
                window_ego_tweets["PersonID"] == ego_id
            ]
            ego_tweet_count = len(ego_window_tw)

            ego_covid_count = 0
            kw_pre = kw_con = kw_prep = kw_act = 0.0
            se_pre = se_con = se_prep = se_act = 0.0

            if ego_tweet_count > 0 and has_full_text:
                text_series = ego_window_tw["full_text"].dropna().astype(str)
                n_texts = len(text_series)

                if n_texts > 0:
                    ego_covid_count = int(
                        series_contains_any(text_series, COVID_KEYWORDS).sum()
                    )

                    # Simple keyword scoring (original)
                    kw_pre = int(series_contains_any(
                        text_series, TTM_PRECONTEMPLATION).sum()) / n_texts
                    kw_con = int(series_contains_any(
                        text_series, TTM_CONTEMPLATION).sum()) / n_texts
                    kw_prep = int(series_contains_any(
                        text_series, TTM_PREPARATION).sum()) / n_texts
                    kw_act = int(series_contains_any(
                        text_series, TTM_ACTION).sum()) / n_texts

                    # Seed-expansion scoring
                    se_pre = int(series_contains_any(
                        text_series,
                        expanded_dicts["precontemplation"]).sum()) / n_texts
                    se_con = int(series_contains_any(
                        text_series,
                        expanded_dicts["contemplation"]).sum()) / n_texts
                    se_prep = int(series_contains_any(
                        text_series,
                        expanded_dicts["preparation"]).sum()) / n_texts
                    se_act = int(series_contains_any(
                        text_series,
                        expanded_dicts["action"]).sum()) / n_texts

                    # GoEmotions reinforcement for seed-expansion
                    if goemotions_active:
                        tweet_ids = ego_window_tw["Tweet_ID"].dropna().astype(
                            int
                        ).tolist()
                        for tid in tweet_ids:
                            dominant = get_dominant_emotion(emotion_cache, tid)
                            if dominant and dominant in EMOTION_TTM_REINFORCEMENT:
                                boosts = EMOTION_TTM_REINFORCEMENT[dominant]
                                for stage, bonus in boosts.items():
                                    if stage == "precontemplation":
                                        se_pre += bonus / n_texts
                                    elif stage == "contemplation":
                                        se_con += bonus / n_texts
                                    elif stage == "preparation":
                                        se_prep += bonus / n_texts
                                    elif stage == "action":
                                        se_act += bonus / n_texts

            # Precontemplation default when no COVID tweets detected
            if has_full_text and ego_tweet_count >= MIN_TWEETS_FOR_TTM:
                if ego_covid_count == 0:
                    kw_pre = max(kw_pre, 0.5)
                    se_pre = max(se_pre, 0.5)

            # Assign TTM stage
            if ego_tweet_count < MIN_TWEETS_FOR_TTM:
                assigned_stage = "insufficient_data"
                continuous_score = np.nan
                kw_stage = "insufficient_data"
                kw_continuous = np.nan
            else:
                se_scores = {
                    "precontemplation": se_pre, "contemplation": se_con,
                    "preparation": se_prep, "action": se_act,
                }
                assigned_stage = max(se_scores, key=se_scores.get)
                se_total = se_pre + se_con + se_prep + se_act
                if se_total > 0:
                    continuous_score = (
                        0 * se_pre + 1 * se_con
                        + 2 * se_prep + 3 * se_act
                    ) / se_total
                else:
                    continuous_score = 0.0

                kw_scores = {
                    "precontemplation": kw_pre, "contemplation": kw_con,
                    "preparation": kw_prep, "action": kw_act,
                }
                kw_stage = max(kw_scores, key=kw_scores.get)
                kw_total = kw_pre + kw_con + kw_prep + kw_act
                if kw_total > 0:
                    kw_continuous = (
                        0 * kw_pre + 1 * kw_con
                        + 2 * kw_prep + 3 * kw_act
                    ) / kw_total
                else:
                    kw_continuous = 0.0

            def _round_or_nan(v):
                if isinstance(v, float) and np.isnan(v):
                    return np.nan
                return round(v, 4)

            ego_window_activation[ego_id].append({
                "ego_tweet_count": ego_tweet_count,
                "ego_covid_tweet_count": ego_covid_count,
                "precontemplation_score": round(se_pre, 4),
                "contemplation_score": round(se_con, 4),
                "preparation_score": round(se_prep, 4),
                "action_score": round(se_act, 4),
                "assigned_ttm_stage": assigned_stage,
                "continuous_activation_score": _round_or_nan(continuous_score),
                "kw_precontemplation_score": round(kw_pre, 4),
                "kw_contemplation_score": round(kw_con, 4),
                "kw_preparation_score": round(kw_prep, 4),
                "kw_action_score": round(kw_act, 4),
                "kw_assigned_ttm_stage": kw_stage,
                "kw_continuous_activation_score": _round_or_nan(kw_continuous),
            })

    return ego_window_activation


# =============================================================================
# COMBINE INTO ACTIVATION TABLE + COMPUTE CHANGE
# =============================================================================

def build_activation_table(ego_window_risks, ego_window_activation,
                           ambient_df, ego_ids, all_windows):
    """Combine risk density and activation scoring into a single table."""
    rows = []

    ambient_lookup = {}
    for _, row in ambient_df.iterrows():
        key = (row["phase"], row["window_start"])
        ambient_lookup[key] = row.get("pct_covid", 0.0)

    for ego_id in sorted(ego_ids):
        risk_list = ego_window_risks[ego_id]
        act_list = ego_window_activation[ego_id]
        prev_continuous = np.nan

        for i, (risk, act) in enumerate(zip(risk_list, act_list)):
            row = {"ego_PersonID": ego_id}
            row.update(risk)
            row.update(act)

            ambient_key = (risk["phase"], risk["window_start"])
            row["ambient_risk"] = round(ambient_lookup.get(ambient_key, 0.0), 4)

            current = act["continuous_activation_score"]
            if np.isnan(prev_continuous) or (
                isinstance(current, float) and np.isnan(current)
            ):
                row["activation_change"] = np.nan
            else:
                row["activation_change"] = round(current - prev_continuous, 4)
            prev_continuous = current

            rows.append(row)

    df = pd.DataFrame(rows)

    col_order = [
        "ego_PersonID", "phase", "window_start", "window_end", "window_number",
        "inner_5_risk", "middle_15_risk", "outer_50_risk", "outer_150_risk",
        "ambient_risk",
        "keyword_risk_inner_5", "sentiment_risk_inner_5", "emotion_risk_inner_5",
        "keyword_risk_middle_15", "sentiment_risk_middle_15", "emotion_risk_middle_15",
        "keyword_risk_outer_50", "sentiment_risk_outer_50", "emotion_risk_outer_50",
        "keyword_risk_outer_150", "sentiment_risk_outer_150", "emotion_risk_outer_150",
        "ego_tweet_count", "ego_covid_tweet_count",
        "precontemplation_score", "contemplation_score",
        "preparation_score", "action_score",
        "assigned_ttm_stage", "continuous_activation_score",
        "activation_change",
        "kw_precontemplation_score", "kw_contemplation_score",
        "kw_preparation_score", "kw_action_score",
        "kw_assigned_ttm_stage", "kw_continuous_activation_score",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    return df


# =============================================================================
# PART C: LATENCY MEASUREMENT
# =============================================================================

def compute_latency(activation_df):
    """
    For each ego in each phase, compute activation latency.
    """
    rows = []

    for (ego_id, phase), group in activation_df.groupby(
        ["ego_PersonID", "phase"]
    ):
        group = group.sort_values("window_number")

        first_risk = {}
        for layer in DUNBAR_LAYERS:
            col = f"{layer}_risk"
            if col in group.columns:
                risk_windows = group[group[col] >= RISK_THRESHOLD]
                if len(risk_windows) > 0:
                    first_risk[layer] = int(
                        risk_windows.iloc[0]["window_number"]
                    )
                else:
                    first_risk[layer] = np.nan
            else:
                first_risk[layer] = np.nan

        activated = group[
            (group["assigned_ttm_stage"] != "precontemplation")
            & (group["assigned_ttm_stage"] != "insufficient_data")
        ]
        if len(activated) > 0:
            first_activation = int(activated.iloc[0]["window_number"])
        else:
            first_activation = np.nan

        stage_order = {
            "precontemplation": 0, "contemplation": 1,
            "preparation": 2, "action": 3,
        }
        valid_stages = group[
            group["assigned_ttm_stage"] != "insufficient_data"
        ]["assigned_ttm_stage"]
        if len(valid_stages) > 0:
            max_stage = max(
                valid_stages, key=lambda s: stage_order.get(s, -1)
            )
        else:
            max_stage = "insufficient_data"

        latency = {}
        for layer in DUNBAR_LAYERS:
            fr = first_risk[layer]
            if not np.isnan(fr) and not np.isnan(first_activation):
                latency[layer] = first_activation - fr
            else:
                latency[layer] = np.nan

        row = {
            "ego_PersonID": ego_id,
            "phase": phase,
            "first_risk_window_inner_5": first_risk.get("inner_5", np.nan),
            "first_risk_window_middle_15": first_risk.get("middle_15", np.nan),
            "first_risk_window_outer_50": first_risk.get("outer_50", np.nan),
            "first_risk_window_outer_150": first_risk.get("outer_150", np.nan),
            "first_activation_window": first_activation,
            "latency_from_inner_5": latency.get("inner_5", np.nan),
            "latency_from_middle_15": latency.get("middle_15", np.nan),
            "latency_from_outer_50": latency.get("outer_50", np.nan),
            "latency_from_outer_150": latency.get("outer_150", np.nan),
            "max_activation_stage": max_stage,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary_report(activation_df, latency_df, ambient_df,
                         has_full_text, all_windows):
    """Print a human-readable summary."""
    n_egos = activation_df["ego_PersonID"].nunique()
    n_windows = len(all_windows)

    print("\n" + "=" * 70)
    print("MODULE 5: RISK SIGNAL & BEHAVIORAL ACTIVATION -- SUMMARY REPORT")
    print("=" * 70)

    print(f"\n--- Data availability ---")
    print(f"  full_text column: {'AVAILABLE' if has_full_text else 'NOT IN DATA'}")
    if not has_full_text:
        print(f"  IMPORTANT: full_text is required for keyword/sentiment/GoEmotions")
        print(f"  signals. All risk and TTM scores will be zero without it.")

    print(f"\n--- Time structure ---")
    print(f"  Total biweekly windows: {n_windows}")
    for phase_key, phase_def in PHASES.items():
        phase_windows = [w for w in all_windows if w["phase"] == phase_key]
        print(f"  {phase_def['label']:45s}: {len(phase_windows)} windows")

    print(f"\n--- Activation table ---")
    print(f"  Egos: {n_egos}")
    print(f"  Rows (ego x window): {len(activation_df):,}")

    print(f"\n--- Risk density by layer ---")
    for layer in DUNBAR_LAYERS:
        col = f"{layer}_risk"
        if col in activation_df.columns:
            mean_risk = activation_df[col].mean()
            max_risk = activation_df[col].max()
            nonzero = (activation_df[col] > 0).sum()
            print(f"  {layer:15s}: mean={mean_risk:.4f}, max={max_risk:.4f}, "
                  f"nonzero={nonzero}")

    print(f"\n--- TTM stage distribution ---")
    stage_counts = activation_df["assigned_ttm_stage"].value_counts()
    for stage in ["precontemplation", "contemplation", "preparation",
                  "action", "insufficient_data"]:
        count = stage_counts.get(stage, 0)
        pct = count / len(activation_df) * 100
        print(f"  {stage:25s}: {count:>6,} ({pct:5.1f}%)")

    valid_scores = activation_df["continuous_activation_score"].dropna()
    if len(valid_scores) > 0:
        print(f"\n--- Continuous activation score ---")
        print(f"  Mean: {valid_scores.mean():.2f}, "
              f"Median: {valid_scores.median():.2f}, "
              f"Std: {valid_scores.std():.2f}")

    if len(latency_df) > 0:
        print(f"\n--- Activation latency ---")
        for layer in DUNBAR_LAYERS:
            col = f"latency_from_{layer}"
            if col in latency_df.columns:
                valid = latency_df[col].dropna()
                if len(valid) > 0:
                    print(f"  {layer:15s}: mean={valid.mean():.1f}, "
                          f"median={valid.median():.1f}, n={len(valid)}")

        inner_lat = latency_df["latency_from_inner_5"].dropna()
        outer_lat = latency_df["latency_from_outer_150"].dropna()
        if len(inner_lat) > 0 and len(outer_lat) > 0:
            print(f"\n--- Core hypothesis (latency) ---")
            print(f"  Inner-5 mean: {inner_lat.mean():.1f} windows")
            print(f"  Outer-150 mean: {outer_lat.mean():.1f} windows")
            diff = outer_lat.mean() - inner_lat.mean()
            direction = "SUPPORTS" if diff > 0 else "DOES NOT SUPPORT"
            print(f"  Difference: {diff:.1f} windows -- {direction} hypothesis")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ego_network_df = load_ego_network(INPUT_DIR, INPUT_DATE)
    ego_ids = set(ego_network_df["ego_PersonID"].unique())
    print(f"Unique egos: {len(ego_ids)}")

    all_windows = generate_all_windows()
    print(f"Biweekly windows generated: {len(all_windows)}")

    print("\nLoading iteratively filtered raw tweet data...")
    all_network_ids = set(ego_network_df["alter_PersonID"].unique())
    all_network_ids.update(ego_ids)
    tweets_df, has_full_text = load_all_tweets(DATA_DIR, all_network_ids)

    emotion_cache = None
    if ENABLE_GOEMOTIONS and has_full_text:
        print("\nPre-classifying tweets with GoEmotions...")
        emotion_cache, ge_available = classify_all_tweets_goemotions(
            tweets_df, has_full_text, EMOTION_CACHE_DIR
        )
        if not ge_available:
            emotion_cache = None
    else:
        if not ENABLE_GOEMOTIONS:
            print("\nGoEmotions DISABLED")
        elif not has_full_text:
            print("\nGoEmotions skipped (no full_text)")

    print("\nBuilding TTM dictionaries...")
    expanded_dicts = load_or_build_expanded_dictionaries(
        tweets_df, has_full_text
    )

    print("\nPart A: Computing risk density per Dunbar layer...")
    ego_window_risks = compute_layer_risk_density(
        tweets_df, ego_network_df, all_windows, has_full_text,
        emotion_cache=emotion_cache,
    )

    print("\nComputing ambient risk timeseries...")
    ambient_df = compute_ambient_risk(
        tweets_df, all_windows, has_full_text,
        emotion_cache=emotion_cache,
    )

    print("\nPart B: Computing ego behavioral activation...")
    ego_window_activation = compute_ego_activation(
        tweets_df, ego_ids, all_windows, has_full_text,
        emotion_cache=emotion_cache,
        expanded_dicts=expanded_dicts,
    )

    print("\nBuilding activation table...")
    activation_df = build_activation_table(
        ego_window_risks, ego_window_activation, ambient_df,
        ego_ids, all_windows
    )

    print("\nPart C: Computing activation latency...")
    latency_df = compute_latency(activation_df)

    print_summary_report(activation_df, latency_df, ambient_df,
                         has_full_text, all_windows)

    activation_output = os.path.join(
        OUTPUT_DIR, f"ego_phase_activation_{RUN_DATE}.csv"
    )
    latency_output = os.path.join(
        OUTPUT_DIR, f"ego_latency_{RUN_DATE}.csv"
    )
    ambient_output = os.path.join(
        OUTPUT_DIR, f"ambient_risk_timeseries_{RUN_DATE}.csv"
    )

    activation_df.to_csv(activation_output, index=False)
    latency_df.to_csv(latency_output, index=False)
    ambient_df.to_csv(ambient_output, index=False)

    print(f"\nOutputs saved:")
    print(f"  Activation: {activation_output} ({len(activation_df):,} rows)")
    print(f"  Latency: {latency_output} ({len(latency_df):,} rows)")
    print(f"  Ambient risk: {ambient_output} ({len(ambient_df):,} rows)")
    print(f"\nNext step: Run Module 6 (Statistical Analysis)")

    return activation_df, latency_df, ambient_df


if __name__ == "__main__":
    activation_df, latency_df, ambient_df = main()
