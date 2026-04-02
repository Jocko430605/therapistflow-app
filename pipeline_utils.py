"""
Pipeline Utilities: Shared constants and helpers
==================================================
Pipeline: COVID-19 Ego-Network Behavioral Activation Study
Version: 1.0
Authors: Ben (Kaiser Permanente) / Dr. Christian Lopez (Lafayette College)

Purpose:
    Shared phase definitions, file discovery, and date/phase parsing used
    across all pipeline modules.  Centralised here to prevent drift between
    modules (e.g. a phase date changed in one but not another).
"""

import os
import glob
import pandas as pd
import numpy as np

# =============================================================================
# PHASE DEFINITIONS
# Anchored to WA DOH genomic sequencing variant doubling rates
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
    Output: pd.Series of tz-aware (UTC) Timestamps; unparseable → NaT
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
    Output: pd.Series of phase key strings ('phase_1_novel', ..., 'outside_phases', 'unknown')
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
