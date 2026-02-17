# ============================================================
# CULTURAL HUMILITY ANALYSIS — TRACK 2
# Google Drive Excel → Bigram Counts → Mann-Kendall → Line Graphs
#
# Instructions:
#   1. Run this entire cell in Google Colab
#   2. Approve Google Drive mount when prompted
#   3. Three PNG graphs + one Excel stats file will download automatically
# ============================================================

# ── STEP 0: Mount Google Drive ───────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── IMPORTS ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import theilslopes, norm
from google.colab import files

# ── CONFIGURATION ────────────────────────────────────────────
FILE_PATH = (
    '/content/drive/MyDrive/'
    'Colab Cultural Humility Project 1.13.2026/'
    'cultural_humility_analysis_results 2.17.2026.JGJ.Human.in.loop.xlsx'
)
SHEET_NAME       = 'Articles'
CATEGORY_COLUMN  = 'Journal /Category'   # no space after the slash
YEAR_COLUMN      = 'Published'
TEXT_COLUMNS     = ['Title', 'Abstract']

DISCIPLINES_TO_KEEP = ['Nursing', 'Medicine', 'Public Health', 'Counseling & Related']

YEAR_START = 2015
YEAR_END   = 2025
year_range = list(range(YEAR_START, YEAR_END + 1))

# ── BIGRAM SETS (longest phrase first for clean deduplication) ─
BIGRAM_SETS = {
    'Humility Set':   ['cultural humility',   'culturally humble'],
    'Competence Set': ['cultural competence', 'cultural competency', 'culturally competent'],
    'Awareness Set':  ['cultural awareness',  'culturally aware'],
}

# ── DISCIPLINE COLORS ─────────────────────────────────────────
DISCIPLINE_COLORS = {
    'Nursing':              '#E63946',
    'Counseling & Related': '#457B9D',
    'Medicine':             '#1D3557',
    'Public Health':        '#2A9D8F',
}

# ── GRAPH CONFIGS ─────────────────────────────────────────────
graph_configs = [
    ('Humility Set',
     'Cultural Humility Mentions by Discipline (2015–2025)',
     'trend_graph_cultural_humility.png'),
    ('Competence Set',
     'Cultural Competence Mentions by Discipline (2015–2025)',
     'trend_graph_cultural_competence.png'),
    ('Awareness Set',
     'Cultural Awareness Mentions by Discipline (2015–2025)',
     'trend_graph_cultural_awareness.png'),
]

# ============================================================
# STEP 1 — LOAD & FILTER DATA
# ============================================================
print("=" * 55)
print("STEP 1: Loading data from Google Drive")
print("=" * 55)

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
print(f"  Loaded {len(df)} total rows, {len(df.columns)} columns")

# Keep only the 4 valid disciplines (Other/Unknown already recategorized by user)
df = df[df[CATEGORY_COLUMN].isin(DISCIPLINES_TO_KEEP)].copy()
print(f"  After discipline filter: {len(df)} articles")
print(f"\n  Discipline breakdown:")
for disc, count in df[CATEGORY_COLUMN].value_counts().items():
    print(f"    {disc}: {count}")

# Parse year from the Published column
df['Year'] = pd.to_numeric(df[YEAR_COLUMN].astype(str).str[:4], errors='coerce')
df = df[df['Year'].between(YEAR_START, YEAR_END)].copy()
df['Year'] = df['Year'].astype(int)
print(f"\n  After year filter ({YEAR_START}–{YEAR_END}): {len(df)} articles")

# Combine Title + Abstract for bigram searching
df['combined_text'] = (
    df[TEXT_COLUMNS]
    .fillna('')
    .agg(' '.join, axis=1)
    .str.lower()
)

# ============================================================
# HELPER — Count bigram set mentions (longest-match-first)
# ============================================================
def count_bigram_set(text, phrases):
    """
    Count total mentions of all phrases in a text string.
    Processes longest phrases first to avoid double-counting
    (e.g. 'cultural competence' removed before 'competence').
    """
    phrases_sorted = sorted(phrases, key=len, reverse=True)
    working = text
    total = 0
    for phrase in phrases_sorted:
        count = working.count(phrase)
        total += count
        working = working.replace(phrase, ' ')
    return total

# ============================================================
# STEP 2 — COUNT BIGRAMS PER ARTICLE
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: Counting bigrams per article")
print("=" * 55)

for set_name, phrases in BIGRAM_SETS.items():
    df[set_name] = df['combined_text'].apply(
        lambda t: count_bigram_set(t, phrases)
    )
    total = int(df[set_name].sum())
    articles_with_hits = int((df[set_name] > 0).sum())
    print(f"  {set_name}: {total} total mentions across {articles_with_hits} articles")

# ============================================================
# MANN-KENDALL TEST IMPLEMENTATION
# ============================================================
def mann_kendall_test(x):
    """
    Non-parametric Mann-Kendall trend test.
    Returns: (tau, p_value, trend_label)
    """
    n = len(x)
    if n < 3:
        return np.nan, np.nan, 'insufficient data'

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    var_s = n * (n - 1) * (2 * n + 5) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2 * (1 - norm.cdf(abs(z)))
    tau     = s / (0.5 * n * (n - 1))

    if p_value < 0.05:
        trend = 'increasing' if s > 0 else 'decreasing'
    else:
        trend = 'no significant trend'

    return tau, p_value, trend

# ============================================================
# STEP 3 — BUILD YEARLY PIVOT TABLES + STATISTICS
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: Yearly counts + Mann-Kendall statistics")
print("=" * 55)

all_stats   = []
yearly_data = {}   # set_name → DataFrame indexed by year, columns = disciplines

for set_name in BIGRAM_SETS:
    set_df = df[['Year', CATEGORY_COLUMN, set_name]].copy()
    set_df.columns = ['Year', 'Discipline', 'Mentions']

    # Sum mentions per (Year, Discipline)
    grouped = (
        set_df
        .groupby(['Year', 'Discipline'])['Mentions']
        .sum()
        .reset_index()
    )

    # Pivot: rows = years, columns = disciplines
    # .reindex() ensures ALL 4 disciplines always appear as columns
    pivot = (
        grouped
        .pivot_table(index='Year', columns='Discipline',
                     values='Mentions', fill_value=0)
        .reindex(index=year_range, fill_value=0)
        .reindex(columns=DISCIPLINES_TO_KEEP, fill_value=0)
    )

    yearly_data[set_name] = pivot

    print(f"\n  [{set_name}]")
    print(f"  {'Discipline':<25} {'Total':>7} {'Tau':>8} {'p-value':>9} {'Trend'}")
    print(f"  {'-'*70}")

    for discipline in DISCIPLINES_TO_KEEP:
        series = pivot[discipline].values
        tau, p_val, trend = mann_kendall_test(series)

        # Theil-Sen slope (robust linear trend estimator)
        if len(series) >= 2 and np.std(series) > 0:
            ts = theilslopes(series, year_range)
            slope     = ts.slope
            intercept = ts.intercept
        else:
            slope     = 0.0
            intercept = float(series[0]) if len(series) > 0 else 0.0

        tau_str = f"{tau:.4f}"   if not np.isnan(tau)   else 'N/A'
        p_str   = f"{p_val:.4f}" if not np.isnan(p_val) else 'N/A'
        total   = int(pivot[discipline].sum())

        print(f"  {discipline:<25} {total:>7} {tau_str:>8} {p_str:>9}  {trend}")

        all_stats.append({
            'Term Set':          set_name,
            'Discipline':        discipline,
            'Total Mentions':    total,
            'Mann-Kendall Tau':  round(tau, 4)   if not np.isnan(tau)   else 'N/A',
            'p-value':           round(p_val, 4) if not np.isnan(p_val) else 'N/A',
            'Trend':             trend,
            'Theil-Sen Slope':   round(slope, 4),
            'Theil-Sen Intercept': round(intercept, 4),
        })

stats_df = pd.DataFrame(all_stats)

# ============================================================
# STEP 4 — SAVE STATISTICS TO EXCEL
# ============================================================
stats_file = 'mann_kendall_results.xlsx'
stats_df.to_excel(stats_file, index=False)
print(f"\n  ✓ Saved statistics to {stats_file}")

# ============================================================
# STEP 5 — DRAW THREE LINE GRAPHS (one per cultural term set)
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: Drawing graphs")
print("=" * 55)

for set_name, title, filename in graph_configs:
    pivot = yearly_data[set_name]

    fig, ax = plt.subplots(figsize=(13, 7))

    for discipline in DISCIPLINES_TO_KEEP:
        color  = DISCIPLINE_COLORS.get(discipline, '#333333')
        values = pivot[discipline].values
        ax.plot(
            year_range, values,
            marker='o', linewidth=2.5, markersize=7,
            label=discipline, color=color
        )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=16)
    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Number of Mentions', fontsize=13)
    ax.set_xticks(year_range)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: str(int(x)))
    )
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(
        title='Discipline', fontsize=11, title_fontsize=12,
        loc='upper left', framealpha=0.9
    )
    ax.grid(True, linestyle='--', alpha=0.45)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Saved {filename}")

# ============================================================
# STEP 6 — DOWNLOAD FILES INDIVIDUALLY (no ZIP)
# ============================================================
print("\n" + "=" * 55)
print("STEP 6: Downloading files")
print("=" * 55)

files.download(stats_file)
print(f"  ✓ Downloading {stats_file}")

for _, _, filename in graph_configs:
    files.download(filename)
    print(f"  ✓ Downloading {filename}")

print("\nAll done! Four files downloaded:")
print(f"  1. {stats_file}")
for _, _, filename in graph_configs:
    print(f"  2–4. {filename}")
