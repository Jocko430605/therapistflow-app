# ============================================================
# CULTURAL HUMILITY ANALYSIS — TRACK 2
# Google Drive Excel → Bigram Counts → Mann-Kendall → Line Graphs
#
# OUTPUT (5 files downloaded individually):
#   1. mann_kendall_results.xlsx         — stats (overall + per-discipline)
#   2. trend_graph_combined.png          — all 3 term sets, all disciplines
#   3. trend_graph_cultural_humility.png — Humility Set, 4 discipline lines
#   4. trend_graph_cultural_competence.png
#   5. trend_graph_cultural_awareness.png
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

# ── COLORS ───────────────────────────────────────────────────
DISCIPLINE_COLORS = {
    'Nursing':              '#E63946',
    'Counseling & Related': '#457B9D',
    'Medicine':             '#1D3557',
    'Public Health':        '#2A9D8F',
}

TERM_SET_COLORS = {
    'Humility Set':   '#1f77b4',   # blue
    'Competence Set': '#c94b7d',   # pink  (matches Cass's graph)
    'Awareness Set':  '#ff7f0e',   # orange
}

# ── GRAPH CONFIGS (per-discipline graphs) ─────────────────────
per_disc_graphs = [
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
print("=" * 60)
print("STEP 1: Loading data from Google Drive")
print("=" * 60)

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
print(f"  Loaded {len(df)} total rows, {len(df.columns)} columns")

print(f"\n  Raw discipline distribution:")
for disc, count in df[CATEGORY_COLUMN].value_counts().items():
    print(f"    {disc}: {count}")

# Keep only the 4 valid disciplines
df = df[df[CATEGORY_COLUMN].isin(DISCIPLINES_TO_KEEP)].copy()
print(f"\n  After discipline filter: {len(df)} articles")

df['Year'] = pd.to_numeric(df[YEAR_COLUMN].astype(str).str[:4], errors='coerce')
df = df[df['Year'].between(YEAR_START, YEAR_END)].copy()
df['Year'] = df['Year'].astype(int)
print(f"  After year filter ({YEAR_START}–{YEAR_END}): {len(df)} articles")

print(f"\n  Final discipline breakdown:")
for disc, count in df[CATEGORY_COLUMN].value_counts().items():
    print(f"    {disc}: {count}")

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
print("\n" + "=" * 60)
print("STEP 2: Counting bigrams per article")
print("=" * 60)

for set_name, phrases in BIGRAM_SETS.items():
    df[set_name] = df['combined_text'].apply(
        lambda t: count_bigram_set(t, phrases)
    )
    total = int(df[set_name].sum())
    articles_with_hits = int((df[set_name] > 0).sum())
    print(f"  {set_name}: {total} total mentions across {articles_with_hits} articles")

# ============================================================
# MANN-KENDALL TEST
# ============================================================
def mann_kendall_test(x):
    n = len(x)
    if n < 3:
        return np.nan, np.nan, 'insufficient data'
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:   s += 1
            elif diff < 0: s -= 1
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:   z = (s - 1) / np.sqrt(var_s)
    elif s < 0: z = (s + 1) / np.sqrt(var_s)
    else:       z = 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    tau     = s / (0.5 * n * (n - 1))
    trend   = ('increasing' if s > 0 else 'decreasing') if p_value < 0.05 else 'no significant trend'
    return tau, p_value, trend

# ============================================================
# STEP 3 — BUILD YEARLY PIVOT TABLES + STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Yearly counts + Mann-Kendall statistics")
print("=" * 60)

all_stats       = []
yearly_data     = {}   # set_name → pivot (year × discipline)
combined_yearly = {}   # set_name → Series (total across all disciplines by year)

for set_name in BIGRAM_SETS:
    set_df = df[['Year', CATEGORY_COLUMN, set_name]].copy()
    set_df.columns = ['Year', 'Discipline', 'Mentions']

    grouped = (
        set_df
        .groupby(['Year', 'Discipline'])['Mentions']
        .sum()
        .reset_index()
    )

    # Pivot: rows = years, columns = disciplines
    # .reindex() guarantees all 4 discipline columns always exist
    pivot = (
        grouped
        .pivot_table(index='Year', columns='Discipline',
                     values='Mentions', fill_value=0)
        .reindex(index=year_range, fill_value=0)
        .reindex(columns=DISCIPLINES_TO_KEEP, fill_value=0)
    )

    yearly_data[set_name]     = pivot
    combined_yearly[set_name] = pivot.sum(axis=1)   # all disciplines combined

    # ── Overall trend (all disciplines combined) ─────────────
    series_all = combined_yearly[set_name].values
    tau_all, p_all, trend_all = mann_kendall_test(series_all)
    if len(series_all) >= 2 and np.std(series_all) > 0:
        ts_all    = theilslopes(series_all, year_range)
        slope_all = ts_all.slope
    else:
        slope_all = 0.0

    total_all = int(pivot.values.sum())
    print(f"\n  [{set_name}]  ALL DISCIPLINES COMBINED")
    print(f"    Total mentions : {total_all}")
    print(f"    Trend          : {trend_all}")
    print(f"    Theil-Sen slope: {slope_all:.3f} mentions/year")
    print(f"    p-value        : {p_all:.4f}" if not np.isnan(p_all) else "    p-value: N/A")
    print(f"    Kendall's Tau  : {tau_all:.4f}" if not np.isnan(tau_all) else "    Tau: N/A")

    all_stats.append({
        'Term Set':              set_name,
        'Discipline':            'ALL (Combined)',
        'Total Mentions':        total_all,
        'Mann-Kendall Tau':      round(tau_all, 4)   if not np.isnan(tau_all) else 'N/A',
        'p-value':               round(p_all, 4)     if not np.isnan(p_all)   else 'N/A',
        'Trend':                 trend_all,
        'Theil-Sen Slope':       round(slope_all, 4),
        'Theil-Sen Intercept':   'N/A',
    })

    # ── Per-discipline trends ─────────────────────────────────
    print(f"\n  [{set_name}]  BY DISCIPLINE")
    print(f"  {'Discipline':<25} {'Total':>7} {'Tau':>8} {'p-value':>9} {'Trend'}")
    print(f"  {'-'*70}")

    for discipline in DISCIPLINES_TO_KEEP:
        series = pivot[discipline].values
        tau, p_val, trend = mann_kendall_test(series)
        if len(series) >= 2 and np.std(series) > 0:
            ts        = theilslopes(series, year_range)
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
            'Term Set':            set_name,
            'Discipline':          discipline,
            'Total Mentions':      total,
            'Mann-Kendall Tau':    round(tau, 4)   if not np.isnan(tau)   else 'N/A',
            'p-value':             round(p_val, 4) if not np.isnan(p_val) else 'N/A',
            'Trend':               trend,
            'Theil-Sen Slope':     round(slope, 4),
            'Theil-Sen Intercept': round(intercept, 4),
        })

stats_df = pd.DataFrame(all_stats)

# ── Summary table ─────────────────────────────────────────────
print("\n\n  OVERALL SUMMARY TABLE")
overall = stats_df[stats_df['Discipline'] == 'ALL (Combined)'][
    ['Term Set', 'Total Mentions', 'Trend', 'Theil-Sen Slope', 'p-value', 'Mann-Kendall Tau']
]
print(overall.to_string(index=False))

# ============================================================
# STEP 4 — SAVE STATISTICS TO EXCEL (two sheets)
# ============================================================
stats_file = 'mann_kendall_results.xlsx'
with pd.ExcelWriter(stats_file, engine='openpyxl') as writer:
    stats_df[stats_df['Discipline'] == 'ALL (Combined)'].to_excel(
        writer, sheet_name='Overall Trends', index=False)
    stats_df[stats_df['Discipline'] != 'ALL (Combined)'].to_excel(
        writer, sheet_name='By Discipline', index=False)
    stats_df.to_excel(writer, sheet_name='All Stats', index=False)
print(f"\n  ✓ Saved {stats_file}")

# ============================================================
# HELPER — Add data point labels to a plotted line
# ============================================================
def add_data_labels(ax, x_vals, y_vals, color, fontsize=8.5):
    for x, y in zip(x_vals, y_vals):
        if y > 0:
            ax.annotate(
                str(int(y)),
                xy=(x, y),
                xytext=(0, 8),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=fontsize, color=color,
                fontweight='bold'
            )

# ============================================================
# STEP 5 — GRAPH 1: COMBINED (all 3 term sets, all disciplines)
#          This replicates Cass's graph exactly.
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Graph 1 — Combined (all disciplines, 3 term sets)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 8))

for set_name in BIGRAM_SETS:
    color  = TERM_SET_COLORS[set_name]
    values = combined_yearly[set_name].values
    ax.plot(
        year_range, values,
        marker='o', linewidth=2.5, markersize=7,
        label=set_name, color=color
    )
    add_data_labels(ax, year_range, values, color, fontsize=9)

ax.set_title(
    'Trend of Cultural Term Mentions Over Time\n(All Disciplines Combined)',
    fontsize=16, fontweight='bold', pad=15
)
ax.set_xlabel('Year', fontsize=13)
ax.set_ylabel('Total Mentions', fontsize=13)
ax.set_xticks(year_range)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

combined_file = 'trend_graph_combined_all_disciplines.png'
plt.savefig(combined_file, dpi=150, bbox_inches='tight')
plt.show()
print(f"  ✓ Saved {combined_file}")

# ============================================================
# STEP 6 — GRAPHS 2–4: PER TERM SET, 4 DISCIPLINE LINES
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Graphs 2–4 — Per term set, by discipline")
print("=" * 60)

for set_name, title, filename in per_disc_graphs:
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
        add_data_labels(ax, year_range, values, color, fontsize=8)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=16)
    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Number of Mentions', fontsize=13)
    ax.set_xticks(year_range)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
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
# STEP 7 — DOWNLOAD ALL 5 FILES INDIVIDUALLY (no ZIP)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Downloading files")
print("=" * 60)

all_downloads = (
    [stats_file, combined_file]
    + [fname for _, _, fname in per_disc_graphs]
)

for fname in all_downloads:
    files.download(fname)
    print(f"  ✓ {fname}")

print(f"\nAll done! {len(all_downloads)} files downloaded:")
for i, fname in enumerate(all_downloads, 1):
    print(f"  {i}. {fname}")
