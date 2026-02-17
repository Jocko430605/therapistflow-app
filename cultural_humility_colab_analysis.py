# ============================================================
# CULTURAL HUMILITY ANALYSIS — TRACK 2
# Google Drive Excel → Bigram Analysis → Mann-Kendall → Line Graphs
#
# KEY DESIGN:
#   - Every article is classified into EXACTLY ONE primary term set
#     (whichever of Humility / Competence / Awareness has the most
#     mentions in that article's Title + Abstract).
#   - The 3 per-discipline graphs therefore total = all articles.
#   - The combined graph shows total MENTION counts (like Cass's graph).
#   - NO discipline filter — all articles (including Other/Unknown) included.
#
# OUTPUT (5 files downloaded individually):
#   1. mann_kendall_results.xlsx
#   2. trend_graph_combined_all_disciplines.png
#   3. trend_graph_cultural_humility.png
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
SHEET_NAME      = 'Articles'
CATEGORY_COLUMN = 'Journal /Category'   # no space after the slash
YEAR_COLUMN     = 'Published'
TEXT_COLUMNS    = ['Title', 'Abstract']

YEAR_START = 2015
YEAR_END   = 2025
year_range = list(range(YEAR_START, YEAR_END + 1))

SEP = "=" * 80

# ── BIGRAM SETS ───────────────────────────────────────────────
# Longest phrase first within each set (deduplication)
BIGRAM_SETS = {
    'Humility Set':   ['cultural humility',   'culturally humble'],
    'Competence Set': ['cultural competence', 'cultural competency', 'culturally competent'],
    'Awareness Set':  ['cultural awareness',  'culturally aware'],
}

# Priority when an article ties across sets (or has no mentions)
SET_PRIORITY = ['Competence Set', 'Humility Set', 'Awareness Set']

# ── COLORS ───────────────────────────────────────────────────
DISCIPLINE_COLORS = {
    'Nursing':              '#E63946',
    'Counseling & Related': '#457B9D',
    'Medicine':             '#1D3557',
    'Public Health':        '#2A9D8F',
    'Other':                '#888888',
    'Unknown':              '#BBBBBB',
}

TERM_SET_COLORS = {
    'Humility Set':   '#1f77b4',   # blue
    'Competence Set': '#c94b7d',   # pink  (matches Cass's graph)
    'Awareness Set':  '#ff7f0e',   # orange
}

# ============================================================
# STEP 1 — LOAD ALL DATA  (no discipline filter)
# ============================================================
print(SEP)
print("LOADING DATA")
print(SEP)

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

df['Year'] = pd.to_numeric(df[YEAR_COLUMN].astype(str).str[:4], errors='coerce')
df = df[df['Year'].between(YEAR_START, YEAR_END)].copy()
df['Year'] = df['Year'].astype(int)

print(f"✓ Loaded {len(df)} articles")
print(f"✓ Year range: {YEAR_START} - {YEAR_END}")
print(f"\n  Discipline distribution:")
for disc, count in df[CATEGORY_COLUMN].value_counts().items():
    print(f"  {disc}: {count}")

# Ordered discipline list (keeps known disciplines in a sensible order)
_known_order = ['Nursing', 'Counseling & Related', 'Other', 'Unknown',
                'Public Health', 'Medicine']
all_disciplines = sorted(
    df[CATEGORY_COLUMN].dropna().unique().tolist(),
    key=lambda x: (_known_order.index(x) if x in _known_order else 99, x)
)

# Build combined text for bigram searching
df['combined_text'] = (
    df[TEXT_COLUMNS]
    .fillna('')
    .agg(' '.join, axis=1)
    .str.lower()
)

# ============================================================
# HELPERS
# ============================================================
def count_bigram_set(text, phrases):
    """Count total mentions of a set of phrases (longest-match-first)."""
    phrases_sorted = sorted(phrases, key=len, reverse=True)
    working = text
    total = 0
    for phrase in phrases_sorted:
        total  += working.count(phrase)
        working = working.replace(phrase, ' ')
    return total

def mann_kendall_test(x):
    """Non-parametric Mann-Kendall test. Returns (tau, p_value, trend)."""
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
    tau = s / (0.5 * n * (n - 1))
    trend = ('increasing' if s > 0 else 'decreasing') if p_value < 0.05 else 'no significant trend'
    return tau, p_value, trend

def fmt_trend(trend):
    if trend == 'increasing': return 'INCREASING **'
    if trend == 'decreasing': return 'DECREASING **'
    return '→ NO TREND'

def get_slope(series):
    if len(series) >= 2 and np.std(series) > 0:
        return theilslopes(series, year_range).slope
    return 0.0

def get_color(discipline, idx=0):
    if discipline in DISCIPLINE_COLORS:
        return DISCIPLINE_COLORS[discipline]
    return plt.cm.tab10.colors[idx % 10]

def add_data_labels(ax, x_vals, y_vals, color, fontsize=8.5):
    for x, y in zip(x_vals, y_vals):
        if y > 0:
            ax.annotate(
                str(int(y)), xy=(x, y), xytext=(0, 8),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=fontsize, color=color, fontweight='bold'
            )

# ============================================================
# STEP 2 — COUNT BIGRAMS (per article, per phrase and per set)
# ============================================================
print(f"\n{SEP}")
print("COUNTING BIGRAMS")
print(SEP)

# Per-phrase counts (printed like Cass's output)
print("\n✓ Bigram counts per article:")
for set_name, phrases in BIGRAM_SETS.items():
    for phrase in phrases:
        total    = int(df['combined_text'].str.count(phrase).sum())
        n_arts   = int((df['combined_text'].str.contains(phrase)).sum())
        print(f"  {phrase.title()}: {total:4d} mentions across {n_arts:4d} articles")

# Per-set counts (used for analysis)
for set_name, phrases in BIGRAM_SETS.items():
    df[set_name] = df['combined_text'].apply(
        lambda t: count_bigram_set(t, phrases)
    )

# ── Classify each article into its PRIMARY term set ──────────
# Rule: whichever set has the most mentions.
# Ties broken by SET_PRIORITY order.
# Articles with zero mentions in all sets → default to first in SET_PRIORITY.
def classify_article(row):
    counts = {s: row[s] for s in BIGRAM_SETS}
    max_count = max(counts.values())
    for s in SET_PRIORITY:               # priority for ties / zero
        if counts[s] == max_count:
            return s

df['Primary Set'] = df.apply(classify_article, axis=1)

print(f"\n✓ Primary term set classification:")
for s in BIGRAM_SETS:
    n = int((df['Primary Set'] == s).sum())
    print(f"  {s}: {n} articles")
print(f"  TOTAL: {len(df)} articles")

# ============================================================
# STEP 3 — AGGREGATE BY YEAR × DISCIPLINE
# ============================================================
print(f"\n{SEP}")
print("AGGREGATING BY YEAR × DISCIPLINE")
print(SEP)

# mention-based pivot (for combined graph + overall stats)
mention_pivot  = {}   # set_name → DataFrame(year × discipline, mentions)
combined_ment  = {}   # set_name → Series(year, total mentions all disciplines)

# article-based pivot (for per-discipline graphs — totals = all articles)
article_pivot  = {}   # set_name → DataFrame(year × discipline, article count)
combined_art   = {}   # set_name → Series(year, total articles all disciplines)

for set_name in BIGRAM_SETS:
    # ── mentions ─────────────────────────────────────────────
    m_df = df[['Year', CATEGORY_COLUMN, set_name]].copy()
    m_df.columns = ['Year', 'Discipline', 'Mentions']
    m_grouped = m_df.groupby(['Year', 'Discipline'])['Mentions'].sum().reset_index()
    m_pivot = (
        m_grouped
        .pivot_table(index='Year', columns='Discipline',
                     values='Mentions', fill_value=0)
        .reindex(index=year_range, fill_value=0)
        .reindex(columns=all_disciplines, fill_value=0)
    )
    mention_pivot[set_name] = m_pivot
    combined_ment[set_name] = m_pivot.sum(axis=1)

    # ── article counts (primary-set articles only) ───────────
    a_df = df[df['Primary Set'] == set_name][['Year', CATEGORY_COLUMN]].copy()
    a_df = a_df.assign(Count=1)
    a_grouped = a_df.groupby(['Year', CATEGORY_COLUMN])['Count'].sum().reset_index()
    a_grouped.columns = ['Year', 'Discipline', 'Count']
    a_pivot = (
        a_grouped
        .pivot_table(index='Year', columns='Discipline',
                     values='Count', fill_value=0)
        .reindex(index=year_range, fill_value=0)
        .reindex(columns=all_disciplines, fill_value=0)
    )
    article_pivot[set_name] = a_pivot
    combined_art[set_name]  = a_pivot.sum(axis=1)

total_rows = sum(len(v) * len(v.columns) for v in mention_pivot.values())
print(f"✓ Aggregation complete")
print(f"✓ Total rows: {total_rows}")

# ============================================================
# STEP 4 — TREND ANALYSIS: OVERALL (all disciplines combined)
#          Uses MENTION counts — consistent with Cass's analysis
# ============================================================
print(f"\n{SEP}")
print("TREND ANALYSIS: OVERALL (All Disciplines Combined)")
print(SEP)

all_stats    = []
overall_rows = []

for set_name in BIGRAM_SETS:
    series = combined_ment[set_name].values
    tau, p_val, trend = mann_kendall_test(series)
    slope = get_slope(series)

    print(f"\n {set_name}:  {fmt_trend(trend)}")
    print(f"   Slope: {slope:.3f} mentions/year")
    if not np.isnan(p_val):
        print(f"   p-value: {p_val:.4f}")
    if not np.isnan(tau):
        print(f"   Kendall's Tau (effect size): {tau:.3f}")

    overall_rows.append({
        'Cultural Term':   set_name,
        'Trend':           trend,
        'Theil-Sen Slope': round(slope, 3),
        'Mann-Kendall p':  round(p_val, 4) if not np.isnan(p_val) else 'N/A',
        "Kendall's Tau":   round(tau, 3)   if not np.isnan(tau)   else 'N/A',
    })
    all_stats.append({
        'Term Set': set_name, 'Discipline': 'ALL (Combined)',
        'Total Mentions': int(combined_ment[set_name].sum()),
        'Total Articles': int(combined_art[set_name].sum()),
        'Trend': trend, 'Theil-Sen Slope': round(slope, 4),
        'p-value': round(p_val, 4) if not np.isnan(p_val) else 'N/A',
        'Mann-Kendall Tau': round(tau, 4) if not np.isnan(tau) else 'N/A',
    })

print(f"\n{pd.DataFrame(overall_rows).to_string(index=False)}")

# ============================================================
# STEP 5 — TREND ANALYSIS: BY DISCIPLINE (each term set)
#          Uses MENTION counts — consistent with Cass's analysis
# ============================================================
per_disc_stats = {}

for set_name in BIGRAM_SETS:
    print(f"\n{SEP}")
    print(f"TREND ANALYSIS: BY DISCIPLINE ({set_name})")
    print(SEP)

    pivot     = mention_pivot[set_name]
    a_pivot   = article_pivot[set_name]
    disc_rows = []

    for discipline in all_disciplines:
        series = pivot[discipline].values
        tau, p_val, trend = mann_kendall_test(series)
        slope = get_slope(series)
        total_ment = int(pivot[discipline].sum())
        total_art  = int(a_pivot[discipline].sum())

        print(f"\n {discipline}:  {fmt_trend(trend)}")
        print(f"   Total mentions: {total_ment}")
        print(f"   Growth rate: {slope:.3f} mentions/year")
        if not np.isnan(p_val):
            print(f"   p-value: {p_val:.4f}")

        disc_rows.append({
            'Discipline':  discipline,
            'Total':       total_ment,
            'Trend':       trend,
            'p-value':     round(p_val, 4) if not np.isnan(p_val) else 'N/A',
            'Slope/year':  round(slope, 3),
        })
        all_stats.append({
            'Term Set': set_name, 'Discipline': discipline,
            'Total Mentions': total_ment, 'Total Articles': total_art,
            'Trend': trend, 'Theil-Sen Slope': round(slope, 4),
            'p-value': round(p_val, 4) if not np.isnan(p_val) else 'N/A',
            'Mann-Kendall Tau': round(tau, 4) if not np.isnan(tau) else 'N/A',
        })

    disc_df = pd.DataFrame(disc_rows)
    print(f"\n{disc_df.to_string(index=False)}")
    per_disc_stats[set_name] = disc_df

# ============================================================
# STEP 6 — SAVE STATISTICS TO EXCEL
# ============================================================
stats_file = 'mann_kendall_results.xlsx'
with pd.ExcelWriter(stats_file, engine='openpyxl') as writer:
    pd.DataFrame(overall_rows).to_excel(
        writer, sheet_name='Overall Trends', index=False)
    for set_name, disc_df in per_disc_stats.items():
        disc_df.to_excel(
            writer, sheet_name=set_name.replace(' ', '_')[:31], index=False)
    pd.DataFrame(all_stats).to_excel(
        writer, sheet_name='All Stats', index=False)
print(f"\n✓ Saved {stats_file}")

# ============================================================
# GRAPH 1 — COMBINED: all 3 term sets, MENTION counts
#           (replicates Cass's combined graph)
# ============================================================
print(f"\n{SEP}")
print("DRAWING GRAPH 1 — Combined (all disciplines, mention counts)")
print(SEP)

fig, ax = plt.subplots(figsize=(14, 8))
for set_name in BIGRAM_SETS:
    color  = TERM_SET_COLORS[set_name]
    values = combined_ment[set_name].values
    ax.plot(year_range, values,
            marker='o', linewidth=2.5, markersize=7,
            label=set_name, color=color)
    add_data_labels(ax, year_range, values, color, fontsize=9)

ax.set_title(
    'Trend of Cultural Term Mentions Over Time\n(All Disciplines Combined)',
    fontsize=16, fontweight='bold', pad=15)
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
print(f"✓ Saved {combined_file}")

# ============================================================
# GRAPHS 2–4 — PER TERM SET: ARTICLE COUNTS by discipline
#   Y-axis = number of articles whose PRIMARY term set = this set
#   All 3 graphs together sum to total articles in dataset
# ============================================================
print(f"\n{SEP}")
print("DRAWING GRAPHS 2–4 — Per term set (article counts by discipline)")
print(SEP)

per_disc_files = [
    ('Humility Set',
     'Cultural Humility: Articles by Discipline (2015–2025)',
     'trend_graph_cultural_humility.png'),
    ('Competence Set',
     'Cultural Competence: Articles by Discipline (2015–2025)',
     'trend_graph_cultural_competence.png'),
    ('Awareness Set',
     'Cultural Awareness: Articles by Discipline (2015–2025)',
     'trend_graph_cultural_awareness.png'),
]

for set_name, title, filename in per_disc_files:
    pivot = article_pivot[set_name]
    n_articles = int(pivot.values.sum())

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, discipline in enumerate(all_disciplines):
        color  = get_color(discipline, i)
        values = pivot[discipline].values
        ax.plot(year_range, values,
                marker='o', linewidth=2.5, markersize=7,
                label=discipline, color=color)
        add_data_labels(ax, year_range, values, color, fontsize=8)

    ax.set_title(
        f"{title}\n(n = {n_articles} articles whose primary term is this set)",
        fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Number of Articles', fontsize=13)
    ax.set_xticks(year_range)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(title='Discipline', fontsize=11, title_fontsize=12,
              loc='upper left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.45)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved {filename}  ({n_articles} articles)")

# ── Verify total ─────────────────────────────────────────────
grand_total = sum(int(article_pivot[s].values.sum()) for s in BIGRAM_SETS)
print(f"\n  Article totals per graph:")
for s in BIGRAM_SETS:
    print(f"    {s}: {int(article_pivot[s].values.sum())} articles")
print(f"  GRAND TOTAL across 3 graphs: {grand_total} articles")

# ============================================================
# DOWNLOAD ALL 5 FILES INDIVIDUALLY (no ZIP)
# ============================================================
print(f"\n{SEP}")
print("DOWNLOADING FILES")
print(SEP)

all_downloads = (
    [stats_file, combined_file]
    + [fname for _, _, fname in per_disc_files]
)

for fname in all_downloads:
    files.download(fname)
    print(f"✓ {fname}")

print(f"\nAll done! {len(all_downloads)} files downloaded:")
for i, fname in enumerate(all_downloads, 1):
    print(f"  {i}. {fname}")
