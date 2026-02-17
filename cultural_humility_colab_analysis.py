# ============================================================
# CULTURAL HUMILITY ANALYSIS — TRACK 2
# Google Drive Excel → Bigram Analysis → Line Graphs
#
# DISCIPLINE CLASSIFICATION:
#   Uses Column E (Journal name) + Column A (Title) + Column C (Abstract)
#   to classify every article into one of the 4 disciplines.
#   If Journal /Category already has a valid discipline, that is used first.
#   All other values (journal names, Other, Unknown, blank) are resolved
#   automatically via keyword matching.
#
# GRAPH LOGIC:
#   Graph 1 (combined): Total MENTION counts, all disciplines — unchanged.
#   Graphs 2–4 (per term set): ARTICLE counts by discipline.
#     Each article is assigned to exactly ONE term set (most mentions wins).
#     The 3 graphs together = all articles in the dataset.
#
# OUTPUT — 5 files downloaded individually:
#   1. mann_kendall_results.xlsx
#   2. trend_graph_combined_all_disciplines.png
#   3. trend_graph_cultural_humility.png
#   4. trend_graph_cultural_competence.png
#   5. trend_graph_cultural_awareness.png
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

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
CATEGORY_COLUMN = 'Journal /Category'  # may contain journal names, Other, Unknown
JOURNAL_COLUMN  = 'Journal'            # Column E — actual journal name
YEAR_COLUMN     = 'Published'
TEXT_COLUMNS    = ['Title', 'Abstract']   # Columns A and C

YEAR_START = 2015
YEAR_END   = 2025
year_range = list(range(YEAR_START, YEAR_END + 1))

# The only 4 valid discipline labels we want on graphs
VALID_DISCIPLINES = ['Nursing', 'Counseling & Related', 'Public Health', 'Medicine']

# ── DISCIPLINE KEYWORD CLASSIFIER ────────────────────────────
# Searches: Journal name (E) + Title (A) + Abstract (C)
DISCIPLINE_KEYWORDS = {
    'Nursing': [
        'nursing', 'nurse ', 'nurses', 'midwifery', 'midwife',
        'nurse practitioner', 'nurse educator', 'nursing education',
        'nursing practice', 'nursing research', 'nursing care',
        'clinical nursing', 'neonatal', 'registered nurse',
    ],
    'Counseling & Related': [
        'counseling', 'counselling', 'counselor', 'counsellor',
        'psychology', 'psychological', 'psychotherapy', 'psychotherapist',
        'social work', 'social worker', 'mental health counseling',
        'marriage and family', 'family therapy', 'family therapist',
        'school counseling', 'rehabilitation counseling',
        'multicultural counseling', 'substance abuse counseling',
        'behavioral health', 'child welfare', 'case management',
    ],
    'Public Health': [
        'public health', 'community health', 'epidemiology', 'epidemiolog',
        'preventive medicine', 'health promotion', 'global health',
        'population health', 'health education', 'health equity',
        'health disparit', 'environmental health', 'occupational health',
        'maternal health', 'child health', 'health policy',
    ],
    'Medicine': [
        'medicine', 'medical school', 'medical education', 'physician',
        'academic medicine', 'clinical medicine', 'medical training',
        'surgery', 'surgical', 'pediatrics', 'geriatrics',
        'oncology', 'cardiology', 'hospital medicine',
        'medical student', 'medical resident',
    ],
}
# Tie-breaking priority (largest known groups first)
DISC_PRIORITY = ['Nursing', 'Counseling & Related', 'Public Health', 'Medicine']

def _infer_discipline_from_text(row):
    """Score each discipline by keyword hits in Journal + Title + Abstract."""
    text = ' '.join([
        str(row.get(JOURNAL_COLUMN, '') or ''),
        str(row.get('Title',         '') or ''),
        str(row.get('Abstract',      '') or ''),
    ]).lower()

    scores = {d: 0 for d in VALID_DISCIPLINES}
    for disc, keywords in DISCIPLINE_KEYWORDS.items():
        for kw in keywords:
            scores[disc] += text.count(kw)

    max_score = max(scores.values())
    if max_score == 0:
        return None   # no keyword match
    for d in DISC_PRIORITY:    # break ties by priority
        if scores[d] == max_score:
            return d

def resolve_discipline(row):
    """
    1. Use Journal /Category if it already holds a valid discipline.
    2. Otherwise infer from keyword matching (Journal + Title + Abstract).
    3. Fall back to 'Counseling & Related' if completely unresolvable.
    """
    existing = str(row[CATEGORY_COLUMN]).strip() if pd.notna(row[CATEGORY_COLUMN]) else ''
    if existing in VALID_DISCIPLINES:
        return existing
    inferred = _infer_discipline_from_text(row)
    return inferred or 'Counseling & Related'

# ── BIGRAM SETS ───────────────────────────────────────────────
BIGRAM_SETS = {
    'Humility Set':   ['cultural humility',   'culturally humble'],
    'Competence Set': ['cultural competence', 'cultural competency', 'culturally competent'],
    'Awareness Set':  ['cultural awareness',  'culturally aware'],
}
SET_PRIORITY = ['Competence Set', 'Humility Set', 'Awareness Set']

# ── COLORS ───────────────────────────────────────────────────
DISCIPLINE_COLORS = {
    'Nursing':              '#E63946',
    'Counseling & Related': '#457B9D',
    'Medicine':             '#1D3557',
    'Public Health':        '#2A9D8F',
}
TERM_SET_COLORS = {
    'Humility Set':   '#1f77b4',
    'Competence Set': '#c94b7d',
    'Awareness Set':  '#ff7f0e',
}

# ============================================================
# STEP 1 — LOAD DATA + CLASSIFY DISCIPLINES
# ============================================================
print("Loading data...")
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

df['Year'] = pd.to_numeric(df[YEAR_COLUMN].astype(str).str[:4], errors='coerce')
df = df[df['Year'].between(YEAR_START, YEAR_END)].copy()
df['Year'] = df['Year'].astype(int)
print(f"  ✓ {len(df)} articles  ({YEAR_START}–{YEAR_END})")

df['Discipline'] = df.apply(resolve_discipline, axis=1)
print(f"\n  Discipline breakdown after classification:")
for d in VALID_DISCIPLINES:
    n = int((df['Discipline'] == d).sum())
    print(f"    {d}: {n}")
print(f"    TOTAL: {len(df)}")

# ============================================================
# STEP 2 — COUNT BIGRAMS + CLASSIFY INTO TERM SETS
# ============================================================
print("\nCounting bigrams...")

df['combined_text'] = (
    df[TEXT_COLUMNS].fillna('').agg(' '.join, axis=1).str.lower()
)

def count_bigram_set(text, phrases):
    phrases_sorted = sorted(phrases, key=len, reverse=True)
    working, total = text, 0
    for phrase in phrases_sorted:
        total  += working.count(phrase)
        working = working.replace(phrase, ' ')
    return total

for set_name, phrases in BIGRAM_SETS.items():
    df[set_name] = df['combined_text'].apply(
        lambda t: count_bigram_set(t, phrases)
    )

# Assign each article to the term set with the most mentions
def classify_primary_set(row):
    counts    = {s: row[s] for s in BIGRAM_SETS}
    max_count = max(counts.values())
    for s in SET_PRIORITY:          # break ties / zeros by priority
        if counts[s] == max_count:
            return s

df['Primary Set'] = df.apply(classify_primary_set, axis=1)
print(f"  Primary term set assignment:")
for s in BIGRAM_SETS:
    print(f"    {s}: {int((df['Primary Set'] == s).sum())} articles")
print(f"    TOTAL: {len(df)} articles")

# ============================================================
# STEP 3 — AGGREGATE BY YEAR × DISCIPLINE
# ============================================================
mention_pivot = {}   # for combined graph (mention counts)
combined_ment = {}
article_pivot = {}   # for per-discipline graphs (article counts)

for set_name in BIGRAM_SETS:
    # ── mention counts (all articles) ────────────────────────
    m_df = df[['Year', 'Discipline', set_name]].copy()
    m_df.columns = ['Year', 'Discipline', 'Mentions']
    m_grp = m_df.groupby(['Year', 'Discipline'])['Mentions'].sum().reset_index()
    m_piv = (
        m_grp
        .pivot_table(index='Year', columns='Discipline',
                     values='Mentions', fill_value=0)
        .reindex(index=year_range, fill_value=0)
        .reindex(columns=VALID_DISCIPLINES, fill_value=0)
    )
    mention_pivot[set_name] = m_piv
    combined_ment[set_name] = m_piv.sum(axis=1)

    # ── article counts (primary-set articles only) ────────────
    a_df = df[df['Primary Set'] == set_name][['Year', 'Discipline']].copy()
    a_df['Count'] = 1
    a_grp = a_df.groupby(['Year', 'Discipline'])['Count'].sum().reset_index()
    a_piv = (
        a_grp
        .pivot_table(index='Year', columns='Discipline',
                     values='Count', fill_value=0)
        .reindex(index=year_range, fill_value=0)
        .reindex(columns=VALID_DISCIPLINES, fill_value=0)
    )
    article_pivot[set_name] = a_piv

# ============================================================
# STEP 4 — STATISTICS (Mann-Kendall + Theil-Sen → Excel only)
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
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    tau   = s / (0.5 * n * (n - 1))
    trend = ('increasing' if s > 0 else 'decreasing') if p_value < 0.05 else 'no significant trend'
    return tau, p_value, trend

def get_slope(series):
    if len(series) >= 2 and np.std(series) > 0:
        return theilslopes(series, year_range).slope
    return 0.0

rows = []
for set_name in BIGRAM_SETS:
    # Overall (all disciplines combined, mention counts)
    s_all = combined_ment[set_name].values
    tau, p, trend = mann_kendall_test(s_all)
    rows.append({'Term Set': set_name, 'Discipline': 'ALL (Combined)',
                 'Metric': 'Mentions', 'Total': int(s_all.sum()),
                 'Trend': trend,
                 'Theil-Sen Slope': round(get_slope(s_all), 4),
                 'p-value': round(p, 4)   if not np.isnan(p)   else 'N/A',
                 'Tau':     round(tau, 4) if not np.isnan(tau) else 'N/A'})
    # Per discipline (mention counts)
    for d in VALID_DISCIPLINES:
        s_d = mention_pivot[set_name][d].values
        tau, p, trend = mann_kendall_test(s_d)
        rows.append({'Term Set': set_name, 'Discipline': d,
                     'Metric': 'Mentions', 'Total': int(s_d.sum()),
                     'Trend': trend,
                     'Theil-Sen Slope': round(get_slope(s_d), 4),
                     'p-value': round(p, 4)   if not np.isnan(p)   else 'N/A',
                     'Tau':     round(tau, 4) if not np.isnan(tau) else 'N/A'})
    # Per discipline (article counts)
    for d in VALID_DISCIPLINES:
        s_a = article_pivot[set_name][d].values
        tau, p, trend = mann_kendall_test(s_a)
        rows.append({'Term Set': set_name, 'Discipline': d,
                     'Metric': 'Articles', 'Total': int(s_a.sum()),
                     'Trend': trend,
                     'Theil-Sen Slope': round(get_slope(s_a), 4),
                     'p-value': round(p, 4)   if not np.isnan(p)   else 'N/A',
                     'Tau':     round(tau, 4) if not np.isnan(tau) else 'N/A'})

stats_df   = pd.DataFrame(rows)
stats_file = 'mann_kendall_results.xlsx'
with pd.ExcelWriter(stats_file, engine='openpyxl') as writer:
    stats_df[stats_df['Discipline'] == 'ALL (Combined)'].drop(columns='Discipline').to_excel(
        writer, sheet_name='Overall Trends', index=False)
    stats_df[
        (stats_df['Discipline'] != 'ALL (Combined)') &
        (stats_df['Metric'] == 'Mentions')
    ].to_excel(writer, sheet_name='By Discipline (Mentions)', index=False)
    stats_df[
        (stats_df['Discipline'] != 'ALL (Combined)') &
        (stats_df['Metric'] == 'Articles')
    ].to_excel(writer, sheet_name='By Discipline (Articles)', index=False)
    stats_df.to_excel(writer, sheet_name='All Stats', index=False)
print(f"\n  ✓ Stats saved to {stats_file}")

# ============================================================
# HELPER — data-point labels
# ============================================================
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
# GRAPH 1 — COMBINED: all 3 term sets, all disciplines
#           Uses MENTION counts — matches Cass's graph exactly.
#           User confirmed: "The first graph is perfect."
# ============================================================
print("\nDrawing graphs...")

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
print(f"  ✓ {combined_file}")

# ============================================================
# GRAPHS 2–4 — PER TERM SET: article counts by discipline
#   - 4 discipline lines only (no journal names in legend)
#   - All 3 graphs together = all articles
# ============================================================
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
    fig, ax = plt.subplots(figsize=(13, 7))

    for discipline in VALID_DISCIPLINES:
        color  = DISCIPLINE_COLORS[discipline]
        values = pivot[discipline].values
        ax.plot(year_range, values,
                marker='o', linewidth=2.5, markersize=7,
                label=discipline, color=color)
        add_data_labels(ax, year_range, values, color, fontsize=8)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=16)
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
    print(f"  ✓ {filename}  (n={int(pivot.values.sum())} articles)")

grand_total = sum(int(article_pivot[s].values.sum()) for s in BIGRAM_SETS)
print(f"\n  Article count per graph:")
for s in BIGRAM_SETS:
    print(f"    {s}: {int(article_pivot[s].values.sum())}")
print(f"  Grand total: {grand_total}")

# ============================================================
# DOWNLOAD ALL 5 FILES
# ============================================================
all_downloads = [stats_file, combined_file] + [f for _, _, f in per_disc_files]
print(f"\nDownloading {len(all_downloads)} files...")
for fname in all_downloads:
    files.download(fname)
    print(f"  ✓ {fname}")
print("\nDone.")
