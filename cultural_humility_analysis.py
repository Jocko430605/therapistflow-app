# ============================================================
# 🌐 CULTURAL HUMILITY ANALYSIS: FREE APIS + TRENDS
# ============================================================

# ----------------------------
# 1. INSTALL PACKAGES
# ----------------------------
!pip install pandas matplotlib seaborn requests lxml openpyxl beautifulsoup4 -q

# ----------------------------
# 2. IMPORTS
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import warnings
warnings.filterwarnings("ignore")

import requests
from xml.etree import ElementTree as ET
from google.colab import files
from bs4 import BeautifulSoup

# ----------------------------
# 3. CONFIGURATION
# ----------------------------
YOUR_EMAIL = "justinjacques@humantheorygroup.com"

# API MAXIMUM LIMITS (articles per query):
# - arXiv: 2,000
# - DOAJ: 100 (per request limit)
# - Europe PMC: 1,000
# - ERIC: 2,000
# - NIH/PubMed: 500 (recommended max)
# - DOE OSTI: 1,000
# - NASA STI: 1,000
# - Crossref: 1,000
# - ClinicalTrials.gov: 1,000
# - OSF Preprints: 100 (per request limit)
# - SciELO: 1,000
# - CORE: 100

cultural_terms = ["cultural humility", "cultural competence", "cultural awareness"]
discipline_terms = ["nursing", "medicine", "public health", "counseling", "psychology", "social work", "therapy"]
year_range = list(range(2015, 2026))

def group_discipline(discipline):
    """Group related disciplines together"""
    if discipline.lower() in ["counseling", "psychology", "social work", "therapy"]:
        return "Counseling & Related"
    return discipline.title()

def clean_jats_xml(text):
    """Remove JATS XML tags from abstracts"""
    if not text:
        return ""
    try:
        # Use BeautifulSoup to strip all XML/HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except:
        # Fallback: simple regex to remove tags
        return re.sub(r'<[^>]+>', '', text)

# ----------------------------
# 4. FREE ACADEMIC & GOVERNMENT APIS
# ----------------------------
class FreeAcademicAPIs:
    """Collection of free academic and government APIs (NO OpenAlex)"""

    def __init__(self):
        self.base_urls = {
            'arxiv': 'http://export.arxiv.org/api/query',
            'doaj': 'https://doaj.org/api/v1/search/articles/',
            'core': 'https://core.ac.uk:443/api-v2/search/works',
            'europe_pmc': 'https://www.ebi.ac.uk/europepmc/webservices/rest/search/',
            'eric': 'https://api.ies.ed.gov/eric/',
            'nih': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
            'doe': 'https://www.osti.gov/api/v1/records/',
            'nasa': 'https://ntrs.nasa.gov/api/citations/search',
            'crossref': 'https://api.crossref.org/works',
            'clinical_trials': 'https://clinicaltrials.gov/api/v2/studies',
            'osf_preprints': 'https://api.osf.io/v2/preprints/',
            'scielo': 'https://search.scielo.org/api/v1/search'
        }

    def search_arxiv(self, query: str, max_results: int = 2000) -> list:
        """Search arXiv for academic papers (MAX: 2000)"""
        print(f"  📚 Searching arXiv (requesting up to {max_results})...")
        try:
            params = {'search_query': query, 'max_results': max_results, 'start': 0}
            response = requests.get(self.base_urls['arxiv'], params=params, timeout=30)
            root = ET.fromstring(response.content)
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title')
                summary = entry.find('{http://www.w3.org/2005/Atom}summary')
                published = entry.find('{http://www.w3.org/2005/Atom}published')
                if title is not None:
                    papers.append({
                        "Title": title.text.strip() if title.text else "No title",
                        "Published": int(published.text[:4]) if published is not None else None,
                        "Abstract": summary.text.strip() if summary is not None else "",
                        "Source": "arXiv",
                        "Journal": "N/A",
                        "Citation_Info": "N/A",
                        "ISSN": "N/A"
                    })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ arXiv error: {e}")
            return []

    def search_doaj(self, query: str, max_results: int = 100) -> list:
        """Search Directory of Open Access Journals (MAX: 100 per request)"""
        print(f"  📚 Searching DOAJ (requesting up to {max_results})...")
        try:
            params = {'q': query, 'pageSize': max_results}
            response = requests.get(self.base_urls['doaj'], params=params, timeout=20)
            data = response.json()
            papers = []
            for article in data.get('results', []):
                bibjson = article.get('bibjson', {})
                papers.append({
                    "Title": bibjson.get('title', 'No title'),
                    "Published": int(bibjson.get('year')) if bibjson.get('year') else None,
                    "Abstract": bibjson.get('abstract', ''),
                    "Source": "DOAJ",
                    "Journal": "N/A",
                    "Citation_Info": "N/A",
                    "ISSN": "N/A"
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ DOAJ error: {e}")
            return []

    def search_europe_pmc(self, query: str, max_results: int = 1000) -> list:
        """Search Europe PubMed Central (MAX: 1000)"""
        print(f"  📚 Searching Europe PMC (requesting up to {max_results})...")
        try:
            params = {
                'query': query,
                'format': 'json',
                'pageSize': max_results,
                'resultType': 'core'
            }
            response = requests.get(self.base_urls['europe_pmc'], params=params, timeout=30)
            data = response.json()
            papers = []
            for article in data.get('resultList', {}).get('result', []):
                papers.append({
                    "Title": article.get('title', 'No title'),
                    "Published": int(article.get('pubYear')) if article.get('pubYear') else None,
                    "Abstract": article.get('abstractText', ''),
                    "Source": "Europe PMC",
                    "Journal": "N/A",
                    "Citation_Info": "N/A",
                    "ISSN": "N/A"
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ Europe PMC error: {e}")
            return []

    def search_eric(self, query: str, max_results: int = 2000) -> list:
        """Search ERIC - Education Resources Information Center (MAX: 2000)"""
        print(f"  📚 Searching ERIC (requesting up to {max_results})...")
        try:
            params = {'search': query, 'rows': max_results, 'format': 'json'}
            response = requests.get(self.base_urls['eric'], params=params, timeout=30)
            data = response.json()
            papers = []
            for article in data.get('response', {}).get('docs', []):
                papers.append({
                    "Title": article.get('title', 'No title'),
                    "Published": int(article.get('publicationyear', '0')[:4]) if article.get('publicationyear') else None,
                    "Abstract": article.get('abstract', ''),
                    "Source": "ERIC",
                    "Journal": "N/A",
                    "Citation_Info": "N/A",
                    "ISSN": "N/A"
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ ERIC error: {e}")
            return []

    def search_nih(self, query: str, max_results: int = 500) -> list:
        """Search NIH/PubMed (MAX: 500 recommended)"""
        print(f"  📚 Searching NIH/PubMed (requesting up to {max_results})...")
        try:
            search_url = f"{self.base_urls['nih']}esearch.fcgi"
            search_params = {'db': 'pubmed', 'term': query, 'retmax': max_results, 'retmode': 'json'}
            search_resp = requests.get(search_url, params=search_params, timeout=30)
            pmids = search_resp.json().get('esearchresult', {}).get('idlist', [])

            if not pmids:
                print(f"    ✅ Found 0 articles")
                return []

            fetch_url = f"{self.base_urls['nih']}efetch.fcgi"
            fetch_params = {'db': 'pubmed', 'id': ','.join(pmids[:max_results]), 'retmode': 'xml'}
            fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=45)
            root = ET.fromstring(fetch_resp.content)

            papers = []
            for article in root.findall('.//PubmedArticle'):
                title_elem = article.find('.//ArticleTitle')
                year_elem = article.find('.//PubDate/Year')
                abstract_elem = article.find('.//AbstractText')
                papers.append({
                    "Title": title_elem.text if title_elem is not None else "No title",
                    "Published": int(year_elem.text) if year_elem is not None and year_elem.text.isdigit() else None,
                    "Abstract": abstract_elem.text if abstract_elem is not None else "",
                    "Source": "NIH/PubMed",
                    "Journal": "N/A",
                    "Citation_Info": "N/A",
                    "ISSN": "N/A"
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ NIH/PubMed error: {e}")
            return []

    def search_doe(self, query: str, max_results: int = 1000) -> list:
        """Search DOE OSTI - Department of Energy (MAX: 1000)"""
        print(f"  📚 Searching DOE OSTI (requesting up to {max_results})...")
        try:
            params = {'query': query, 'size': max_results}
            response = requests.get(self.base_urls['doe'], params=params, timeout=30)
            data = response.json()
            papers = []
            for record in data.get('records', []):
                papers.append({
                    "Title": record.get('title', 'No title'),
                    "Published": int(record.get('publication_date', '')[:4]) if record.get('publication_date') else None,
                    "Abstract": record.get('description', ''),
                    "Source": "DOE OSTI",
                    "Journal": "N/A",
                    "Citation_Info": "N/A",
                    "ISSN": "N/A"
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ DOE OSTI error: {e}")
            return []

    def search_nasa(self, query: str, max_results: int = 1000) -> list:
        """Search NASA STI - Scientific and Technical Information (MAX: 1000)"""
        print(f"  📚 Searching NASA STI (requesting up to {max_results})...")
        try:
            params = {'q': query, 'page.size': max_results}
            response = requests.get(self.base_urls['nasa'], params=params, timeout=30)
            data = response.json()
            papers = []
            for doc in data.get('hits', []):
                papers.append({
                    "Title": doc.get('title', 'No title'),
                    "Published": int(doc.get('created', '')[:4]) if doc.get('created') else None,
                    "Abstract": doc.get('abstract', ''),
                    "Source": "NASA STI",
                    "Journal": "N/A",
                    "Citation_Info": "N/A",
                    "ISSN": "N/A"
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ NASA STI error: {e}")
            return []

    def search_crossref(self, query: str, max_results: int = 1000) -> list:
        """Search Crossref - Massive multi-disciplinary database (MAX: 1000)"""
        print(f"  📚 Searching Crossref (requesting up to {max_results})...")
        try:
            params = {
                'query': query,
                'rows': max_results,
                'filter': 'from-pub-date:2015,until-pub-date:2025'
            }
            response = requests.get(self.base_urls['crossref'], params=params, timeout=30)
            data = response.json()
            papers = []
            for item in data.get('message', {}).get('items', []):
                # Extract year from published date
                year = None
                if 'published' in item and 'date-parts' in item['published']:
                    date_parts = item['published']['date-parts'][0]
                    if date_parts:
                        year = int(date_parts[0])

                # Extract journal metadata
                journal_title = item.get('container-title', ['Unknown Journal'])[0] if item.get('container-title') else 'Unknown Journal'
                volume = item.get('volume', '')
                issue = item.get('issue', '')
                pages = item.get('page', '')
                issn_list = item.get('ISSN', [])
                issn = issn_list[0] if issn_list else ''

                # Build citation info
                citation_parts = []
                if volume:
                    citation_parts.append(f"Vol {volume}")
                if issue:
                    citation_parts.append(f"Issue {issue}")
                if pages:
                    citation_parts.append(f"pp. {pages}")
                citation_info = ', '.join(citation_parts) if citation_parts else 'N/A'

                # Get and clean abstract (remove JATS XML tags)
                raw_abstract = item.get('abstract', '')
                clean_abstract = clean_jats_xml(raw_abstract)

                papers.append({
                    "Title": item.get('title', ['No title'])[0] if item.get('title') else 'No title',
                    "Published": year,
                    "Abstract": clean_abstract,
                    "Source": "Crossref",
                    "Journal": journal_title,
                    "Citation_Info": citation_info,
                    "ISSN": issn
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ Crossref error: {e}")
            return []

    def search_clinical_trials(self, query: str, max_results: int = 1000) -> list:
        """Search ClinicalTrials.gov - Medical & nursing clinical trials (MAX: 1000)"""
        print(f"  📚 Searching ClinicalTrials.gov (requesting up to {max_results})...")
        try:
            params = {
                'query.term': query,
                'pageSize': max_results,
                'format': 'json'
            }
            response = requests.get(self.base_urls['clinical_trials'], params=params, timeout=30)
            data = response.json()
            papers = []
            for study in data.get('studies', []):
                protocol = study.get('protocolSection', {})
                identification = protocol.get('identificationModule', {})
                description = protocol.get('descriptionModule', {})
                status = protocol.get('statusModule', {})

                # Extract year from start date
                year = None
                if 'startDateStruct' in status:
                    year = status['startDateStruct'].get('year')

                papers.append({
                    "Title": identification.get('officialTitle', identification.get('briefTitle', 'No title')),
                    "Published": int(year) if year else None,
                    "Abstract": description.get('briefSummary', description.get('detailedDescription', '')),
                    "Source": "ClinicalTrials.gov",
                    "Journal": "N/A",
                    "Citation_Info": "N/A",
                    "ISSN": "N/A"
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ ClinicalTrials.gov error: {e}")
            return []

    def search_osf_preprints(self, query: str, max_results: int = 100) -> list:
        """Search OSF Preprints (PsyArXiv & SocArXiv) - Psychology & Social Sciences (MAX: 100)"""
        print(f"  📚 Searching OSF Preprints (PsyArXiv/SocArXiv) (requesting up to {max_results})...")
        try:
            params = {
                'filter[subjects]': query,
                'page[size]': max_results
            }
            response = requests.get(self.base_urls['osf_preprints'], params=params, timeout=30)
            data = response.json()
            papers = []
            for item in data.get('data', []):
                attrs = item.get('attributes', {})

                # Extract year from date_published or date_created
                year = None
                date_str = attrs.get('date_published') or attrs.get('date_created', '')
                if date_str:
                    year = int(date_str[:4])

                papers.append({
                    "Title": attrs.get('title', 'No title'),
                    "Published": year,
                    "Abstract": attrs.get('description', ''),
                    "Source": "OSF Preprints",
                    "Journal": "N/A",
                    "Citation_Info": "N/A",
                    "ISSN": "N/A"
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ OSF Preprints error: {e}")
            return []

    def search_scielo(self, query: str, max_results: int = 1000) -> list:
        """Search SciELO - Social Sciences & Regional Health (MAX: 1000)"""
        print(f"  📚 Searching SciELO (requesting up to {max_results})...")
        try:
            params = {
                'q': query,
                'count': max_results,
                'output': 'json'
            }
            response = requests.get(self.base_urls['scielo'], params=params, timeout=30)
            data = response.json()
            papers = []
            for item in data.get('response', {}).get('docs', []):
                # Extract year from publication date
                year = None
                pub_date = item.get('publication_date', '')
                if pub_date:
                    year = int(pub_date[:4])

                papers.append({
                    "Title": item.get('title', 'No title'),
                    "Published": year,
                    "Abstract": item.get('abstract', ''),
                    "Source": "SciELO",
                    "Journal": "N/A",
                    "Citation_Info": "N/A",
                    "ISSN": "N/A"
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ SciELO error: {e}")
            return []

    def search_core(self, query: str, max_results: int = 100) -> list:
        """Search CORE - Open access research papers (MAX: 100)"""
        print(f"  📚 Searching CORE (requesting up to {max_results})...")
        try:
            params = {
                'q': query,
                'limit': max_results
            }
            response = requests.get(self.base_urls['core'], params=params, timeout=30)
            data = response.json()
            papers = []
            for item in data.get('results', []):
                # Extract year
                year = None
                year_published = item.get('yearPublished')
                if year_published:
                    year = int(year_published)

                papers.append({
                    "Title": item.get('title', 'No title'),
                    "Published": year,
                    "Abstract": item.get('abstract', item.get('description', '')),
                    "Source": "CORE",
                    "Journal": "N/A",
                    "Citation_Info": "N/A",
                    "ISSN": "N/A"
                })
            print(f"    ✅ Found {len(papers)} articles")
            return papers
        except Exception as e:
            print(f"    ⚠️ CORE error: {e}")
            return []

# ----------------------------
# 5. FETCH FROM ALL FREE APIS
# ----------------------------
print("\n🌐 FETCHING FROM FREE ACADEMIC & GOVERNMENT APIS")
print("="*60)
print("\n📊 MAXIMUM DATA COLLECTION MODE ENABLED")
print("   • arXiv: 2,000 articles per query")
print("   • DOAJ: 100 articles per query")
print("   • Europe PMC: 1,000 articles per query")
print("   • ERIC: 2,000 articles per query")
print("   • NIH/PubMed: 500 articles per query")
print("   • DOE OSTI: 1,000 articles per query")
print("   • NASA STI: 1,000 articles per query")
print("   • Crossref: 1,000 articles per query")
print("   • ClinicalTrials.gov: 1,000 articles per query")
print("   • OSF Preprints: 100 articles per query")
print("   • SciELO: 1,000 articles per query")
print("   • CORE: 100 articles per query")
print(f"\n   Running 12 APIs × 6 queries = 72 API calls")
print("   ⏱️ Estimated time: 5-8 minutes\n")

api_hub = FreeAcademicAPIs()

# Build comprehensive search queries
queries = [
    "cultural humility healthcare",
    "cultural competence nursing",
    "cultural awareness medicine",
    "cultural humility counseling",
    "cultural competence psychology",
    "cultural awareness therapy"
]

all_results = []

for idx, query in enumerate(queries, 1):
    print(f"\n🔍 Search query {idx}/{len(queries)}: {query}")
    print("-"*60)

    # Search each API with MAXIMUM results
    all_results.extend(api_hub.search_arxiv(query, max_results=2000))
    time.sleep(2)

    all_results.extend(api_hub.search_doaj(query, max_results=100))
    time.sleep(2)

    all_results.extend(api_hub.search_europe_pmc(query, max_results=1000))
    time.sleep(2)

    all_results.extend(api_hub.search_eric(query, max_results=2000))
    time.sleep(2)

    all_results.extend(api_hub.search_nih(query, max_results=500))
    time.sleep(2)

    all_results.extend(api_hub.search_doe(query, max_results=1000))
    time.sleep(2)

    all_results.extend(api_hub.search_nasa(query, max_results=1000))
    time.sleep(2)

    all_results.extend(api_hub.search_crossref(query, max_results=1000))
    time.sleep(2)

    all_results.extend(api_hub.search_clinical_trials(query, max_results=1000))
    time.sleep(2)

    all_results.extend(api_hub.search_osf_preprints(query, max_results=100))
    time.sleep(2)

    all_results.extend(api_hub.search_scielo(query, max_results=1000))
    time.sleep(2)

    all_results.extend(api_hub.search_core(query, max_results=100))
    time.sleep(2)

print(f"\n📊 Total articles collected: {len(all_results)}")

# ----------------------------
# 6. CREATE DATAFRAME & DEDUPLICATE
# ----------------------------
print("\n🔧 Processing and deduplicating...")
df = pd.DataFrame(all_results)

# Remove duplicates by title (case-insensitive)
df['Title_Lower'] = df['Title'].str.lower().fillna('')
initial_count = len(df)
df = df.drop_duplicates(subset=['Title_Lower'], keep='first')
deduplicated_count = len(df)

print(f"✅ Removed {initial_count - deduplicated_count} duplicates")
print(f"✅ Unique articles: {deduplicated_count}")

# ----------------------------
# 7. VALIDATE & FILTER
# ----------------------------
print("\n🔍 Validating articles...")

def passes_validation(row):
    """Check if article meets all criteria"""
    # Year filter
    if not (2015 <= (row["Published"] or 0) <= 2025):
        return False

    # Abstract must exist and be substantial
    abstract = str(row["Abstract"]).lower()
    if len(abstract) < 50:
        return False

    # Must contain at least one cultural term
    has_cultural = any(term in abstract for term in cultural_terms)
    if not has_cultural:
        return False

    # Must contain at least one discipline term
    has_discipline = any(discipline in abstract for discipline in discipline_terms)
    if not has_discipline:
        return False

    # English text check (basic heuristic - check for common English words)
    english_indicators = ['the', 'and', 'of', 'to', 'in', 'a', 'is', 'for', 'that', 'with']
    if not any(word in abstract for word in english_indicators):
        return False

    return True

# Apply validation
final_df = df[df.apply(passes_validation, axis=1)].reset_index(drop=True)

print(f"✅ Validated articles: {len(final_df)}")
print(f"📉 Filtered out: {deduplicated_count - len(final_df)} articles")

# ----------------------------
# 8. TREND ANALYSIS
# ----------------------------
print("\n📊 Performing trend analysis...")

# Initialize data structure for trends
term_data = {
    term: {
        year: {discipline: 0 for discipline in discipline_terms}
        for year in year_range
    }
    for term in cultural_terms
}

# Count mentions
for _, row in final_df.iterrows():
    abstract = str(row["Abstract"]).lower()
    year = row["Published"]

    for term in cultural_terms:
        if term in abstract:
            for discipline in discipline_terms:
                if discipline in abstract:
                    term_data[term][year][discipline] += 1

# Convert to records for visualization
records = []
for term in term_data:
    for year in term_data[term]:
        for discipline, count in term_data[term][year].items():
            if count > 0:
                records.append({
                    "Cultural Term": term.title(),
                    "Year": year,
                    "Discipline": group_discipline(discipline),
                    "Mentions": count
                })

df_trends = pd.DataFrame(records)

print(f"✅ Trend analysis complete: {len(df_trends)} data points")

# ----------------------------
# 9. VISUALIZATIONS
# ----------------------------
print("\n🎨 Creating visualizations...")

# 9.1 STACKED BAR CHARTS
print("  📊 Generating stacked bar charts...")
for term in cultural_terms:
    term_df = df_trends[df_trends["Cultural Term"] == term.title()]
    if not term_df.empty:
        df_plot = term_df.pivot_table(
            index="Year",
            columns="Discipline",
            values="Mentions",
            aggfunc="sum",
            fill_value=0
        ).astype(int)

        ax = df_plot.plot(kind="bar", stacked=True, figsize=(14, 7), colormap="Set2")

        # Add count labels on bars
        for i, year in enumerate(df_plot.index):
            y_offset = 0
            for discipline in df_plot.columns:
                value = df_plot.loc[year, discipline]
                if value > 0:
                    ax.text(i, y_offset + value / 2, str(value),
                           ha='center', va='center', fontsize=9, fontweight='bold')
                    y_offset += value

        plt.title(f"Mentions of '{term.title()}' by Discipline (2015–2025)",
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Publication Year", fontsize=12, fontweight='bold')
        plt.ylabel("Number of Mentions", fontsize=12, fontweight='bold')
        plt.legend(title="Discipline", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

# 9.2 LINE CHARTS
print("  📈 Generating line charts...")
for term in cultural_terms:
    term_df = df_trends[df_trends["Cultural Term"] == term.title()]
    if not term_df.empty:
        df_plot = term_df.pivot_table(
            index="Year",
            columns="Discipline",
            values="Mentions",
            aggfunc="sum",
            fill_value=0
        ).astype(int)

        plt.figure(figsize=(14, 7))
        for discipline in df_plot.columns:
            plt.plot(df_plot.index, df_plot[discipline], marker='o',
                    linewidth=2.5, markersize=8, label=discipline)

            # Add count labels on points
            for x, y in zip(df_plot.index, df_plot[discipline]):
                if y > 0:
                    plt.text(x, y + 0.3, str(y), ha='center', va='bottom',
                            fontsize=9, fontweight='bold')

        plt.title(f"Trend of '{term.title()}' Mentions by Discipline (2015–2025)",
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Publication Year", fontsize=12, fontweight='bold')
        plt.ylabel("Number of Mentions", fontsize=12, fontweight='bold')
        plt.xticks(df_plot.index, rotation=45)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Discipline", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        plt.tight_layout()
        plt.show()

print("✅ All visualizations created")

# ----------------------------
# 10. SAVE TO EXCEL & AUTO-DOWNLOAD
# ----------------------------
print("\n💾 Saving results to Excel...")

# Create Excel file with multiple sheets
excel_filename = "cultural_humility_analysis_results.xlsx"

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Sheet 1: Main dataset with all articles
    # Define columns to export (include new Crossref fields if they exist)
    export_columns = ['Title', 'Published', 'Abstract', 'Source']
    if 'Journal' in final_df.columns:
        export_columns.append('Journal')
    if 'Citation_Info' in final_df.columns:
        export_columns.append('Citation_Info')
    if 'ISSN' in final_df.columns:
        export_columns.append('ISSN')

    final_df[export_columns].to_excel(
        writer,
        sheet_name='Articles',
        index=False
    )

    # Sheet 2: Trend analysis data
    df_trends.to_excel(
        writer,
        sheet_name='Trend Analysis',
        index=False
    )

    # Sheet 3: Summary statistics
    summary_data = {
        'Metric': [
            'Total Articles',
            'Date Range',
            'Sources Used',
            'Cultural Humility Mentions',
            'Cultural Competence Mentions',
            'Cultural Awareness Mentions'
        ],
        'Value': [
            len(final_df),
            '2015-2025',
            ', '.join(final_df['Source'].unique()),
            len(final_df[final_df['Abstract'].str.contains('cultural humility', case=False, na=False)]),
            len(final_df[final_df['Abstract'].str.contains('cultural competence', case=False, na=False)]),
            len(final_df[final_df['Abstract'].str.contains('cultural awareness', case=False, na=False)])
        ]
    }
    pd.DataFrame(summary_data).to_excel(
        writer,
        sheet_name='Summary',
        index=False
    )

print(f"✅ Excel file created: {excel_filename}")

# Auto-download in Colab
print("\n⬇️ Downloading file...")
try:
    files.download(excel_filename)
    print(f"✅ Downloaded: {excel_filename}")
except Exception as e:
    print(f"⚠️ Auto-download failed: {e}")
    print(f"📁 File is available in the Colab file browser (left panel): {excel_filename}")

# ----------------------------
# 11. FINAL SUMMARY
# ----------------------------
print("\n" + "="*60)
print("🎉 ANALYSIS COMPLETE!")
print("="*60)
print(f"\n📊 RESULTS SUMMARY:")
print(f"   • Total validated articles: {len(final_df)}")
print(f"   • Year range: 2015-2025")
print(f"   • Sources: {', '.join(sorted(final_df['Source'].unique()))}")
print(f"\n📈 CULTURAL TERMS FOUND:")
for term in cultural_terms:
    count = len(final_df[final_df['Abstract'].str.contains(term, case=False, na=False)])
    print(f"   • {term.title()}: {count} articles")
print(f"\n🏥 DISCIPLINES ANALYZED:")
for discipline in discipline_terms:
    count = len(final_df[final_df['Abstract'].str.contains(discipline, case=False, na=False)])
    print(f"   • {discipline.title()}: {count} mentions")
print(f"\n💾 OUTPUT FILE: {excel_filename}")
print(f"   📄 Sheet 1: Articles ({len(final_df)} rows)")
print(f"   📄 Sheet 2: Trend Analysis ({len(df_trends)} data points)")
print(f"   📄 Sheet 3: Summary Statistics")
print("\n✅ All data has been saved and is ready for analysis!")
print("="*60)
