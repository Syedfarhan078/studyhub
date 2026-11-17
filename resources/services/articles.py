import html
import logging
import re
import urllib.parse
from datetime import datetime, timezone
import requests
import feedparser
from resources.models import ArticleResource

logger = logging.getLogger(__name__)

BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


def clean_html_snippet(raw_html: str) -> str:
    """Strips HTML tags and unescapes character entities for clean previews."""
    if not raw_html:
        return ""
    clean = re.sub(r'<[^>]+>', '', raw_html)
    return html.unescape(clean).strip()


def clean_topic_title(raw_title: str) -> str:
    """Removes leading numbering, bullets, or chapter codes for accurate search."""
    if not raw_title:
        return ""
    cleaned = re.sub(r'^(?:Topic\s*\d+[:.\s]*|\d+(?:\.\d+)*[:.\s]*|[-*•]\s*)', '', raw_title, flags=re.IGNORECASE)
    cleaned = cleaned.replace('"', '').replace("'", "").strip()
    return cleaned or raw_title.strip()


def fetch_wikipedia_articles(topic, max_results=3) -> int:
    """
    Fetches textbook-quality encyclopedic articles from Wikipedia MediaWiki API.
    Fast, reliable, free, and returns accurate mathematical and conceptual definitions.
    """
    subject_name = ""
    if hasattr(topic, 'module') and hasattr(topic.module, 'subject') and topic.module.subject:
        subject_name = topic.module.subject.name.strip()

    cleaned_title = clean_topic_title(topic.title)
    search_query = f"{cleaned_title} {subject_name}".strip()

    saved_count = 0
    try:
        wiki_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "utf8": 1,
            "format": "json",
            "srlimit": max_results,
        }
        resp = requests.get(wiki_url, params=params, headers=BROWSER_HEADERS, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            search_items = data.get("query", {}).get("search", [])
            for item in search_items:
                page_title = item.get("title", "")
                raw_snippet = item.get("snippet", "")
                snippet = clean_html_snippet(raw_snippet)
                if not page_title:
                    continue
                safe_slug = urllib.parse.quote(page_title.replace(" ", "_"))
                page_url = f"https://en.wikipedia.org/wiki/{safe_slug}"

                ArticleResource.objects.update_or_create(
                    topic=topic,
                    url=page_url,
                    defaults={
                        'title': page_title,
                        'source_name': "Wikipedia Academic",
                        'snippet': f"{snippet}..." if snippet else f"Encyclopedic academic overview and technical details for {page_title}.",
                        'published_date': None,
                    }
                )
                saved_count += 1
    except Exception as e:
        logger.warning("Wikipedia API fetch error for '%s': %s", topic.title, e)

    return saved_count


def fetch_technical_site_articles(topic, max_results=6) -> int:
    """
    Searches top engineering and programming tutorial portals (GeeksforGeeks,
    TutorialsPoint, Javatpoint, W3Schools) via targeted query feeds.
    """
    cleaned_title = clean_topic_title(topic.title)
    # Search top undergraduate engineering portals without arbitrary 3-word truncation
    query = f'"{cleaned_title}" (site:geeksforgeeks.org OR site:tutorialspoint.com OR site:javatpoint.com OR site:w3schools.com)'
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    saved_count = 0
    try:
        resp = requests.get(rss_url, headers=BROWSER_HEADERS, timeout=6)
        if resp.status_code == 200:
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:max_results]:
                title = html.unescape(entry.get('title', ''))
                link = entry.get('link', '')
                if not title or not link:
                    continue

                source_name = "Engineering Resource"
                link_lower = link.lower()
                title_lower = title.lower()

                if "geeksforgeeks" in link_lower or "geeksforgeeks" in title_lower:
                    source_name = "GeeksforGeeks"
                elif "tutorialspoint" in link_lower or "tutorialspoint" in title_lower:
                    source_name = "TutorialsPoint"
                elif "javatpoint" in link_lower or "javatpoint" in title_lower:
                    source_name = "Javatpoint"
                elif "w3schools" in link_lower or "w3schools" in title_lower:
                    source_name = "W3Schools"
                elif hasattr(entry, 'source') and isinstance(entry.source, dict) and entry.source.get('title'):
                    source_name = entry.source['title']
                elif ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    title = parts[0]
                    source_name = parts[1]

                snippet = clean_html_snippet(entry.get('summary', ''))
                published_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except Exception:
                        pass

                ArticleResource.objects.update_or_create(
                    topic=topic,
                    url=link,
                    defaults={
                        'title': title,
                        'source_name': source_name,
                        'snippet': snippet or f"Educational article and guide on {cleaned_title}.",
                        'published_date': published_date,
                    }
                )
                saved_count += 1
    except Exception as e:
        logger.warning("Technical sites RSS fetch error for '%s': %s", topic.title, e)

    return saved_count


def fetch_arxiv_articles(topic, max_results=2) -> int:
    """
    Fetches open-access research papers from arXiv for advanced computer science concepts.
    """
    cleaned_title = clean_topic_title(topic.title)
    saved_count = 0
    try:
        arxiv_query = urllib.parse.quote(cleaned_title)
        arxiv_url = f"https://export.arxiv.org/api/query?search_query=all:{arxiv_query}&start=0&max_results={max_results}"
        arxiv_resp = requests.get(arxiv_url, headers=BROWSER_HEADERS, timeout=6)

        if arxiv_resp.status_code == 200:
            arxiv_feed = feedparser.parse(arxiv_resp.content)
            for entry in arxiv_feed.entries:
                title = html.unescape(entry.get('title', '')).replace('\n', ' ').strip()
                link = entry.get('link', '')
                summary = html.unescape(entry.get('summary', '')).replace('\n', ' ').strip()

                if not title or not link:
                    continue

                published_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except Exception:
                        pass

                ArticleResource.objects.update_or_create(
                    topic=topic,
                    url=link,
                    defaults={
                        'title': title,
                        'source_name': "arXiv Research",
                        'snippet': summary[:280] + "..." if len(summary) > 280 else summary,
                        'published_date': published_date,
                    }
                )
                saved_count += 1
    except Exception as e:
        logger.warning("arXiv fetch error for '%s': %s", topic.title, e)

    return saved_count


def get_or_fetch_articles_for_topic(topic, force_refresh=False):
    """
    Discovers educational articles and documentation for a syllabus topic.
    1. Returns cached articles from PostgreSQL if available.
    2. On fresh retrieval, aggregates from:
       - Wikipedia Academic (Core concepts and definitions)
       - Top Engineering Portals (GeeksforGeeks, TutorialsPoint, Javatpoint)
       - arXiv Research (Advanced research enrichments if needed)
    """
    cached_articles = topic.articles.all()
    if cached_articles.exists() and not force_refresh:
        return cached_articles

    if force_refresh:
        topic.articles.all().delete()

    # Step 1: Query Wikipedia Academic (reliable, accurate conceptual ground truth)
    wiki_count = fetch_wikipedia_articles(topic, max_results=3)

    # Step 2: Query Targeted Engineering Tutorial Sites
    tech_count = fetch_technical_site_articles(topic, max_results=5)

    # Step 3: If still under 3 articles, enrich with arXiv Computer Science papers
    total_found = wiki_count + tech_count
    if total_found < 3:
        fetch_arxiv_articles(topic, max_results=2)

    return topic.articles.all()
