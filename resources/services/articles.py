import html
import re
import urllib.parse
from datetime import datetime, timezone
import requests
import feedparser
from resources.models import ArticleResource

BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


def clean_html_snippet(raw_html):
    """Strips HTML tags and unescapes character entities for clean previews."""
    if not raw_html:
        return ""
    clean = re.sub(r'<[^>]+>', '', raw_html)
    return html.unescape(clean).strip()


def get_or_fetch_articles_for_topic(topic, force_refresh=False):
    """
    Discovers educational articles and research papers for a syllabus topic.
    Uses browser-emulated headers and multi-source feeds (Google Educational RSS & arXiv).
    """
    cached_articles = topic.articles.all()
    if cached_articles.exists() and not force_refresh:
        return cached_articles

    # 1. Clean and simplify the search phrase (take 2-3 essential words)
    raw_kw = topic.keywords.values_list('keyword', flat=True).first()
    if raw_kw:
        # Take the first 3 words of the keyword to keep the search focused
        search_words = [w for w in raw_kw.split() if len(w) > 2][:3]
        search_term = " ".join(search_words)
    else:
        search_term = f"{topic.title}"

    print(f"\n[DEBUG] Searching articles for keyword: '{search_term}'")

    if force_refresh:
        topic.articles.all().delete()

    discovered_count = 0

    # ---------------------------------------------------------
    # SOURCE 1: Google Educational Technical Feeds
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # SOURCE 1: Target GeeksforGeeks, W3Schools, and Top Student Sites
    # ---------------------------------------------------------
    try:
        # Specifically target GeeksforGeeks and W3Schools first
        query = f"{search_term} (site:geeksforgeeks.org OR site:w3schools.com OR site:analyticsvidhya.com)"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        resp = requests.get(rss_url, headers=BROWSER_HEADERS, timeout=7)
        if resp.status_code == 200:
            feed = feedparser.parse(resp.content)
            print(f"[DEBUG] Targeted Student Feed returned {len(feed.entries)} entries")

            for entry in feed.entries[:10]:
                title = html.unescape(entry.get('title', ''))
                link = entry.get('link', '')
                if not title or not link:
                    continue

                source_name = "Educational Resource"
                if "geeksforgeeks" in link.lower() or "geeksforgeeks" in title.lower():
                    source_name = "GeeksforGeeks"
                elif "w3schools" in link.lower() or "w3schools" in title.lower():
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
                    published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                ArticleResource.objects.update_or_create(
                    topic=topic,
                    url=link,
                    defaults={
                        'title': title,
                        'source_name': source_name,
                        'snippet': snippet or f"Educational article and guide on {search_term}.",
                        'published_date': published_date,
                    }
                )
                discovered_count += 1
    except Exception as e:
        print(f"[DEBUG] Source 1 error: {e}")

    # ---------------------------------------------------------
    # SOURCE 2: arXiv Open Computer Science Repository (Fallback & Enrichment)
    # ---------------------------------------------------------
    if discovered_count < 3:
        try:
            arxiv_query = urllib.parse.quote(search_term)
            arxiv_url = f"https://export.arxiv.org/api/query?search_query=all:{arxiv_query}&start=0&max_results=5"
            arxiv_resp = requests.get(arxiv_url, headers=BROWSER_HEADERS, timeout=7)

            if arxiv_resp.status_code == 200:
                arxiv_feed = feedparser.parse(arxiv_resp.content)
                print(f"[DEBUG] arXiv returned {len(arxiv_feed.entries)} papers")

                for entry in arxiv_feed.entries:
                    title = html.unescape(entry.get('title', '')).replace('\n', ' ')
                    link = entry.get('link', '')
                    summary = html.unescape(entry.get('summary', '')).replace('\n', ' ')

                    published_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                    ArticleResource.objects.update_or_create(
                        topic=topic,
                        url=link,
                        defaults={
                            'title': title,
                            'source_name': "arXiv Computer Science",
                            'snippet': summary[:280] + "...",
                            'published_date': published_date,
                        }
                    )
                    discovered_count += 1
        except Exception as e:
            print(f"[DEBUG] arXiv Source error: {e}")

    return topic.articles.all()