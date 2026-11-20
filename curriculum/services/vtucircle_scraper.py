"""
VTUCircle Syllabus Scraper and Parser Service.
Enables live search, extraction, and ingestion of official VTU 5-module syllabi
directly from vtucircle.com for 2022, 2024, and 2025 schemes.
"""

import os
import re
import json
import logging
from html import unescape
from typing import Optional, Dict, Any, List
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

VTUCIRCLE_BASE = "https://vtucircle.com"
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

CATALOG_CACHE_PATH = os.path.join(settings.BASE_DIR, 'curriculum', 'data', 'vtucircle_catalog.json')


def clean_html_to_text(html_content: str) -> str:
    """Strips HTML tags, decodes entities, and normalizes whitespace."""
    if not html_content:
        return ""
    text = re.sub(r'<[^>]+>', ' ', html_content)
    text = ' '.join(unescape(text).split())
    return text.strip()


def parse_vtucircle_page(html_text: str, source_url: str = "") -> Optional[Dict[str, Any]]:
    """
    Parses a VTUCircle post page and extracts the 5 module syllabus text.
    Handles <div class="syllabus-box-set"> and MODULE-X headers.
    """
    if not html_text:
        return None

    # Title extraction: check <h1 class="entry-title">, or <title> cleaned of site branding
    page_title = ""
    entry_title_match = re.search(r'<h1[^>]*class=[\'"][^\'"]*entry-title[^\'"]*[\'"][^>]*>([\s\S]*?)</h1>', html_text, re.IGNORECASE)
    if entry_title_match:
        page_title = clean_html_to_text(entry_title_match.group(1))
    
    if not page_title or page_title.upper() == 'VTU CIRCLE':
        title_tag = re.search(r'<title>([^<]+)</title>', html_text, re.IGNORECASE)
        if title_tag:
            clean_title = clean_html_to_text(title_tag.group(1))
            # Remove "VTU Circle »" or "- VTU Circle"
            clean_title = re.sub(r'VTU\s*Circle\s*»', '', clean_title, flags=re.IGNORECASE)
            clean_title = re.sub(r'[-–]\s*VTU\s*Circle', '', clean_title, flags=re.IGNORECASE)
            page_title = clean_title.strip()

    if not page_title or page_title.upper() == 'VTU CIRCLE':
        # Fallback to non-brand H1
        all_h1s = re.findall(r'<h1[^>]*>([\s\S]*?)</h1>', html_text, re.IGNORECASE)
        for h in all_h1s:
            cleaned = clean_html_to_text(h)
            if cleaned and cleaned.upper() != 'VTU CIRCLE' and 'MODULE' not in cleaned.upper():
                page_title = cleaned
                break

    # Strategy 1: syllabus-box-set divs (standard on VTUCircle)
    boxes = re.findall(r'<div\s+class=[\'"][^\'"]*syllabus-box-set[^\'"]*[\'"]>([\s\S]*?)</div>', html_text, re.IGNORECASE)
    if len(boxes) >= 5:
        modules = []
        for i, box in enumerate(boxes[:5], 1):
            text = clean_html_to_text(box)
            title = f"Module {i}"
            if ':' in text[:60]:
                title = text.split(':', 1)[0].strip()
            modules.append({
                'module_number': i,
                'title': title,
                'raw_text': text,
            })
        return {
            'page_title': page_title,
            'modules': modules,
            'source_url': source_url,
        }

    # Strategy 2: Splitting by MODULE-1 to MODULE-5 headings
    module_splits = re.split(r'MODULE[\s\-]*([1-5])', html_text, flags=re.IGNORECASE)
    if len(module_splits) >= 11:
        modules = []
        for idx in range(1, 11, 2):
            mod_num = int(module_splits[idx])
            content = module_splits[idx + 1]
            content_clean = re.split(r'(?:note-download-btn|download\s+pdf|related\s+posts|<footer)', content, flags=re.IGNORECASE)[0]
            text = clean_html_to_text(content_clean)
            if len(text) > 30:
                title = f"Module {mod_num}"
                if ':' in text[:60]:
                    title = text.split(':', 1)[0].strip()
                modules.append({
                    'module_number': mod_num,
                    'title': title,
                    'raw_text': text,
                })
        if len(modules) == 5:
            return {
                'page_title': page_title,
                'modules': modules,
                'source_url': source_url,
            }

    return None


def search_vtucircle(query: str, timeout: int = 6) -> List[str]:
    """
    Performs search on vtucircle.com/?s=<query>
    Returns ranked candidate URLs for syllabus pages.
    """
    if not query:
        return []

    search_url = f"{VTUCIRCLE_BASE}/?s={requests.utils.quote(query)}"
    try:
        r = requests.get(search_url, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code != 200:
            return []

        urls = re.findall(r'<h[23][^>]*class=[\'"][^\'"]*entry-title[^\'"]*[\'"][^>]*>\s*<a\s+href=[\'"]([^\'"]+)[\'"]', r.text, re.IGNORECASE)
        if not urls:
            urls = re.findall(r'<a\s+href=[\'"](https://vtucircle\.com/[^\'"]+/)[\'"]', r.text, re.IGNORECASE)

        filtered = []
        for u in urls:
            u_clean = u.split('?')[0].rstrip('/') + '/'
            if any(skip in u_clean for skip in ['/category/', '/tag/', '/author/', '/page/', '/wp-content/', '/contact', '/about', '/privacy']):
                continue
            if u_clean == f"{VTUCIRCLE_BASE}/":
                continue
            if u_clean not in filtered:
                filtered.append(u_clean)

        def score_url(url_str):
            score = 0
            if '-module-' not in url_str and '-pdf' not in url_str and '-written' not in url_str and '-program-' not in url_str:
                score += 10
            if 'syllabus' in url_str:
                score += 5
            return score

        filtered.sort(key=score_url, reverse=True)
        return filtered
    except Exception as e:
        logger.warning("VTUCircle search failed for '%s': %s", query, e)
        return []


def is_result_relevant(url: str, page_title: str, course_code: str, subject_name: str) -> bool:
    """
    Validates whether a candidate URL or parsed page actually corresponds to the requested course.
    Prevents false matches where search fallback returns completely unrelated pages (e.g. Physics for Maths).
    """
    u_lower = url.lower()
    title_lower = (page_title or "").lower()

    # Reject generic hub / category pages
    if any(hub in u_lower for hub in ['/p-and-c-cycle', '/vtu-syllabus', '/first-year', '/academic-reports', '/vtu-sgpa', '/vtu-cgpa', '/question-paper']):
        return False

    code_clean = re.sub(r'[^a-zA-Z0-9]', '', course_code).lower()
    if code_clean and len(code_clean) >= 4:
        # Check if course code is represented in URL slug or title
        if code_clean in u_lower or code_clean in title_lower:
            return True
        # Base code without leading 1 or 2 (e.g. 1BCS305 -> BCS305, 1BMATCS301 -> BMATCS301)
        base_code = re.sub(r'^[12]', '', code_clean)
        if base_code and len(base_code) >= 4 and (base_code in u_lower or base_code in title_lower):
            return True

    # If course code wasn't matched, check keyword overlap with subject name
    if subject_name:
        stopwords = {'and', 'or', 'the', 'of', 'for', 'in', 'to', 'with', 'using', 'laboratory', 'lab', 'introduction'}
        name_words = [w for w in re.findall(r'[a-zA-Z]{3,}', subject_name.lower()) if w not in stopwords]
        if name_words:
            matched_words = [w for w in name_words if w in u_lower or w in title_lower]
            min_required = 1 if len(name_words) == 1 else 2
            if len(matched_words) >= min_required:
                return True

    return False


def fetch_and_parse_vtucircle_syllabus(course_code: str = "", subject_name: str = "", timeout: int = 6) -> Optional[Dict[str, Any]]:
    """
    Finds and extracts syllabus from VTUCircle by searching for course code or subject name.
    Returns 5 module outlines with raw text. Strictly validates that the retrieved page matches the requested course.
    """
    queries = []
    if course_code:
        queries.append(course_code.strip())
    if subject_name:
        queries.append(subject_name.strip())
    if course_code and subject_name:
        queries.append(f"{course_code} {subject_name}".strip())

    visited = set()
    for q in queries:
        candidate_urls = search_vtucircle(q, timeout=timeout)
        for u in candidate_urls[:3]:
            if u in visited:
                continue
            visited.add(u)
            if not is_result_relevant(u, "", course_code, subject_name):
                continue
            try:
                r = requests.get(u, headers=DEFAULT_HEADERS, timeout=timeout)
                if r.status_code == 200:
                    parsed = parse_vtucircle_page(r.text, source_url=u)
                    if parsed and len(parsed.get('modules', [])) == 5:
                        if not is_result_relevant(u, parsed.get('page_title', ''), course_code, subject_name):
                            continue
                        # Extract course code and title heuristics
                        save_to_local_cache(
                            code=course_code,
                            name=subject_name or parsed.get('page_title', ''),
                            page_title=parsed.get('page_title', ''),
                            source_url=u,
                            modules=parsed['modules']
                        )
                        return parsed
            except Exception as e:
                logger.warning("Error fetching VTUCircle candidate %s: %s", u, e)

    return None


def save_to_local_cache(code: str, name: str, page_title: str, source_url: str, modules: List[Dict[str, Any]]):
    """Saves dynamically fetched syllabus into local JSON cache for 0ms future lookups."""
    try:
        os.makedirs(os.path.dirname(CATALOG_CACHE_PATH), exist_ok=True)
        cache_data = {}
        if os.path.exists(CATALOG_CACHE_PATH):
            with open(CATALOG_CACHE_PATH, 'r', encoding='utf-8') as f:
                try:
                    cache_data = json.load(f)
                except Exception:
                    cache_data = {}

        key = (code or name or page_title).upper().replace(' ', '_')
        cache_data[key] = {
            'canonical_name': name or page_title,
            'canonical_code': code,
            'page_title': page_title,
            'aliases': [a for a in [code, name, page_title] if a],
            'source_url': source_url,
            'modules': modules,
        }

        with open(CATALOG_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        logger.info("Saved VTUCircle syllabus for '%s' to local cache.", key)
    except Exception as e:
        logger.warning("Failed to save syllabus to local cache: %s", e)


def load_local_vtucircle_cache() -> Dict[str, Any]:
    """Loads previously crawled or dynamic cached syllabi from disk."""
    if not os.path.exists(CATALOG_CACHE_PATH):
        return {}
    try:
        with open(CATALOG_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load local VTUCircle cache: %s", e)
        return {}
