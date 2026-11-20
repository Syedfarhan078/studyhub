import os
import re
import json
import time
import requests
from django.core.management.base import BaseCommand
from django.conf import settings
from curriculum.services.vtucircle_scraper import (
    VTUCIRCLE_BASE,
    DEFAULT_HEADERS,
    parse_vtucircle_page,
    CATALOG_CACHE_PATH,
)


class Command(BaseCommand):
    help = "Bulk crawls and ingests official 5-module VTU syllabi from vtucircle.com"

    def add_arguments(self, parser):
        parser.add_argument(
            '--limit',
            type=int,
            default=60,
            help='Maximum number of subjects to crawl and ingest.'
        )
        parser.add_argument(
            '--query',
            type=str,
            default='',
            help='Filter subject slugs by keyword (e.g. "bcs", "1bcs", "bad", "bai").'
        )
        parser.add_argument(
            '--url',
            type=str,
            default='',
            help='Direct URL to crawl a single subject page.'
        )

    def handle(self, *args, **options):
        single_url = options.get('url')
        limit = options.get('limit', 60)
        query_filter = options.get('query', '').lower()

        os.makedirs(os.path.dirname(CATALOG_CACHE_PATH), exist_ok=True)
        catalog = {}
        if os.path.exists(CATALOG_CACHE_PATH):
            try:
                with open(CATALOG_CACHE_PATH, 'r', encoding='utf-8') as f:
                    catalog = json.load(f)
            except Exception:
                catalog = {}

        if single_url:
            self.stdout.write(f"Crawling single URL: {single_url}")
            try:
                r = requests.get(single_url, headers=DEFAULT_HEADERS, timeout=10)
                parsed = parse_vtucircle_page(r.text, source_url=single_url)
                if parsed and len(parsed['modules']) == 5:
                    slug = single_url.rstrip('/').split('/')[-1]
                    code_match = re.search(r'(?:[1-2]?[a-z]{2,4}\d{2,4}[a-z]?)', slug, re.IGNORECASE)
                    code = code_match.group(0).upper() if code_match else slug.upper()
                    key = code
                    catalog[key] = {
                        'canonical_name': parsed.get('page_title', slug),
                        'canonical_code': code,
                        'page_title': parsed.get('page_title', ''),
                        'aliases': [code, slug.replace('-', ' '), parsed.get('page_title', '')],
                        'source_url': single_url,
                        'modules': parsed['modules']
                    }
                    with open(CATALOG_CACHE_PATH, 'w', encoding='utf-8') as f:
                        json.dump(catalog, f, indent=2, ensure_ascii=False)
                    self.stdout.write(self.style.SUCCESS(f"Successfully ingested {key}: {parsed.get('page_title')}"))
                else:
                    self.stdout.write(self.style.WARNING("Could not extract 5 modules from URL."))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error crawling {single_url}: {e}"))
            return

        self.stdout.write("Fetching post-sitemap from vtucircle.com...")
        try:
            r = requests.get(f"{VTUCIRCLE_BASE}/post-sitemap.xml", headers=DEFAULT_HEADERS, timeout=12)
            urls = re.findall(r'<loc>([^<]+)</loc>', r.text)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to fetch sitemap: {e}"))
            return

        candidate_urls = []
        for u in urls:
            slug = u.rstrip('/').split('/')[-1].lower()
            if any(skip in slug for skip in ['-pdf', '-program-', '-written', '-textbook', '-lab', '-viva']):
                continue
            if query_filter and query_filter not in slug:
                continue
            # Look for engineering course codes
            if re.search(r'(?:bcs|bad|bai|bme|bcv|bec|bee|bis|bds|21cs|18cs|1bcs|1bcv|1bme|1bec|1bee)', slug):
                candidate_urls.append(u)

        self.stdout.write(f"Found {len(candidate_urls)} candidate subject URLs matching criteria (limit={limit}).")

        ingested_count = 0
        for idx, u in enumerate(candidate_urls[:limit], 1):
            slug = u.rstrip('/').split('/')[-1]
            try:
                res = requests.get(u, headers=DEFAULT_HEADERS, timeout=8)
                if res.status_code == 200:
                    parsed = parse_vtucircle_page(res.text, source_url=u)
                    if parsed and len(parsed.get('modules', [])) == 5:
                        code_match = re.search(r'(?:[1-2]?[a-z]{2,4}\d{2,4}[a-z]?)', slug, re.IGNORECASE)
                        code = code_match.group(0).upper() if code_match else slug.upper()
                        key = code
                        catalog[key] = {
                            'canonical_name': parsed.get('page_title', slug),
                            'canonical_code': code,
                            'page_title': parsed.get('page_title', ''),
                            'aliases': [code, slug.replace('-', ' '), parsed.get('page_title', '')],
                            'source_url': u,
                            'modules': parsed['modules']
                        }
                        ingested_count += 1
                        self.stdout.write(self.style.SUCCESS(f"[{idx}] Ingested {code}: {parsed.get('page_title')}"))
                    else:
                        self.stdout.write(f"[{idx}] Skipped (no 5 modules): {slug}")
                time.sleep(0.15)
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"[{idx}] Error on {slug}: {e}"))

        with open(CATALOG_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)

        self.stdout.write(self.style.SUCCESS(
            f"Done! Successfully ingested {ingested_count} subjects into {CATALOG_CACHE_PATH}. "
            f"Total cached subjects in catalog: {len(catalog)}"
        ))
