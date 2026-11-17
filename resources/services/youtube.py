import html
import logging
import re
import requests
from django.utils.dateparse import parse_datetime
from decouple import config
from resources.models import VideoResource

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def clean_topic_title(raw_title: str) -> str:
    """Removes leading numbers, bullets, or chapter codes for cleaner search queries."""
    if not raw_title:
        return ""
    # Strip leading bullets, numbers, or section prefixes like "1.2 ", "Topic 3: "
    cleaned = re.sub(r'^(?:Topic\s*\d+[:.\s]*|\d+(?:\.\d+)*[:.\s]*|[-*•]\s*)', '', raw_title, flags=re.IGNORECASE)
    # Remove extra quotes
    cleaned = cleaned.replace('"', '').replace("'", "").strip()
    return cleaned or raw_title.strip()


def build_educational_youtube_query(topic) -> str:
    """
    Constructs an educational, context-anchored search query.
    Binds the topic title with the parent subject name to guarantee university-level lecture results.
    Example: 'Symmetric cipher model Network Security lecture tutorial'
    """
    subject_name = ""
    if hasattr(topic, 'module') and hasattr(topic.module, 'subject') and topic.module.subject:
        subject_name = topic.module.subject.name.strip()

    cleaned_title = clean_topic_title(topic.title)

    # Check if any pre-generated keyword already contains strong academic intent
    keywords = list(topic.keywords.values_list('keyword', flat=True))
    best_keyword = None
    for kw in keywords:
        kw_lower = kw.lower()
        if any(term in kw_lower for term in ("lecture", "tutorial", "nptel", "engineering", "gate", "vtu")):
            best_keyword = kw.strip()
            break

    if best_keyword:
        query = best_keyword
    elif subject_name:
        query = f'"{cleaned_title}" {subject_name} lecture tutorial'
    else:
        query = f'"{cleaned_title}" computer science engineering lecture'

    return query


def get_or_fetch_videos_for_topic(topic, force_refresh=False):
    """
    Retrieves educational videos for a specific syllabus topic.
    
    1. Checks if videos are already cached in PostgreSQL.
    2. If cached and force_refresh is False, returns cached videos (0 API quota used!).
    3. If not cached (or force_refresh is True), queries the YouTube Data API v3
       using the context-anchored query, stores results in DB, and returns them.
    """
    cached_videos = topic.videos.all()
    if cached_videos.exists() and not force_refresh:
        return cached_videos

    api_key = config('YOUTUBE_API_KEY', default=None)
    if not api_key or api_key == 'your_key_will_go_here_later':
        logger.warning("YouTube API key is missing or not configured in .env.")
        return cached_videos

    base_query = build_educational_youtube_query(topic)
    # Add negative modifiers to exclude clickbait, entertainment, and shorts
    educational_query = f"{base_query} -shorts -song -trailer -movie -status"

    params = {
        'part': 'snippet',
        'q': educational_query,
        'type': 'video',
        'videoEmbeddable': 'true',
        'maxResults': 9,
        'relevanceLanguage': 'en',
        'safeSearch': 'moderate',
        'key': api_key,
    }

    logger.info("Querying YouTube Data API for topic '%s': '%s'", topic.title, educational_query)

    try:
        response = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=8)
        
        # Handle Quota Limit or Forbidden (403)
        if response.status_code == 403:
            logger.error("YouTube API quota exceeded or key invalid (403 Forbidden).")
            return cached_videos

        # If strict negative search returned 0 items, retry once with relaxed base query
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            if not items and educational_query != base_query:
                logger.info("Relaxing YouTube query to '%s' to expand results.", base_query)
                params['q'] = base_query
                response = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=8)
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('items', [])
        else:
            response.raise_for_status()
            data = response.json()
            items = data.get('items', [])

        if not items:
            logger.warning("YouTube returned 0 results for query: '%s'", params['q'])
            return cached_videos

        # If force refreshing, clear older cached videos for this topic
        if force_refresh:
            topic.videos.all().delete()

        # Parse and save new video records into PostgreSQL
        for item in items:
            video_id = item.get('id', {}).get('videoId')
            snippet = item.get('snippet', {})

            if not video_id:
                continue

            # YouTube API returns HTML-encoded entities like &quot; and &#39;
            clean_title = html.unescape(snippet.get('title', ''))
            clean_description = html.unescape(snippet.get('description', ''))
            clean_channel = html.unescape(snippet.get('channelTitle', ''))

            thumbnails = snippet.get('thumbnails', {})
            thumb_url = (
                thumbnails.get('high', {}).get('url') or
                thumbnails.get('medium', {}).get('url') or
                thumbnails.get('default', {}).get('url') or ''
            )

            published_at = parse_datetime(snippet.get('publishedAt', ''))

            VideoResource.objects.update_or_create(
                topic=topic,
                youtube_video_id=video_id,
                defaults={
                    'title': clean_title,
                    'channel_title': clean_channel,
                    'thumbnail_url': thumb_url,
                    'description': clean_description,
                    'published_at': published_at,
                }
            )

        return topic.videos.all()

    except requests.exceptions.RequestException as e:
        logger.error("Network error while connecting to YouTube API: %s", e)
        return cached_videos
